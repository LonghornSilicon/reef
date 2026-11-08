"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
import { Icon } from "@/components/icons";
import { portPoint, routePath, type Box } from "@/lib/model/routing";
import type { ViewSnapshot } from "@/lib/model/view";
import { MATURITY_COLOR, type Lens } from "./lens";
import { useViewport } from "./use-viewport";

type Props = {
  snapshot: ViewSnapshot;
  focusId: string;
  selectedId: string | null;
  lens: Lens;
  onSelect(blockId: string): void;
  onEnter(blockId: string): void;
  onLeave(): void;
  canLeave: boolean;
};

type Emphasis = "active" | "dim" | "normal";

const FALLBACK_CANVAS = { width: 960, height: 620 };

/**
 * Semantic zoom.
 *
 * Canvas text scales with the view, so at fit-zoom a 15px title rasterises to
 * about 5px of grey mush. Labels are therefore sized in screen pixels and
 * converted back into canvas units, and detail is gated on how large a block
 * actually is on screen rather than on a single global zoom threshold — a
 * large composite can show its modules at the same zoom where a small one
 * cannot.
 */
const DETAIL_MIN_W = 132;
const DETAIL_MIN_H = 64;
const MIN_LABEL_PX = 10.5;
/** Screen size a nested module label renders at, whenever it renders at all. */
const MODULE_LABEL_PX = 8.8;

/**
 * SVG text neither wraps nor ellipsises, and clipping cuts glyphs mid-stroke.
 * This trims to what actually fits and marks the cut, returning null when even
 * a few characters would not fit — an unlabelled module still reads as
 * composition, a half-glyph does not.
 */
function fitLabel(text: string, available: number, fontSize: number, minChars = 4): string | null {
  const charWidth = fontSize * 0.55;
  const maxChars = Math.floor(available / charWidth);
  if (maxChars < minChars) return null;
  if (text.length <= maxChars) return text;
  return `${text.slice(0, Math.max(1, maxChars - 1)).trimEnd()}\u2026`;
}

/** Port marks sit flush to the inside of their owning block edge. */
function portMark(side: string, rect: { w: number; h: number }, offset: number) {
  const span = 14;
  const thickness = 3;
  if (side === "left") return { x: 0, y: rect.h * offset - span / 2, width: thickness, height: span };
  if (side === "right") return { x: rect.w - thickness, y: rect.h * offset - span / 2, width: thickness, height: span };
  if (side === "top") return { x: rect.w * offset - span / 2, y: 0, width: span, height: thickness };
  return { x: rect.w * offset - span / 2, y: rect.h - thickness, width: span, height: thickness };
}


/**
 * One level of a composite's children, mapped from its own authored canvas
 * into the header-less area of the parent node.
 *
 * Showing the insides is what makes a composite readable without entering it,
 * and it mirrors how the incumbent viewer drew nested blocks. Only one level
 * is drawn: deeper nesting stops being legible and starts being texture.
 */
function NodeInternals({
  snapshot,
  blockId,
  rect,
  headerHeight,
  selectedId,
  scale,
  onSelect,
}: {
  snapshot: ViewSnapshot;
  blockId: string;
  rect: { w: number; h: number };
  headerHeight: number;
  selectedId: string | null;
  /** Current view scale, so labels can be dropped before they turn to mush. */
  scale: number;
  onSelect(id: string): void;
}) {
  const layout = snapshot.layouts[blockId];
  if (!layout) return null;

  const padX = 9;
  const padBottom = 9;
  const innerW = rect.w - padX * 2;
  const innerH = rect.h - headerHeight - padBottom;
  if (innerW < 24 || innerH < 18) return null;

  const sx = innerW / layout.canvas.width;
  const sy = innerH / layout.canvas.height;

  return (
    <g className="node-internals" transform={`translate(${padX} ${headerHeight})`}>
      {Object.entries(layout.nodes).map(([childId, child]) => {
        const block = snapshot.blocks[childId];
        if (!block) return null;
        const w = child.w * sx;
        const h = child.h * sy;

        // Labels are sized in screen pixels, then converted back into canvas
        // units, so a module name is either legible or absent — never mush.
        const labelSize = MODULE_LABEL_PX / scale;
        const label = h * scale >= 15 ? fitLabel(block.name, w - 9, labelSize, 3) : null;

        return (
          <g
            key={childId}
            className={`internal-node${childId === selectedId ? " selected" : ""}${block.childIds.length ? " composite" : ""}${label ? "" : " unlabelled"}`}
            transform={`translate(${child.x * sx} ${child.y * sy})`}
            role="button"
            tabIndex={-1}
            aria-label={block.name}
            onClick={(event) => {
              event.stopPropagation();
              onSelect(childId);
            }}
          >
            <rect width={w} height={h} rx={1.5} />
            <title>{block.name}</title>
            {label && (
              <text x={5} y={labelSize * 0.82 + 6} style={{ fontSize: labelSize }}>
                {label}
              </text>
            )}
          </g>
        );
      })}
    </g>
  );
}

export function DiagramCanvas({ snapshot, focusId, selectedId, lens, onSelect, onEnter, onLeave, canLeave }: Props) {
  const layout = snapshot.layouts[focusId];
  const canvas = layout?.canvas ?? FALLBACK_CANVAS;
  const viewport = useViewport(canvas, focusId);
  const { frameRef, view, panBy, zoomBy, fit, revealRect } = viewport;
  const fittedFocus = useRef<string | null>(null);

  const nodes = useMemo(() => {
    if (!layout) return [];
    return Object.entries(layout.nodes)
      .map(([id, rect]) => ({ id, rect, block: snapshot.blocks[id] }))
      .filter((entry) => Boolean(entry.block))
      .sort((a, b) => a.rect.y - b.rect.y || a.rect.x - b.rect.x);
  }, [layout, snapshot.blocks]);

  // Routing runs Dijkstra per edge, so it is memoised per canvas. Edge geometry
  // depends only on the authored layout, never on selection or lens.
  const edges = useMemo(() => {
    if (!layout) return [];
    const boxes: Box[] = Object.values(layout.nodes);
    return snapshot.connections
      .filter((connection) => layout.nodes[connection.from] && layout.nodes[connection.to])
      .map((connection) => {
        const fromPort = snapshot.ports[connection.from]?.find((port) => port.id === connection.fromPort);
        const toPort = snapshot.ports[connection.to]?.find((port) => port.id === connection.toPort);
        if (!fromPort || !toPort) return null;
        const fromBox = layout.nodes[connection.from];
        const toBox = layout.nodes[connection.to];
        return {
          connection,
          color: snapshot.protocols[connection.protocol]?.color ?? "var(--muted)",
          d: routePath(
            portPoint(fromBox, fromPort.side, fromPort.offset),
            portPoint(toBox, toPort.side, toPort.offset),
            fromPort.side,
            toPort.side,
            {
              obstacles: boxes.filter((box) => box !== fromBox && box !== toBox),
              width: layout.canvas.width,
              height: layout.canvas.height,
            },
          ),
        };
      })
      .filter((edge): edge is NonNullable<typeof edge> => Boolean(edge));
  }, [layout, snapshot.connections, snapshot.ports, snapshot.protocols]);

  const stepBlocks = useMemo(() => {
    if (lens.kind !== "sequence") return null;
    const sequence = snapshot.sequences.find((item) => item.id === lens.sequenceId);
    if (!sequence) return null;
    const index = Math.min(lens.step, sequence.steps.length - 1);
    return new Set(sequence.steps[index]?.blocks ?? []);
  }, [lens, snapshot.sequences]);

  const nodeEmphasis = useCallback(
    (blockId: string): Emphasis => {
      if (lens.kind === "protocol") {
        return (snapshot.ports[blockId] ?? []).some((port) => port.protocol === lens.protocol) ? "active" : "dim";
      }
      if (stepBlocks) return stepBlocks.has(blockId) ? "active" : "dim";
      return "normal";
    },
    [lens, snapshot.ports, stepBlocks],
  );

  const edgeEmphasis = useCallback(
    (protocol: string, from: string, to: string): Emphasis => {
      if (lens.kind === "protocol") return protocol === lens.protocol ? "active" : "dim";
      if (stepBlocks) return stepBlocks.has(from) && stepBlocks.has(to) ? "active" : "dim";
      return "normal";
    },
    [lens, stepBlocks],
  );

  // Arrows pan when the frame itself holds focus; a focused node handles its
  // own Enter and lets everything else bubble up to here.
  const onKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      const step = event.shiftKey ? 60 : 18;
      const actions: Record<string, () => void> = {
        ArrowLeft: () => panBy(step, 0),
        ArrowRight: () => panBy(-step, 0),
        ArrowUp: () => panBy(0, step),
        ArrowDown: () => panBy(0, -step),
        "+": () => zoomBy(1.2),
        "=": () => zoomBy(1.2),
        "-": () => zoomBy(1 / 1.2),
        _: () => zoomBy(1 / 1.2),
        "0": () => fit(),
      };
      const action = actions[event.key];
      if (!action) return;
      event.preventDefault();
      action();
    },
    [panBy, zoomBy, fit],
  );

  // Bring the selection into view when it arrives from the tree, a connection
  // link, or a deep link rather than from a click on the canvas.
  //
  // Skipped on the render where the canvas changed: fit() already frames the
  // whole canvas there, and centring as well would race it using a viewport
  // from before the fit.
  useEffect(() => {
    const changedCanvas = fittedFocus.current !== focusId;
    fittedFocus.current = focusId;
    if (changedCanvas || !selectedId) return;
    const rect = layout?.nodes[selectedId];
    if (!rect) return;
    revealRect(rect);
    // Keyed on selection and focus only: re-running on every viewport change
    // would fight the user's own panning.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, focusId]);

  if (!layout) {
    return (
      <div className="diagram-shell">
        <div className="diagram-frame">
          <p className="diagram-missing">This block has no authored canvas. Its record is in the inspector.</p>
        </div>
      </div>
    );
  }

  const showPortLabels = view.k >= 0.72;
  const focusBlock = snapshot.blocks[focusId];

  return (
    <div className="diagram-shell">
      <div
        ref={frameRef}
        className={`diagram-frame${viewport.panning ? " panning" : ""}`}
        tabIndex={0}
        role="application"
        aria-label={`Architecture diagram for ${focusBlock?.name ?? focusId}. Tab moves between blocks, arrow keys pan, plus and minus zoom, zero fits.`}
        onKeyDown={onKeyDown}
        onPointerDown={viewport.onPointerDown}
        onPointerMove={viewport.onPointerMove}
        onPointerUp={viewport.onPointerUp}
        onPointerCancel={viewport.onPointerUp}
      >
        <svg className="diagram-svg" width="100%" height="100%" role="presentation">
          <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
            <rect className="diagram-ground" x={0} y={0} width={canvas.width} height={canvas.height} rx={4} />

            <g className="edge-layer">
              {edges.map(({ connection, d, color }) => {
                const emphasis = edgeEmphasis(connection.protocol, connection.from, connection.to);
                const linked = selectedId === connection.from || selectedId === connection.to;
                return (
                  <g key={connection.id} className={`edge emphasis-${emphasis}${linked ? " linked" : ""}`}>
                    <path className="edge-line" d={d} style={{ stroke: color }} />
                    <title>{`${snapshot.blocks[connection.from]?.name} → ${snapshot.blocks[connection.to]?.name} · ${connection.label}`}</title>
                  </g>
                );
              })}
            </g>

            <g className="node-layer">
              {nodes.map(({ id, rect, block }) => {
                const emphasis = nodeEmphasis(id);
                const selected = id === selectedId;
                const enterable = Boolean(snapshot.layouts[id]) && block.childIds.length > 0;
                const titleSize = Math.max(8, Math.min(15, Math.min(rect.w / 12, rect.h / 3.2)));
                // Keep the name legible when zoomed out. The header grows with
                // it so the modules below are never overlapped by the title.
                const renderedTitle = Math.max(titleSize, MIN_LABEL_PX / view.k);
                // A top-side port draws its label along the upper edge, so the
                // title drops below it rather than colliding with it.
                const topPortInset = (snapshot.ports[id] ?? []).some((port) => port.side === "top") ? 13 : 0;
                const headerHeight = renderedTitle + 16 + topPortInset;
                const hasInternals =
                  enterable &&
                  rect.w * view.k > DETAIL_MIN_W &&
                  rect.h * view.k > DETAIL_MIN_H &&
                  rect.h - headerHeight > 20;
                return (
                  <g
                    key={id}
                    className={`diagram-node emphasis-${emphasis}${selected ? " selected" : ""}`}
                    data-diagram-node={id}
                    transform={`translate(${rect.x} ${rect.y})`}
                    tabIndex={0}
                    role="button"
                    aria-label={`${block.name}, ${block.kind}, ${block.maturity}${enterable ? ", composite, Alt+Enter to open" : ""}`}
                    aria-pressed={selected}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelect(id);
                    }}
                    onDoubleClick={(event) => {
                      event.stopPropagation();
                      if (enterable) onEnter(id);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && event.altKey) {
                        if (!enterable) return;
                        event.preventDefault();
                        event.stopPropagation();
                        onEnter(id);
                        return;
                      }
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        event.stopPropagation();
                        onSelect(id);
                      }
                    }}
                  >
                    <rect className="node-body" width={rect.w} height={rect.h} rx={2} />
                    {/* Selection marks the header band, not the whole body: filling a
                        composite with ink would bury the internals it exists to show. */}
                    {selected && (
                      <rect className="node-selection" width={rect.w} height={Math.min(rect.h, headerHeight - 4)} rx={2} />
                    )}
                    {lens.kind === "readiness" && (
                      <rect className="node-readiness" width={rect.w} height={3} style={{ fill: MATURITY_COLOR[block.maturity] }} />
                    )}
                    {/* SVG text neither wraps nor ellipsises, so a long name is
                        clipped to its own block rather than bleeding across the canvas. */}
                    <text className="node-title" x={9} y={renderedTitle * 0.82 + 9 + topPortInset} style={{ fontSize: renderedTitle }}>
                      {fitLabel(block.name, rect.w - 26, renderedTitle, 3) ?? ""}
                    </text>
                    {hasInternals && <line className="node-rule" x1={0} y1={headerHeight - 4} x2={rect.w} y2={headerHeight - 4} />}
                    {hasInternals && (
                      <NodeInternals
                        snapshot={snapshot}
                        blockId={id}
                        rect={rect}
                        headerHeight={headerHeight}
                        selectedId={selectedId}
                        scale={view.k}
                        onSelect={onSelect}
                      />
                    )}
                    {enterable && (
                      <g className="node-enter" transform={`translate(${rect.w - 17} 6)`} aria-hidden="true">
                        <path d="M0 3.5 3.5 7 0 10.5" />
                      </g>
                    )}
                    {(snapshot.ports[id] ?? []).map((port) => {
                      const color = snapshot.protocols[port.protocol]?.color ?? "var(--muted)";
                      const mark = portMark(port.side, rect, port.offset);
                      const horizontal = port.side === "left" || port.side === "right";
                      const dim = lens.kind === "protocol" && port.protocol !== lens.protocol;
                      return (
                        <g key={port.id} className={`node-port${dim ? " dim" : ""}`}>
                          <rect {...mark} style={{ fill: color }} />
                          {showPortLabels && (
                            <text
                              className="port-label"
                              x={horizontal ? (port.side === "left" ? 8 : rect.w - 8) : rect.w * port.offset}
                              y={horizontal ? rect.h * port.offset + 3 : port.side === "top" ? 13 : rect.h - 7}
                              textAnchor={horizontal ? (port.side === "left" ? "start" : "end") : "middle"}
                            >
                              {port.label}
                            </text>
                          )}
                        </g>
                      );
                    })}
                  </g>
                );
              })}
            </g>
          </g>
        </svg>
      </div>

      <div className="diagram-dock" role="group" aria-label="Diagram controls">
        <button type="button" onClick={onLeave} disabled={!canLeave} title="Up one level (Alt+Up)" aria-label="Up one level">
          <Icon name="level-up" />
        </button>
        <span className="dock-rule" aria-hidden="true" />
        <button type="button" onClick={() => zoomBy(1 / 1.2)} aria-label="Zoom out">
          <Icon name="zoomOut" />
        </button>
        <output>{Math.round(view.k * 100)}%</output>
        <button type="button" onClick={() => zoomBy(1.2)} aria-label="Zoom in">
          <Icon name="zoomIn" />
        </button>
        <button type="button" onClick={fit} title="Fit to view (0)" aria-label="Fit diagram to view">
          <Icon name="fit" />
        </button>
      </div>
    </div>
  );
}
