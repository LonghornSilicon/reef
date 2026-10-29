"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type Viewport = { x: number; y: number; k: number };

// Low enough that a 2000px-wide authored canvas can still be fitted whole on a
// narrow screen. Seeing all of it small beats seeing a legible corner of it.
const MIN_SCALE = 0.1;
const MAX_SCALE = 3;
const PAD = 28;

function clampScale(k: number) {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, k));
}

/**
 * Pan/zoom state for the diagram.
 *
 * The canvas is sized by its container, never by the viewport height: that is
 * what lets the page scroll normally while the diagram still fills the space
 * it is given. Zoom-to-fit is recomputed whenever the authored canvas changes
 * so entering a composite never leaves you looking at empty space.
 */
export function useViewport(canvas: { width: number; height: number }, focusKey: string) {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const [view, setView] = useState<Viewport>({ x: 0, y: 0, k: 1 });
  const [panning, setPanning] = useState(false);

  const fit = useCallback(() => {
    const frame = frameRef.current;
    if (!frame) return;
    const { clientWidth, clientHeight } = frame;
    if (!clientWidth || !clientHeight) return;
    const k = clampScale(Math.min((clientWidth - PAD * 2) / canvas.width, (clientHeight - PAD * 2) / canvas.height));
    setView({
      k,
      x: (clientWidth - canvas.width * k) / 2,
      y: (clientHeight - canvas.height * k) / 2,
    });
  }, [canvas.width, canvas.height]);

  // Re-fit when the authored canvas changes (entering or leaving a composite).
  useEffect(() => {
    fit();
  }, [fit, focusKey]);

  // Re-fit on container resize so the diagram never ends up mostly off-screen.
  useEffect(() => {
    const frame = frameRef.current;
    if (!frame || typeof ResizeObserver === "undefined") return;
    let first = true;
    const observer = new ResizeObserver(() => {
      if (first) {
        first = false;
        return;
      }
      fit();
    });
    observer.observe(frame);
    return () => observer.disconnect();
  }, [fit]);

  const zoomBy = useCallback((factor: number, origin?: { x: number; y: number }) => {
    setView((current) => {
      const k = clampScale(current.k * factor);
      if (k === current.k) return current;
      const frame = frameRef.current;
      const point = origin ?? (frame ? { x: frame.clientWidth / 2, y: frame.clientHeight / 2 } : { x: 0, y: 0 });
      const ratio = k / current.k;
      return { k, x: point.x - (point.x - current.x) * ratio, y: point.y - (point.y - current.y) * ratio };
    });
  }, []);

  const panBy = useCallback((dx: number, dy: number) => {
    setView((current) => ({ ...current, x: current.x + dx, y: current.y + dy }));
  }, []);

  // Non-passive wheel listener: React's onWheel is passive, so preventDefault
  // there is ignored and the page would scroll while zooming.
  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;
    const onWheel = (event: WheelEvent) => {
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        const rect = frame.getBoundingClientRect();
        zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12, { x: event.clientX - rect.left, y: event.clientY - rect.top });
        return;
      }
      // Trackpad two-finger gestures pan the diagram rather than the document.
      event.preventDefault();
      panBy(-event.deltaX, -event.deltaY);
    };
    frame.addEventListener("wheel", onWheel, { passive: false });
    return () => frame.removeEventListener("wheel", onWheel);
  }, [zoomBy, panBy]);

  const onPointerDown = useCallback((event: React.PointerEvent) => {
    if (event.button !== 0 && event.button !== 1) return;
    const target = event.target as HTMLElement;
    if (target.closest("[data-diagram-node]")) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setPanning(true);
  }, []);

  const onPointerMove = useCallback(
    (event: React.PointerEvent) => {
      if (!panning) return;
      panBy(event.movementX, event.movementY);
    },
    [panning, panBy],
  );

  const onPointerUp = useCallback((event: React.PointerEvent) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setPanning(false);
  }, []);

  /**
   * Bring a canvas-space rect to the middle of the frame if it is not already
   * visible. Reads the viewport through the updater rather than a closure, so
   * it always sees the result of a fit queued earlier in the same flush.
   */
  const revealRect = useCallback((rect: { x: number; y: number; w: number; h: number }, margin = 48) => {
    const frame = frameRef.current;
    if (!frame) return;
    setView((current) => {
      const cx = (rect.x + rect.w / 2) * current.k + current.x;
      const cy = (rect.y + rect.h / 2) * current.k + current.y;
      const visible = cx > margin && cy > margin && cx < frame.clientWidth - margin && cy < frame.clientHeight - margin;
      if (visible) return current;
      return { ...current, x: current.x + frame.clientWidth / 2 - cx, y: current.y + frame.clientHeight / 2 - cy };
    });
  }, []);

  return { frameRef, view, panning, fit, zoomBy, panBy, revealRect, onPointerDown, onPointerMove, onPointerUp };
}
