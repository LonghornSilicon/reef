"use client";

import { useCallback, useEffect, useRef, type RefObject } from "react";

export type Inset = { top: number; right: number; bottom: number; left: number };
type Cam = { x: number; y: number; k: number };

const MIN = 0.15;
const MAX = 2.5;

/**
 * Pan and zoom by writing one transform to the board element. No React state per frame:
 * dragging and wheel zoom never re-render the canvas (ux-directions.md §2).
 */
export function useCamera(viewport: RefObject<HTMLDivElement | null>, board: RefObject<HTMLDivElement | null>) {
  const cam = useRef<Cam>({ x: 0, y: 0, k: 1 });
  const dragged = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const apply = useCallback((animate: boolean) => {
    const el = board.current;
    if (!el) return;
    if (timer.current) clearTimeout(timer.current);
    el.classList.toggle("cam", animate);
    const { x, y, k } = cam.current;
    el.style.transform = `translate3d(${x}px, ${y}px, 0) scale(${k})`;
    if (animate) timer.current = setTimeout(() => el.classList.remove("cam"), 220);
  }, [board]);

  /** Fit a rectangle of board space into the free part of the viewport. */
  const fitRect = useCallback((r: { x: number; y: number; w: number; h: number }, inset: Inset, animate = true, maxK = 1.1) => {
    const vp = viewport.current;
    if (!vp) return;
    const vw = vp.clientWidth - inset.left - inset.right;
    const vh = vp.clientHeight - inset.top - inset.bottom;
    if (vw <= 0 || vh <= 0) return;
    const k = Math.min(maxK, Math.max(MIN, Math.min(vw / r.w, vh / r.h)));
    cam.current = { k, x: inset.left + (vw - r.w * k) / 2 - r.x * k, y: inset.top + (vh - r.h * k) / 2 - r.y * k };
    apply(animate);
  }, [apply, viewport]);

  const zoomAt = useCallback((factor: number, px?: number, py?: number, animate = false) => {
    const vp = viewport.current;
    if (!vp) return;
    const c = cam.current;
    const k = Math.min(MAX, Math.max(MIN, c.k * factor));
    const ox = px ?? vp.clientWidth / 2, oy = py ?? vp.clientHeight / 2;
    cam.current = { k, x: ox - ((ox - c.x) * k) / c.k, y: oy - ((oy - c.y) * k) / c.k };
    apply(animate);
  }, [apply, viewport]);

  /** Pan (animated) just enough that a board rectangle sits inside the free part of the viewport. */
  const reveal = useCallback((r: { x: number; y: number; w: number; h: number }, inset: Inset) => {
    const vp = viewport.current;
    if (!vp) return;
    const { x, y, k } = cam.current;
    const left = x + r.x * k, right = left + r.w * k, top = y + r.y * k, bottom = top + r.h * k;
    const maxX = vp.clientWidth - inset.right, maxY = vp.clientHeight - inset.bottom;
    let dx = 0, dy = 0;
    if (right > maxX) dx = maxX - right;
    if (left + dx < inset.left) dx = inset.left - left;
    if (bottom > maxY) dy = maxY - bottom;
    if (top + dy < inset.top) dy = inset.top - top;
    if (!dx && !dy) return;
    cam.current = { k, x: x + dx, y: y + dy };
    apply(true);
  }, [apply, viewport]);

  /** Board-space point → viewport pixels (for positioning floating UI). */
  const toScreen = useCallback((x: number, y: number) => ({ x: cam.current.x + x * cam.current.k, y: cam.current.y + y * cam.current.k }), []);

  useEffect(() => {
    const vp = viewport.current;
    if (!vp) return;
    let start: { x: number; y: number; cx: number; cy: number; id: number } | null = null;

    const down = (e: PointerEvent) => {
      if (e.button !== 0 || (e.target as HTMLElement).closest(".c-tools, .float, input, select, a")) return;
      start = { x: e.clientX, y: e.clientY, cx: cam.current.x, cy: cam.current.y, id: e.pointerId };
      dragged.current = false;
    };
    const move = (e: PointerEvent) => {
      if (!start || e.pointerId !== start.id) return;
      const dx = e.clientX - start.x, dy = e.clientY - start.y;
      if (!dragged.current && Math.hypot(dx, dy) < 4) return;
      if (!dragged.current) { dragged.current = true; vp.classList.add("panning"); vp.setPointerCapture(e.pointerId); }
      cam.current = { ...cam.current, x: start.cx + dx, y: start.cy + dy };
      apply(false);
    };
    const up = (e: PointerEvent) => {
      if (!start || e.pointerId !== start.id) return;
      start = null;
      vp.classList.remove("panning");
      if (vp.hasPointerCapture(e.pointerId)) vp.releasePointerCapture(e.pointerId);
      // Leave `dragged` set until the click that follows this pointerup has been ignored.
      setTimeout(() => { dragged.current = false; }, 0);
    };
    const wheel = (e: WheelEvent) => {
      if ((e.target as HTMLElement).closest(".float")) return;
      e.preventDefault();
      const r = vp.getBoundingClientRect();
      // Wheel and pinch zoom around the pointer; Shift+wheel pans sideways.
      if (e.shiftKey) {
        cam.current = { ...cam.current, x: cam.current.x - e.deltaY - e.deltaX };
        apply(false);
      } else {
        zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), e.clientX - r.left, e.clientY - r.top);
      }
    };
    vp.addEventListener("pointerdown", down);
    vp.addEventListener("pointermove", move);
    vp.addEventListener("pointerup", up);
    vp.addEventListener("pointercancel", up);
    vp.addEventListener("wheel", wheel, { passive: false });
    return () => {
      vp.removeEventListener("pointerdown", down);
      vp.removeEventListener("pointermove", move);
      vp.removeEventListener("pointerup", up);
      vp.removeEventListener("pointercancel", up);
      vp.removeEventListener("wheel", wheel);
    };
  }, [apply, viewport, zoomAt]);

  return { fitRect, zoomAt, reveal, toScreen, dragged };
}
