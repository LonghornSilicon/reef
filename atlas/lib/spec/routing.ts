/**
 * Orthogonal obstacle-avoiding edge routing.
 *
 * Ported from site/app.js (the incumbent viewer) rather than reinvented: the
 * authored layouts in architecture/soc.yaml were positioned against this
 * router's behaviour, so changing the geometry would change how every
 * reviewed diagram reads. Kept pure so it runs on the server or the client.
 */

import type { Side } from "./v1-schema";

export type Point = { x: number; y: number };
export type Box = { x: number; y: number; w: number; h: number };

const CLEARANCE = 12;
const TURN_COST = 18;
const EDGE_INSET = 4;

type Obstacle = { left: number; right: number; top: number; bottom: number };

export function portPoint(box: Box, side: Side, offset: number): Point {
  if (side === "left") return { x: box.x, y: box.y + box.h * offset };
  if (side === "right") return { x: box.x + box.w, y: box.y + box.h * offset };
  if (side === "top") return { x: box.x + box.w * offset, y: box.y };
  return { x: box.x + box.w * offset, y: box.y + box.h };
}

function offsetPoint(point: Point, side: Side, distance: number): Point {
  if (side === "left") return { x: point.x - distance, y: point.y };
  if (side === "right") return { x: point.x + distance, y: point.y };
  if (side === "top") return { x: point.x, y: point.y - distance };
  return { x: point.x, y: point.y + distance };
}

function segmentBlocked(a: Point, b: Point, obstacles: Obstacle[]): boolean {
  const epsilon = 0.01;
  if (Math.abs(a.y - b.y) < epsilon) {
    const left = Math.min(a.x, b.x);
    const right = Math.max(a.x, b.x);
    return obstacles.some(
      (rect) => a.y > rect.top + epsilon && a.y < rect.bottom - epsilon && right > rect.left + epsilon && left < rect.right - epsilon,
    );
  }
  const top = Math.min(a.y, b.y);
  const bottom = Math.max(a.y, b.y);
  return obstacles.some(
    (rect) => a.x > rect.left + epsilon && a.x < rect.right - epsilon && bottom > rect.top + epsilon && top < rect.bottom - epsilon,
  );
}

function compressRoute(points: Point[]): Point[] {
  const compact: Point[] = [];
  for (const point of points) {
    const previous = compact.at(-1);
    if (!previous || previous.x !== point.x || previous.y !== point.y) compact.push(point);
  }
  return compact.filter((point, index) => {
    if (index === 0 || index === compact.length - 1) return true;
    const before = compact[index - 1];
    const after = compact[index + 1];
    return !((before.x === point.x && point.x === after.x) || (before.y === point.y && point.y === after.y));
  });
}

function straightPath(start: Point, startExit: Point, endExit: Point, end: Point): string {
  return `M ${start.x} ${start.y} L ${startExit.x} ${startExit.y} L ${endExit.x} ${endExit.y} L ${end.x} ${end.y}`;
}

export function routePath(
  start: Point,
  end: Point,
  fromSide: Side,
  toSide: Side,
  options: { obstacles?: Box[]; width: number; height: number },
): string {
  const startExit = offsetPoint(start, fromSide, CLEARANCE + 2);
  const endExit = offsetPoint(end, toSide, CLEARANCE + 2);
  const { width, height } = options;

  const obstacles: Obstacle[] = (options.obstacles ?? []).map((box) => ({
    left: box.x - CLEARANCE,
    right: box.x + box.w + CLEARANCE,
    top: box.y - CLEARANCE,
    bottom: box.y + box.h + CLEARANCE,
  }));

  const xs = new Set<number>([startExit.x, endExit.x, EDGE_INSET, width - EDGE_INSET]);
  const ys = new Set<number>([startExit.y, endExit.y, EDGE_INSET, height - EDGE_INSET]);
  for (const rect of obstacles) {
    xs.add(Math.max(EDGE_INSET, rect.left));
    xs.add(Math.min(width - EDGE_INSET, rect.right));
    ys.add(Math.max(EDGE_INSET, rect.top));
    ys.add(Math.min(height - EDGE_INSET, rect.bottom));
  }

  const xValues = [...xs].filter((value) => value >= 0 && value <= width).sort((a, b) => a - b);
  const yValues = [...ys].filter((value) => value >= 0 && value <= height).sort((a, b) => a - b);

  const points: Point[] = [];
  const pointIndex = new Map<string, number>();
  const blocked = (point: Point) =>
    obstacles.some((rect) => point.x > rect.left && point.x < rect.right && point.y > rect.top && point.y < rect.bottom);

  for (const y of yValues) {
    for (const x of xValues) {
      const point = { x, y };
      if (blocked(point)) continue;
      pointIndex.set(`${x}|${y}`, points.length);
      points.push(point);
    }
  }

  const startIndex = pointIndex.get(`${startExit.x}|${startExit.y}`);
  const endIndex = pointIndex.get(`${endExit.x}|${endExit.y}`);
  if (startIndex === undefined || endIndex === undefined) return straightPath(start, startExit, endExit, end);

  type Neighbor = { index: number; distance: number; direction: "H" | "V" };
  const neighbors: Neighbor[][] = Array.from({ length: points.length }, () => []);
  const connectLine = (indexes: number[]) => {
    indexes.sort((left, right) => points[left].x - points[right].x || points[left].y - points[right].y);
    for (let index = 1; index < indexes.length; index += 1) {
      const aIndex = indexes[index - 1];
      const bIndex = indexes[index];
      const a = points[aIndex];
      const b = points[bIndex];
      if (segmentBlocked(a, b, obstacles)) continue;
      const distance = Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
      const direction = a.x === b.x ? "V" : "H";
      neighbors[aIndex].push({ index: bIndex, distance, direction });
      neighbors[bIndex].push({ index: aIndex, distance, direction });
    }
  };

  const rows = new Map<number, number[]>();
  const columns = new Map<number, number[]>();
  points.forEach((point, index) => {
    (rows.get(point.y) ?? rows.set(point.y, []).get(point.y)!).push(index);
    (columns.get(point.x) ?? columns.set(point.x, []).get(point.x)!).push(index);
  });
  for (const indexes of rows.values()) connectLine(indexes);
  for (const indexes of columns.values()) connectLine(indexes);

  const startKey = `${startIndex}|N`;
  const distances = new Map<string, number>([[startKey, 0]]);
  const previous = new Map<string, string>();
  const queue: Array<{ key: string; index: number; direction: "H" | "V" | "N"; cost: number }> = [
    { key: startKey, index: startIndex, direction: "N", cost: 0 },
  ];
  let finalKey: string | null = null;

  while (queue.length) {
    queue.sort((left, right) => left.cost - right.cost);
    const current = queue.shift()!;
    if (current.cost !== distances.get(current.key)) continue;
    if (current.index === endIndex) {
      finalKey = current.key;
      break;
    }
    for (const next of neighbors[current.index]) {
      const turn = current.direction !== "N" && current.direction !== next.direction ? TURN_COST : 0;
      const cost = current.cost + next.distance + turn;
      const key = `${next.index}|${next.direction}`;
      if (cost >= (distances.get(key) ?? Infinity)) continue;
      distances.set(key, cost);
      previous.set(key, current.key);
      queue.push({ key, index: next.index, direction: next.direction, cost });
    }
  }

  if (!finalKey) return straightPath(start, startExit, endExit, end);

  const routed: Point[] = [];
  for (let key: string | undefined = finalKey; key; key = previous.get(key)) {
    routed.unshift(points[Number(key.split("|")[0])]);
  }
  return compressRoute([start, startExit, ...routed, endExit, end])
    .map((point, index) => `${index ? "L" : "M"} ${point.x} ${point.y}`)
    .join(" ");
}
