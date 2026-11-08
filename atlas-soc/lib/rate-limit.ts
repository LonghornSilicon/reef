type Bucket = { count: number; resetsAt: number };

declare global {
  var __atlasRateLimits: Map<string, Bucket> | undefined;
}

const buckets = globalThis.__atlasRateLimits ?? new Map<string, Bucket>();
globalThis.__atlasRateLimits = buckets;

export function takeRateLimit(key: string, limit: number, windowMs: number) {
  const timestamp = Date.now();
  const current = buckets.get(key);
  if (!current || current.resetsAt <= timestamp) {
    buckets.set(key, { count: 1, resetsAt: timestamp + windowMs });
    return { allowed: true, remaining: limit - 1, resetsAt: timestamp + windowMs };
  }
  if (current.count >= limit) return { allowed: false, remaining: 0, resetsAt: current.resetsAt };
  current.count += 1;
  return { allowed: true, remaining: limit - current.count, resetsAt: current.resetsAt };
}
