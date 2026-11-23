"""KV cache base and size for a firmware build: the layout formulas of npu.hpp.

Usage: kvbase.py <hidden> <layers> <vocab> <row_bytes per 64 weights: 64 int8, 32 int4> [stories] [max_context]
"""
import sys
H, L, V, RB = (int(x) for x in sys.argv[1:5])
S = int(sys.argv[5]) if len(sys.argv) > 5 else 8
CTX = int(sys.argv[6]) if len(sys.argv) > 6 else 256
a64 = lambda b: (b + 63) & ~63
lin = lambda i, o: o // 64 * (i * RB + 256)
VP = (V + 63) // 64 * 64
tok = 0x20000000
pos = a64(tok + V * H)
q = 4 * H; k = q + lin(H, H); v = k + lin(H, H); o = v + lin(H, H); f1 = o + lin(H, H); f2 = f1 + lin(H, 4 * H)
layer = a64(f2 + lin(4 * H, H))
layers = a64(pos + 2048 * H)
fin = layers + L * layer
lm = a64(fin + 2 * H)
kv = a64(lm + lin(H, VP))
cache = L * 16 * (CTX // 16) * (16 * (H // 16))
weights = L * layer - L * 4 * H + lin(H, VP)
print(f"kv_base=0x{kv:08X} kv_mb={max(1, -(-(S * 2 * cache) // (1 << 20)))} linear_and_lm_head_bytes={weights} lm_head_bytes={lin(H, VP)} story_kv_bytes={2*cache}")
