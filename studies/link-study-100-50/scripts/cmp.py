"""Compare two steps: python3 cmp.py <prev prefix> <new prefix> [elf prefix note]"""
import sys, metrics as M
pa, pb = sys.argv[1], sys.argv[2]
base = M.load('runs/s0_dec200.json'); F, P = base['flash'], base['psram']
for B in (1, 2, 4, 8):
    a = M.load(f'runs/{pa}_b{B}.json'); m = M.load(f'runs/{pb}_b{B}.json'); r = m['roi']
    assert m['macs'] == a['macs'], 'MAC count changed'
    print('b%d: %.2f -> %.2f total (%+.1f%%), each %.3f; util %.2f%%; share %.0f%% | flash %.1f%% psram %.1f%% | lsu %.0f%% | attn %.0f%% lm_head %.0f%% | top %s' % (
        B, B*50e6/a['roi'], B*50e6/r, (a['roi']/r-1)*100, 50e6/r, m['macs']/(r*128)*100, max(F, B*P)/r*100,
        m['flash']/r*100, m['psram']/r*100, m['busy']['lsu'], m['kernels']['attention']/r*100, m['kernels']['lm_head']/r*100,
        ', '.join('%s %.0f%%' % (n, v) for v, n in m['stalls'][:4])))
a = M.load(f'runs/{pa}_ttft32.json', 4); t = M.load(f'runs/{pb}_ttft32.json', 4)
print('ttft32: %.3f -> %.3f s (%+.1f%%), share %.0f%% | top %s' % (a['roi']/50e6, t['roi']/50e6, (t['roi']/a['roi']-1)*100,
      t['flash']/t['roi']*100, ', '.join('%s %.0f%%' % (n, v) for v, n in t['stalls'][:3])))
