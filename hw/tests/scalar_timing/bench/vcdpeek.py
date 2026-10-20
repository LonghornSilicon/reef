"""Quick look into a VCD from the terminal (review-packet helper, added from the claude.ai thread).

    python3 bench/vcdpeek.py params <vcd> <name-substring>
        Print the value of every signal/parameter whose name contains the substring
        (case-insensitive), as dumped at time 0. Use it for elaborated parameters,
        e.g. `PipeRegs`.

    python3 bench/vcdpeek.py cycles <vcd> <start_ns> <end_ns> <signal> [<signal> ...]
        Print one row per clock cycle in the window: "cycle ending T" and the value each
        signal held during that cycle (what GTKWave shows). Signal paths are below
        RvvCoreMiniAxi., e.g. core.score.dvu.count. Values above 255 print in hex.
"""
import sys


def params(path, pat):
    pat = pat.lower()
    ids, scope = {}, []
    f = open(path, "r", errors="replace")
    for line in f:
        t = line.split()
        if not t:
            continue
        if t[0] == "$scope":
            scope.append(t[2])
        elif t[0] == "$upscope":
            scope.pop()
        elif t[0] == "$var" and pat in t[4].lower():
            ids.setdefault(t[3], []).append((".".join(scope[-3:] + [t[4]]), int(t[2])))
        elif t[0] == "$enddefinitions":
            break
    if not ids:
        print("no names contain", pat)
        return
    seen = {}
    for line in f:
        if line.startswith("#") and line.strip() != "#0":
            break
        t = line.split()
        if not t:
            continue
        if t[0][0] in "bB" and len(t) == 2 and t[1] in ids:
            v = t[0][1:]
        elif t[0][0] in "01xz" and t[0][1:] in ids:
            v, t = t[0][0], [None, t[0][1:]]
        else:
            continue
        seen.setdefault(t[1], v)
    for i, names in ids.items():
        v = seen.get(i, "?")
        for n, w in names:
            if v != "?" and set(v) <= set("01"):
                x = int(v, 2)
                if w > 32 and w % 32 == 0:  # packed array of 32-bit fields, MSB-first
                    fields = [(x >> (32 * k)) & 0xFFFFFFFF for k in range(w // 32)][::-1]
                    print(f"{n} = {{{','.join(map(str, fields))}}}")
                else:
                    print(f"{n} = {x}")
            else:
                print(f"{n} = {v}")


def cycles(path, start, end, sigs):
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from vcdscan import Vcd
    v = Vcd(path, sigs)
    print("cycle_end_ns  " + "  ".join(s.split(".")[-1] for s in sigs))
    for k in range(len(v.times)):
        t = v.ns(k)
        if start <= t <= end:
            vals = [v.val(s)[k] for s in sigs]
            print(f"{t:10.2f}    " + "  ".join(f"{x:#x}" if x > 255 else str(x) for x in vals))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "params":
        params(a[1], a[2])
    elif len(a) >= 5 and a[0] == "cycles":
        cycles(a[1], float(a[2]), float(a[3]), a[4:])
    else:
        print(__doc__)
