"""Minimal VCD sampler for the RvvCoreMiniAxi scalar-timing waves.

Samples the requested signals once per rising edge of the core clock, using
the value held *during* that cycle (i.e. just before the edge), which is what
a waveform viewer shows for the cycle. Cycle index k = k-th rising edge.

    v = Vcd(path, ["core.score.dispatch.io_inst_0_ready", ...])
    v.val("core.score.dispatch.io_inst_0_ready")[k]   -> int
    v.ns(k)                                            -> time of edge k in ns

Signal names are paths below the top module (RvvCoreMiniAxi.), exactly as in
the VCD scopes. x/z bits read as 0. Host-side tool, no dependencies.
"""

CLOCK = "core.score.clock"


class Vcd:
    def __init__(self, path, signals, clock=CLOCK, top="RvvCoreMiniAxi"):
        want = set(signals) | {clock}
        ids = {}  # vcd id -> [names]
        scope = []
        f = open(path, "r", errors="replace")
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "$scope":
                scope.append(t[2])
            elif t[0] == "$upscope":
                scope.pop()
            elif t[0] == "$var":
                if top in scope:
                    name = ".".join(scope[scope.index(top) + 1:] + [t[4]])
                    if name in want:
                        ids.setdefault(t[3], []).append(name)
            elif t[0] == "$enddefinitions":
                break
        found = {n for v in ids.values() for n in v}
        missing = want - found
        if missing:
            raise KeyError(f"signals not in VCD: {sorted(missing)}")
        cur = {n: 0 for n in found}
        self.samples = {n: [] for n in found}
        self.times = []
        clk_id = next(i for i, ns in ids.items() if clock in ns)
        pending = []
        time = 0

        def flush(t):
            # Apply the changes collected at time t. If the clock rises at t,
            # record the pre-change values as one cycle sample first.
            rise = any(i == clk_id and v == 1 for i, v in pending) and cur[clock] == 0
            if rise:
                self.times.append(t)
                for n, s in self.samples.items():
                    s.append(cur[n])
            for i, v in pending:
                for n in ids[i]:
                    cur[n] = v
            pending.clear()

        for line in f:
            if not line:
                continue
            c = line[0]
            if c == "#":
                flush(time)
                time = int(line[1:])
            elif c in "01xzXZ":
                i = line[1:].strip()
                if i in ids:
                    pending.append((i, 1 if c == "1" else 0))
            elif c in "bB":
                bits, i = line[1:].split()
                if i in ids:
                    bits = bits.replace("x", "0").replace("z", "0").replace("X", "0").replace("Z", "0")
                    pending.append((i, int(bits, 2)))
            # real values / $dumpvars markers are ignored
        flush(time)
        f.close()
        self.n = len(self.times)

    def val(self, name):
        return self.samples[name]

    def ns(self, k):
        return self.times[k] / 1000.0

    def period_ns(self):
        return (self.times[-1] - self.times[0]) / (self.n - 1) / 1000.0
