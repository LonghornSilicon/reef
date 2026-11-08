# dist-coral: GPT-Neo on N Coral chips

Megatron tensor-parallel firmware: each chip runs one rank of
`dist_inference::TpGptNeo` (`../dist-reference/`) with coral's kernels, and
the host combines the ranks at every sync. It is to `dist-reference/` what
`coral/` is to `reference/`, and it must match it **bit for bit**.

```
include/dist_coral/
  tp_blob.hpp      byte layout of one rank's share of the weights
  tp_host_if.hpp   the mailbox protocol: one mpause per sync
firmware/          tp_rank.cc: one rank, stitched from coral/kernels/
host/pack_rank.hpp cuts a rank's share with dist_inference's slicing
sim/multi_sim.cpp  the host: N FuncSims, embedding, rank-order sums, checks
tools/             build_sim.sh (multi_sim), test.sh (Tests A and C)
```

## How a step runs

The host embeds the token and writes it to every chip. Per layer, each chip
normalizes the whole residual stream, attends its own heads, and projects
their context through its input rows of the attention output. It pauses
with that partial; the host sums the partials in rank order, adds the bias
once (`sum_in_rank_order`), writes the sum back and releases the chips. The
MLP does the same with its columns of `c_fc` and rows of `c_proj`. At the
end each chip computes its vocabulary rows of the lm_head and the host
concatenates them. That is 2 syncs per layer plus 1, as `TpGptNeo::syncs()`
records.

A sync is `mpause`: the chip halts, and the host reads and writes its memory
and resumes it. In the simulator this uses `FuncSim::resume()` and
`FuncSim::getHostMemory()`. On the chip it assumes the host can release a
core paused by `mpause`; that is not yet checked against the RTL.

The rank blob and arena sit in a memory region past the highmem map
(`0x80000000`), which multi_sim adds to each chip. The functional model
needs only that the firmware can address it; a real chip needs the external
memory that coral's README already lists as missing.

## Tests

`multi_sim` feeds a prompt a token at a time and then the chips' own greedy
tokens, and checks the logits of every token:

- **Test A:** equal to `TpGptNeo`'s bit for bit. Any difference fails.
- **Test C:** within Test B's tolerance of one chip (`GPTNeo`), fed the same
  tokens: 1e-5 of the largest logit in fp32, 5% in W8A8.

On coral's small random model (4 layers, 4 heads, 20 tokens), measured
7 Nov 2026:

| Precision | Chips | Test A | Test C max rel error | Instructions, all chips |
|---|---|---|---|---|
| fp32 | 1 | pass | 0 | 886,869 |
| fp32 | 2 | pass | 3.9e-7 | 953,386 |
| fp32 | 4 | pass | 4.6e-7 | 1,226,388 |
| W8A8 | 1 | pass | 0 | 954,069 |
| W8A8 | 2 | pass | 1.7% | 1,061,482 |
| W8A8 | 4 | pass | 1.5% | 1,421,844 |

Every run generates the same 20 tokens. Reversing the host's sum order fails
Test A in fp32 at 4 chips on the first token, by one unit in the last place.

## Running it

`multi_sim` needs Linux (`FuncSim` includes `<elf.h>`). From
`inference-engine/` on Linux:

```sh
coral/tools/setup_sim.sh build/coral-sim-linux   # Spike and coral_sim
dist-coral/tools/test.sh build/coral-sim-linux
```

On macOS, build the firmware on the host (`coral/tools/build.sh dist`), then
run `setup_sim.sh`, `build_sim.sh` and `multi_sim` in an `ubuntu:24.04`
container with `build-essential` and `device-tree-compiler`.

## Next

- Per-sync goldens, so a Test A failure names the rank and layer.
- A real checkpoint: TinyStories-28M in W8A8 is about 16 MB per chip at 8
  chips, which multi_sim can map, but Spike runs it slowly.
- Timing: the same protocol on reef_perf, with a cost per sync for the links.
