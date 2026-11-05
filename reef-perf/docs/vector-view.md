# Vector view

`python -m reef_perf.vector_view TRACE.tsv` turns a `reef_perf --trace` file
into one HTML page with two linked panels: which instruction occupies each
vector lane over time, and how many command-queue credits are free. Open the
page in any browser; it has no server and loads nothing, so it works over WSL
or from a file copied to another machine. The matrix engine is not shown.

```sh
build/bin/reef_perf --elf prog.elf --trace prog.tsv
python -m reef_perf.vector_view prog.tsv -o prog.html
```

Scroll zooms, dragging pans, a click places the cursor, space plays the run.
Hovering a bar gives its queue, issue and result-ready cycles. The panel
below the charts lists what each lane is executing and what waits in the
queue at the cursor.

## How lanes are recovered

The trace has no lane index or occupancy, and the vector module doesn't
record them, so the view rebuilds both from the model:

- Occupancy is `result_ready - issue + 2 - vec_latency`, less
  `vec_to_scalar_latency` for vector-to-scalar moves.
- The lane is the lowest-numbered one that frees first, in the order
  instructions reach the vector queue, as `ResourcePool::reserve` picks it.

Replaying those bookings must reproduce every traced issue cycle. The page
reports how many instructions it could not confirm; a run that reports any
was probably made with a different config than `--config` (default
`configs/machines/m3.yaml`). Free credits are the config's `vec_queue_entries`
minus the instructions between their queue and issue events.

## Demo program

The microbenchmarks never back up the vector queue, so
`python/reef_perf/vector_view/demo/vec_burst.S` issues bursts of independent
LMUL=4 adds and multiplies. Assemble it like a microbenchmark:

```sh
riscv64-linux-gnu-as -march=rv32imf_zicsr_zifencei_zve32f -mabi=ilp32f \
    -I microbenchmarks python/reef_perf/vector_view/demo/vec_burst.S -o vb.o
riscv64-linux-gnu-ld -m elf32lriscv -T microbenchmarks/link.ld vb.o -o vb.elf
build/bin/reef_perf --elf vb.elf --trace vb.tsv
python -m reef_perf.vector_view vb.tsv -o vb.html
```

On M3 the queue peaks at 5 of 8 entries: the 8-entry ROB stops dispatch
before the queue fills, so the red "queue full" shading doesn't appear.
