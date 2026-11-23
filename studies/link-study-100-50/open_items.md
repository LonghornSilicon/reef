## Open items (to return to)

1. **Experiment A (low-rank weights) — return to this.** It ran at the speed of int4 weights (52M: 1.85 to 3.33 tokens/s at
   batch 1, +80%; TTFT 1.42 to 0.77 s) with no new hardware, because it halves the flash bytes per token, which is what limits
   decode here, while the matrix work it adds is close to free (the array is idle 90%+ of the time). Not known: accuracy. Next
   steps: (a) factor the real TinyStories-28M weights at these ranks and measure the model's quality (perplexity) offline, per
   matrix and per 16x16 tile (Aarush's form); (b) if it holds, try A with int4 factors (about a quarter of the int8 bytes).
   The Python environment for (a) is installed (scratchpad/mlenv); the run was stopped on request before downloading weights.
2. **Vector-only path:** needs the vector firmware source or a token-200 build; the ELF received is the default 32-token build.
3. **Speculative decoding:** checking-pass cost on the simulator (k + 1 tokens of one story) and guess acceptance offline.
4. **Not simulated, computed only:** KV prefetch with about 1.6 MB of on-chip SRAM (the "links overlapped" limit); an int4 KV
   cache (halves the PSRAM bytes).
