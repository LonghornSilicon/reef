# Models

Each file rebuilds one architecture using only kernels from `operators/`.

---

## Language models

Input `(B, L)` token ids and an optional KV cache; output `(B, L, vocab)` logits and the updated cache.

### `gpt_neo.py`

Learned position embeddings, separate bias-free QKV projections and a tanh-GELU MLP. Attention is unscaled and alternates global with a 256-wide sliding window. Kernels: Embedding, LayerNorm, GroupedQueryAttention, GELU, Linear.
