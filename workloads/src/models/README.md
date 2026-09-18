# Models

Each file rebuilds one architecture using only kernels from `operators/`.

---

## Image classifiers

Input `(B, 3, H, W)`, output `(B, num_classes)`.

### `googlenet.py`

Inception blocks that run 1x1, 3x3, 5x5 and pooled branches and concatenate them. Kernels: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear.

### `resnet.py`

Residual blocks, basic for shallow depths and bottleneck (1x1, 3x3, 1x1) for deep ones; ResNeXt makes the 3x3 a grouped convolution. Kernels: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear.

### `efficientnet.py`

MBConv blocks (expand, depthwise, squeeze-excitation, project) scaled in width and depth; V2 fuses the early blocks into single 3x3 convolutions. Kernels: Conv2d, BatchNorm2d, SiLU, Sigmoid, AdaptiveAvgPool2d, Linear.

## Language models

Input `(B, L)` token ids and an optional KV cache; output `(B, L, vocab)` logits and the updated cache.

### `gpt2.py`

Learned position embeddings, fused QKV attention and a tanh-GELU MLP. Kernels: Embedding, LayerNorm, GroupedQueryAttention, GELU, Linear.

### `llama.py`

RMSNorm, rotary position embeddings, grouped-query attention and a SwiGLU MLP. Kernels: Embedding, RMSNorm, RotaryEmbedding, GroupedQueryAttention, SiLU, Linear.
