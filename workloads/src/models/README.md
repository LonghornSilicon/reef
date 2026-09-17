# Models

Each file rebuilds one architecture using only kernels from `operators/`.

---

## Classification (`classification/`)

Input `(B, 3, H, W)`, output `(B, num_classes)`.

### `vgg.py`

Stacks of 3x3 convolutions with optional BatchNorm, separated by 2x2 max-pools, then a three-layer classifier. Kernels: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear.

### `googlenet.py`

Inception blocks that run 1x1, 3x3, 5x5 and pooled branches and concatenate them. Kernels: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear.

### `resnet.py`

Residual blocks, basic for shallow depths and bottleneck (1x1, 3x3, 1x1) for deep ones; ResNeXt makes the 3x3 a grouped convolution. Kernels: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear.

### `mobilenet_v3.py`

Inverted residual blocks with depthwise convolution and squeeze-excitation, using ReLU or Hardswish per block. Kernels: Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, Linear.

### `efficientnet.py`

MBConv blocks (expand, depthwise, squeeze-excitation, project) scaled in width and depth; V2 fuses the early blocks into single 3x3 convolutions. Kernels: Conv2d, BatchNorm2d, SiLU, Sigmoid, AdaptiveAvgPool2d, Linear.

### `convnext.py`

Blocks of depthwise 7x7 convolution, LayerNorm, a 4x MLP and LayerScale, downsampled by 2x2 stride-2 convolutions. Kernels: Conv2d, LayerNorm, ErfGELU, AdaptiveAvgPool2d, Linear.

## Detection (`detection/`)

Input `(B, 3, H, W)`, output `(B, 4 + num_classes, anchors)` of decoded boxes and class scores; `postprocess` applies NMS.

### `yolov8.py`

A CSP backbone with an SPPF pooling pyramid, a feature-pyramid neck, and an anchor-free head that decodes box distances through DFL. YOLOv3u swaps in a Darknet-53 backbone. Kernels: Conv2d, BatchNorm2d, SiLU, Sigmoid, MaxPool2d, Interpolate, DFL, DistanceToBox, NMS.

### `yolo11.py`

YOLOv8 with `C3k2` blocks and a `C2PSA` attention stage over the deepest feature map. Kernels: those of YOLOv8 plus GroupedQueryAttention.

## Vision transformers (`vision_transformer/`)

Input `(B, 3, H, W)`; classifiers output `(B, num_labels)`.

### `vit.py`

Patch embedding by a strided convolution, a class token and learned positions, pre-norm transformer layers, and a linear head on the class token. Kernels: Conv2d, LayerNorm, GroupedQueryAttention, ErfGELU, Interpolate, Linear.

### `swin.py`

Attention inside shifted 7x7 windows with a learned relative-position bias, merging 2x2 patches between stages. Kernels: Conv2d, LayerNorm, GroupedQueryAttention, ErfGELU, AdaptiveAvgPool2d, Linear.

### `mobilevit.py`

Inverted residuals interleaved with blocks that unfold the feature map into patches, run transformer layers over them, and fold back. Kernels: Conv2d, BatchNorm2d, LayerNorm, SiLU, GroupedQueryAttention, Interpolate, Linear.

### `dinov2.py`

ViT with LayerScale on both residual branches, a SwiGLU feed-forward in the giant size, and a head on the class token concatenated with the mean patch token. Kernels: Conv2d, LayerNorm, GroupedQueryAttention, ErfGELU, SiLU, Interpolate, Linear.

### `segformer.py`

Four encoder stages with overlapping patch embeddings, attention over spatially reduced keys and values, and a depthwise-convolution FFN; a decode head upsamples every stage to 1/4 resolution and fuses them. Output `(B, num_labels, H/4, W/4)`. Kernels: Conv2d, BatchNorm2d, LayerNorm, GroupedQueryAttention, ErfGELU, ReLU, Interpolate, Linear.

## Vision-language models (`vision_language/`)

Inputs `pixel_values` and `input_ids`; outputs unit-normalised image and text embeddings and their scaled cosine similarities.

### `clip.py`

A ViT image tower pooled at the class token and a causal text tower pooled at the end-of-text token, projected with a learned logit scale. Kernels: Conv2d, Embedding, LayerNorm, GroupedQueryAttention, Sigmoid, L2Norm, Linear.

### `siglip.py`

CLIP without a class token: the image tower is pooled by a learned probe attending over the patches, the text tower at the last token, and the logits carry a learned bias. Kernels: Conv2d, Embedding, LayerNorm, GroupedQueryAttention, GELU, L2Norm, Linear.

## Language models (`language/`)

Input `(B, L)` token ids and an optional KV cache; output `(B, L, vocab)` logits and the updated cache.

### `gpt2.py`

Learned position embeddings, fused QKV attention and a tanh-GELU MLP. Kernels: Embedding, LayerNorm, GroupedQueryAttention, GELU, Linear.

### `bloom.py`

Fused QKV attention with an ALiBi bias in place of position embeddings, and a tanh-GELU MLP. Kernels: Embedding, LayerNorm, GroupedQueryAttention, GELU, Linear.

### `llama.py`

RMSNorm, rotary position embeddings, grouped-query attention and a SwiGLU MLP. Kernels: Embedding, RMSNorm, RotaryEmbedding, GroupedQueryAttention, SiLU, Linear.

### `qwen2.py`

Llama with bias on the query, key and value projections. Kernels: as Llama.

### `qwen3.py`

Llama with per-head RMSNorm on queries and keys before the rotary embedding. Kernels: as Llama.
