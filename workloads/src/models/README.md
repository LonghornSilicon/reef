# Models

Each file rebuilds one published architecture from the kernels in `operators/`, using nothing else. The point is that a model here is exactly the reference model, expressed only in the primitives the hardware must support.

**Shape of a file.** One module per reference architecture class. Submodule and parameter names mirror the reference's module tree, so `ours.load_state_dict(reference.state_dict(), strict=True)` succeeds and the published checkpoints load unchanged. Every file ends with a factory, `vit("ViT-B/16")` and so on, that looks the name up in the matching `configs/` table and raises `KeyError` for anything else.

**What is not a kernel.** Reshapes, concatenation, windowing, patch folding, KV-cache concatenation and residual adds are plain tensor code in the model file. A one-line expression that only one model uses (QuickGELU, Hardswish, LayerScale, ALiBi slopes) is written inline here rather than added to `operators/`. Stochastic depth and dropout are identity at inference; dropout modules appear only where the reference has them, and they hold no parameters.

**Verification.** `tests/models/test_<name>.py` builds the reference from the same weights and checks the state-dict key set, the outputs in float32 and bfloat16, the parameter count of every published size, and one behaviour that output parity alone would miss. `test_checkpoints.py` additionally runs real pretrained weights for ResNet and Qwen3.

---

## Classification CNNs

Input `(B, 3, H, W)`, output `(B, num_classes)` logits.

### `vgg.py`

`VGG`, matching torchvision: a `features` sequence of 3x3 convolutions, optional BatchNorm, ReLU and 2x2 max-pools laid out from the config's layer spec, adaptive pooling to 7x7, then the three-layer classifier with dropout. Kernels: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Dropout, Linear.

### `googlenet.py`

`GoogLeNet`, matching torchvision: `BasicConv2d` (conv without bias, BatchNorm eps 1e-3, ReLU), nine `Inception` blocks with 1x1, 3x3, "5x5" (really 3x3, as in torchvision) and pooled-projection branches, `ceil_mode` max-pools between stages, and two `InceptionAux` heads that only run in training. Kernels: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Dropout, Linear.

### `resnet.py`

`ResNet`, matching torchvision V1.5: `BasicBlock` for 18/34 and `Bottleneck` for the rest, with the stride on the 3x3 convolution. ResNeXt widens the bottleneck's 3x3 to `groups * width_per_group` channels per 64 planes and makes it a grouped convolution. Kernels: Conv2d (grouped), BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear.

### `mobilenet_v3.py`

`MobileNetV3`, matching torchvision: `InvertedResidual` blocks (1x1 expand, depthwise kxk, optional `SqueezeExcitation` with two 1x1 convolutions, 1x1 project), ReLU or Hardswish per row, BatchNorm eps 1e-3, and a classifier of Linear, Hardswish, Dropout, Linear. Hardswish and Hardsigmoid are inline clamps. Kernels: Conv2d (depthwise), BatchNorm2d, ReLU, AdaptiveAvgPool2d, Dropout, Linear.

### `efficientnet.py`

`EfficientNet`, matching torchvision: `MBConv` (1x1 expand, depthwise kxk, `SqueezeExcitation` with SiLU and Sigmoid, 1x1 project, residual at stride 1) and `FusedMBConv` (one fused 3x3), channels and repeats rounded exactly as torchvision does, a 1x1 head convolution, then Dropout and Linear. Kernels: Conv2d (depthwise), BatchNorm2d, SiLU, Sigmoid, AdaptiveAvgPool2d, Dropout, Linear.

### `convnext.py`

`ConvNeXt`, matching torchvision: a 4x4 stride-4 stem, `CNBlock` (depthwise 7x7, LayerNorm over channels, Linear 4x, exact GELU, Linear, per-channel `layer_scale`, residual), 2x2 stride-2 downsampling, and a classifier of LayerNorm, Flatten, Linear. `LayerNorm2d` is the operator LayerNorm wrapped in permutes. Kernels: Conv2d (depthwise), LayerNorm, ErfGELU, AdaptiveAvgPool2d, Linear.

## Detection

Input `(B, 3, H, W)` with `H` and `W` multiples of 32. `forward` returns the ultralytics inference tensor `(B, 4 + num_classes, anchors)`: decoded xywh boxes in pixels and sigmoid class scores. `postprocess(preds, conf, iou, max_det)` applies class-aware NMS and returns per-image `(n, 6)` rows of xyxy box, score and class, matching `non_max_suppression`.

### `yolov8.py`

`YOLOv8`, matching ultralytics: `Conv` (conv, BatchNorm eps 1e-3, SiLU), `Bottleneck`, `C2f`, `SPPF` (three chained 5x5 stride-1 max-pools), nearest 2x upsampling, `Concat`, and the anchor-free `Detect` head with `DFL` and `DistanceToBox`. The row tables for `yolov8.yaml` and `yolov3.yaml` are both here, so `YOLOv3u` (Darknet-53 backbone, same head) builds from this file. Kernels: Conv2d, BatchNorm2d, SiLU, Sigmoid, MaxPool2d, Interpolate, DFL, DistanceToBox, NMS.

### `yolo11.py`

`YOLO11`, matching ultralytics: reuses the YOLOv8 blocks and adds `C3k2`, `C3k`, and `C2PSA`, whose `Attention` runs the operator attention kernel over 1x1-convolution QKV with a depthwise 3x3 positional branch and the `key_dim` scale. Kernels: those of YOLOv8 plus GroupedQueryAttention.

## Vision transformers

Classifiers take `(B, 3, H, W)` and return `(B, num_labels)`. ViT and DINOv2 resize their position embeddings bicubically when `H` or `W` differ from the training size.

### `vit.py`

`ViTForImageClassification`, matching transformers: a patch-embedding convolution, class token, learned positions, pre-norm encoder layers (LayerNorm, attention, LayerNorm, MLP with exact GELU), final LayerNorm, and a Linear head on the class token. Kernels: Conv2d, LayerNorm, GroupedQueryAttention, ErfGELU, Interpolate, Dropout, Linear.

### `swin.py`

`SwinTransformer`, matching torchvision V1: 4x4 patch embedding, `ShiftedWindowAttention` (windows of 7, cyclic shift by `torch.roll`, a learned relative-position bias table gathered by a precomputed index, and the shifted-window mask added as -100), MLP with exact GELU, `PatchMerging` (concatenate 2x2, LayerNorm, Linear 4C to 2C), then LayerNorm, average pooling and a Linear head. The tree follows torchvision, so the Hugging Face checkpoints named in the config need a key rename before they load. Kernels: Conv2d, LayerNorm, GroupedQueryAttention (bias argument), ErfGELU, AdaptiveAvgPool2d, Dropout, Linear.

### `mobilevit.py`

`MobileViTForImageClassification`, matching transformers: MobileNetV2-style inverted residuals with SiLU, and `MobileViTLayer` blocks that unfold the feature map into 2x2 patches, run pre-norm transformer layers over them, fold back, and fuse with a 3x3 convolution; feature maps not divisible by the patch size are resized bilinearly around the transformer. Kernels: Conv2d (depthwise), BatchNorm2d, LayerNorm, SiLU, GroupedQueryAttention, Interpolate, Dropout, Linear.

### `dinov2.py`

`Dinov2ForImageClassification`, matching transformers: ViT with patch 14, a mask token, `LayerScale` gains on both residual branches, an MLP with exact GELU (or a SwiGLU FFN for the giant size), and a Linear head on the class token concatenated with the mean patch token. Kernels: Conv2d, LayerNorm, GroupedQueryAttention, ErfGELU, SiLU, Interpolate, Dropout, Linear.

### `segformer.py`

`SegformerForSemanticSegmentation`, matching transformers: a MiT encoder of four stages, each an overlapping patch-embedding convolution, blocks with sequence-reduced attention (a strided convolution shrinks keys and values), and a Mix-FFN whose hidden layer is a depthwise 3x3 convolution; the decode head projects every stage to a common width, upsamples bilinearly to 1/4 resolution, concatenates, fuses with a 1x1 convolution, BatchNorm and ReLU, and classifies with a 1x1 convolution. Output is `(B, num_labels, H/4, W/4)`. Kernels: Conv2d (depthwise), LayerNorm, BatchNorm2d, GroupedQueryAttention, ErfGELU, ReLU, Interpolate, Dropout, Linear.

## Image-text models

`forward(pixel_values, input_ids)` returns unit-normalised `image_embeds` and `text_embeds` and `logits_per_image`, the scaled cosine similarities.

### `clip.py`

`CLIPModel`, matching transformers: a ViT vision tower with a pre-LayerNorm and post-LayerNorm on the class token, a causal text tower pooled at the EOS token, QuickGELU (`x * sigmoid(1.702 x)`) in both, Linear projections without bias, and a learned `logit_scale`. Kernels: Conv2d, Embedding, LayerNorm, GroupedQueryAttention (causal for text), Sigmoid, L2Norm, Linear.

### `siglip.py`

`SiglipModel`, matching transformers: no class token, tanh-GELU MLPs, a vision tower pooled by `SiglipMultiheadAttentionPoolingHead` (a learned probe attending over the patches through a packed-QKV attention with the reference's parameter layout), a text tower pooled at the last token, and learned `logit_scale` and `logit_bias`. Kernels: Conv2d, Embedding, LayerNorm, GroupedQueryAttention, GELU, L2Norm, Linear.

## Decoder language models

`forward(input_ids, past_key_values=None)` returns `(logits, cache)`; the cache is a list of per-layer key and value tensors, and queries are placed at the final positions of the key axis so cached decoding needs no explicit offsets. `generate(input_ids, max_new_tokens)` decodes greedily. The factory ties `lm_head` to the embedding table when the config says so.

### `gpt2.py`

`GPT2LMHeadModel`, matching transformers: learned token and position embeddings, post-attention residual blocks with `Conv1D` projections (a matmul against a weight stored as `(in, out)`, kept in that layout so the checkpoint loads), fused QKV, tanh-GELU MLP, LayerNorm. Kernels: Embedding, LayerNorm, GroupedQueryAttention, GELU, Linear.

### `bloom.py`

`BloomForCausalLM`, matching transformers: an embedding LayerNorm, fused QKV in the `(heads, 3, head_dim)` layout, ALiBi (per-head slopes times key position, built inline and passed as the attention bias), tanh-GELU MLP. Kernels: Embedding, LayerNorm, GroupedQueryAttention (bias argument), GELU, Linear.

### `llama.py`

`LlamaForCausalLM`, matching transformers, and the architecture of TinyLlama and SmolLM2 as well: RMSNorm, rotary embeddings (with Llama 3's frequency smoothing computed inline from `rope_scaling` and handed to the rotary kernel), grouped-query attention, and a SwiGLU MLP. Kernels: Embedding, RMSNorm, RotaryEmbedding, GroupedQueryAttention, SiLU, Linear.

### `qwen2.py`

`Qwen2ForCausalLM`, matching transformers: Llama with bias on the Q, K and V projections. Kernels: as Llama.

### `qwen3.py`

`Qwen3ForCausalLM`, matching transformers: Llama with per-head RMSNorm on queries and keys before the rotary embedding. Kernels: as Llama.
