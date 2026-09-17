# Configs

The hyperparameters of every published model size, as frozen dataclasses. A config holds only what the model in `models/` needs to rebuild the reference architecture exactly; training settings, tokenizers and preprocessing live elsewhere.

**Shape of a file.** Each file has one dataclass per architecture (two-tower models nest a vision and a text config), one named constant per published size, a `<FAMILY>_CONFIGS` dict keyed by display name, and a map from that name to where the weights come from: `TORCHVISION_BUILDERS` (a `torchvision.models` function), `<FAMILY>_REPOS` (a Hugging Face repo id) or `<FAMILY>_SOURCES` (a download URL). Values shared by every size are dataclass defaults; the per-size constants set only what differs.

**Provenance.** Every value was read from the reference, never from memory: torchvision builders instantiated and inspected, Hugging Face `config.json` files, or the upstream YAML and source for ultralytics. Parameter counts below were measured by instantiating the reference and counting parameters (buffers excluded) and are pinned by the tests in `tests/models/`.

**Size limit.** Only variants at or under about 1.5B parameters are kept.

---

## Classification CNNs

### `vgg.py`

`VGGConfig`: the conv layer spec with `"M"` max-pool markers, `batch_norm`, `num_classes`, `dropout`. torchvision.

| size | params |
|---|---|
| VGG-11 / VGG-11-BN | 132.9M / 132.9M |
| VGG-13 / VGG-13-BN | 133.0M / 133.1M |
| VGG-16 / VGG-16-BN | 138.4M / 138.4M |
| VGG-19 / VGG-19-BN | 143.7M / 143.7M |

### `googlenet.py`

`GoogLeNetConfig`: the nine Inception channel tuples, `aux_logits`, dropout rates, BN eps 1e-3. torchvision. GoogLeNet: 13.0M with the two auxiliary heads, 6.6M without.

### `resnet.py`

`ResNetConfig`: block kind, blocks per stage, `groups` and `width_per_group` (ResNeXt), `norm_eps`. torchvision.

| size | params |
|---|---|
| ResNet-18 / 34 / 50 / 101 / 152 | 11.7M / 21.8M / 25.6M / 44.5M / 60.2M |
| ResNeXt-50-32x4d / ResNeXt-101-32x8d | 25.0M / 88.8M |

### `mobilenet_v3.py`

`MobileNetV3Config`: the inverted-residual table (input, kernel, expanded, output, SE, activation, stride, dilation), `last_channel`, BN eps 1e-3 momentum 0.01. torchvision. Small 2.5M, Large 5.5M.

### `efficientnet.py`

`EfficientNetConfig`: the MBConv / FusedMBConv table, `width_mult`, `depth_mult`, `dropout`, `last_channel`, BN eps and momentum (B5–B7 use 1e-3 / 0.01, V2 uses 1e-3 / 0.1). torchvision.

| size | params |
|---|---|
| B0 / B1 / B2 / B3 | 5.3M / 7.8M / 9.1M / 12.2M |
| B4 / B5 / B6 / B7 | 19.3M / 30.4M / 43.0M / 66.3M |
| V2-S / V2-M / V2-L | 21.5M / 54.1M / 118.5M |

### `convnext.py`

`ConvNeXtConfig`: stage table (input channels, output channels, layers), `layer_scale`, `stochastic_depth_prob`, `norm_eps`, `head_norm_eps`. torchvision for T/S/B/L; Hugging Face for XL, whose checkpoint keys differ from the torchvision tree and do not load yet.

| size | params |
|---|---|
| T / S / B / L / XL | 28.6M / 50.2M / 88.6M / 197.8M / 392.9M |

## Detection

### `yolov8.py`

`YOLOv8Config`: `depth`, `width`, `max_channels`, `backbone` (`"csp"` from `yolov8.yaml` or `"darknet53"` from `yolov3.yaml`), `reg_max`, `num_classes`. Weights from the ultralytics GitHub releases.

| size | params |
|---|---|
| n / s / m / l / x | 3.2M / 11.2M / 25.9M / 43.7M / 68.2M |
| YOLOv3u | 103.8M |

### `yolo11.py`

`YOLO11Config`: as YOLOv8 plus `c3k`, which `parse_model` forces on for m/l/x regardless of the YAML. Weights from the ultralytics GitHub releases.

| size | params |
|---|---|
| n / s / m / l / x | 2.6M / 9.5M / 20.1M / 25.4M / 57.0M |

## Vision transformers

### `vit.py`

`ViTConfig`: hidden size, layers, heads, MLP width, patch and image size, `layer_norm_eps`, `num_labels`. Hugging Face; Ti and S come from timm-converted repos.

| size | params |
|---|---|
| Ti/16 / S/16 / B/16 / L/16 / H/14 | 5.7M / 22.1M / 86.6M / 304.3M / 630.8M |

### `swin.py`

`SwinConfig`: `embed_dim`, `depths`, `num_heads`, `window_size`, `mlp_ratio`, `patch_size`. Hugging Face repos hold the weights; the model mirrors torchvision's tree, which ships no `swin_l` builder, so tests build the reference class directly.

| size | params |
|---|---|
| T / S / B / L | 28.3M / 49.6M / 87.8M / 196.5M |

### `mobilevit.py`

`MobileViTConfig`: `hidden_sizes`, `neck_hidden_sizes`, `expand_ratio`, heads, `patch_size`, `image_size`, `output_stride`. Hugging Face. XXS 1.3M, XS 2.3M, S 5.6M.

### `dinov2.py`

`Dinov2Config`: hidden size, layers, heads, `mlp_ratio`, `use_swiglu_ffn` (giant only), `layerscale_value`, patch 14, image 518, `num_labels`. Hugging Face.

| size | params |
|---|---|
| S / B / L / g | 22.1M / 86.6M / 304.4M / 1136.5M |

### `segformer.py`

`SegformerConfig`: per-stage `hidden_sizes`, `depths`, heads, `sr_ratios`, `patch_sizes`, `strides`, `mlp_ratios`, `decoder_hidden_size`, `num_labels`. Hugging Face MiT encoders. Counts include a 1000-class decode head.

| size | params |
|---|---|
| B0 / B1 / B2 / B3 / B4 / B5 | 4.0M / 13.9M / 28.1M / 48.0M / 64.8M / 85.4M |

### `clip.py`

`CLIPConfig` nests `CLIPVisionConfig` and `CLIPTextConfig` (each: hidden size, layers, heads, MLP width, `layer_norm_eps`; vision adds patch and image size; text adds vocabulary, positions and `eos_token_id`), plus `projection_dim` and `logit_scale_init_value`. Hugging Face.

| size | params |
|---|---|
| ViT-B/32 / ViT-B/16 / ViT-L/14 | 151.3M / 149.6M / 427.6M |

### `siglip.py`

`SiglipConfig` nests `SiglipVisionConfig` and `SiglipTextConfig` as CLIP does, with `gelu_pytorch_tanh` activations and a text `projection_size`. Hugging Face.

| size | params |
|---|---|
| B/16 / L/16 / So400m/14 | 203.2M / 652.2M / 878.0M |

## Decoder language models

All decoders share `vocab_size`, `hidden_size`, `intermediate_size` (or the 4x rule), layers, heads, `max_position_embeddings`, norm eps and `tie_word_embeddings`.

### `gpt2.py`

`GPT2Config`: `n_embd`, `n_layer`, `n_head`, `n_positions`, `activation_function` (`gelu_new`), `scale_attn_weights`. Hugging Face.

| size | params |
|---|---|
| GPT-2 / Medium / Large / XL | 124.4M / 354.8M / 774.0M / 1557.6M |

### `bloom.py`

`BloomConfig`: `n_layer`, `n_head`, `hidden_size`, `apply_residual_connection_post_layernorm`. Hugging Face. 560M: 559.2M, 1B1: 1065.3M.

### `llama.py`

`LlamaConfig`: the Llama fields plus `num_key_value_heads`, `head_dim`, `rope_theta`, `attention_bias`, `mlp_bias`, and `rope_scaling` (Llama 3's frequency-smoothing dict, `None` otherwise). Hugging Face; `LLAMA_MIRRORS` gives an ungated re-upload of the gated Llama 3.2 repo with identical config and weights.

| size | params |
|---|---|
| TinyLlama-1.1B / Llama-3.2-1B | 1100.0M / 1235.8M |
| SmolLM2-135M / SmolLM2-360M | 134.5M / 361.8M |

### `qwen2.py`

`Qwen2Config`: as Llama with `qkv_bias` (always on in Qwen2), `rope_theta` 1e6, `use_sliding_window` and `sliding_window` (off in both published sizes). Hugging Face. Qwen2.5-0.5B: 494.0M, Qwen2.5-1.5B: 1543.7M.

### `qwen3.py`

`Qwen3Config`: as Qwen2 without the QKV bias and with per-head Q/K RMSNorm. Hugging Face. Qwen3-0.6B: 0.6B.
