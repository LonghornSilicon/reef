# Configs

## Classification (`classification/`)

### `vgg.py`

| size | params |
|---|---|
| VGG-11 / VGG-11-BN | 132.9M / 132.9M |
| VGG-13 / VGG-13-BN | 133.0M / 133.1M |
| VGG-16 / VGG-16-BN | 138.4M / 138.4M |
| VGG-19 / VGG-19-BN | 143.7M / 143.7M |

### `googlenet.py`

| size | params |
|---|---|
| GoogLeNet | 13.0M |

### `resnet.py`

| size | params |
|---|---|
| ResNet-18 / 34 / 50 / 101 / 152 | 11.7M / 21.8M / 25.6M / 44.5M / 60.2M |
| ResNeXt-50-32x4d / ResNeXt-101-32x8d | 25.0M / 88.8M |

### `mobilenet_v3.py`

| size | params |
|---|---|
| MobileNetV3-Small / MobileNetV3-Large | 2.5M / 5.5M |

### `efficientnet.py`

| size | params |
|---|---|
| B0 / B1 / B2 / B3 | 5.3M / 7.8M / 9.1M / 12.2M |
| B4 / B5 / B6 / B7 | 19.3M / 30.4M / 43.0M / 66.3M |
| V2-S / V2-M / V2-L | 21.5M / 54.1M / 118.5M |

### `convnext.py`

| size | params |
|---|---|
| T / S / B / L / XL | 28.6M / 50.2M / 88.6M / 197.8M / 392.9M |

## Detection (`detection/`)

### `yolov8.py`

| size | params |
|---|---|
| n / s / m / l / x | 3.2M / 11.2M / 25.9M / 43.7M / 68.2M |
| YOLOv3u | 103.8M |

### `yolo11.py`

| size | params |
|---|---|
| n / s / m / l / x | 2.6M / 9.5M / 20.1M / 25.4M / 57.0M |

## Vision transformers (`vision_transformer/`)

### `vit.py`

| size | params |
|---|---|
| Ti/16 / S/16 / B/16 / L/16 / H/14 | 5.7M / 22.1M / 86.6M / 304.3M / 630.8M |

### `swin.py`

| size | params |
|---|---|
| T / S / B / L | 28.3M / 49.6M / 87.8M / 196.5M |

### `mobilevit.py`

| size | params |
|---|---|
| XXS / XS / S | 1.3M / 2.3M / 5.6M |

### `dinov2.py`

| size | params |
|---|---|
| S / B / L / g | 22.1M / 86.6M / 304.4M / 1136.5M |

### `segformer.py`

| size | params |
|---|---|
| B0 / B1 / B2 / B3 / B4 / B5 | 4.0M / 13.9M / 28.1M / 48.0M / 64.8M / 85.4M |

## Vision-language models (`vision_language/`)

### `clip.py`

| size | params |
|---|---|
| ViT-B/32 / ViT-B/16 / ViT-L/14 | 151.3M / 149.6M / 427.6M |

### `siglip.py`

| size | params |
|---|---|
| B/16 / L/16 / So400m/14 | 203.2M / 652.2M / 878.0M |

## Language models (`language/`)

### `gpt2.py`

| size | params |
|---|---|
| GPT-2 / Medium / Large / XL | 124.4M / 354.8M / 774.0M / 1557.6M |

### `bloom.py`

| size | params |
|---|---|
| BLOOM-560M / BLOOM-1B1 | 559.2M / 1065.3M |

### `llama.py`

| size | params |
|---|---|
| TinyLlama-1.1B / Llama-3.2-1B | 1100.0M / 1235.8M |
| SmolLM2-135M / SmolLM2-360M | 134.5M / 361.8M |

### `qwen2.py`

| size | params |
|---|---|
| Qwen2.5-0.5B / Qwen2.5-1.5B | 494.0M / 1543.7M |

### `qwen3.py`

| size | params |
|---|---|
| Qwen3-0.6B | 596.0M |
