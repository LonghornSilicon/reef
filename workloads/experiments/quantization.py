"""Quality, storage and conversion cost of weight, activation and KV quant."""

import argparse
import csv
import importlib
import math
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
import torchvision
import transformers
from datasets import load_dataset
from matplotlib.lines import Line2D
from torch import nn
from torch.utils.data import DataLoader, Subset

from configs.gpt2 import GPT2_REPOS
from configs.llama import LLAMA_REPOS
from experiments import quant, tracer
from experiments.plots import SERIES, SURFACE, panels
from operators.attention import GroupedQueryAttention

RESULTS = Path(__file__).resolve().parents[1] / "results" / "quantization"
LANGUAGE = {
    "gpt2": (GPT2_REPOS, transformers.GPT2LMHeadModel),
    "llama": (LLAMA_REPOS, transformers.LlamaForCausalLM),
}
VISION = ("googlenet", "resnet", "efficientnet")
# (weight bits, activation bits, KV bits); None leaves fp32.
SCHEMES = {
    "fp32": (None, None, None),
    "W8": (8, None, None),
    "W4": (4, None, None),
    "A8": (None, 8, None),
    "W8A8": (8, 8, None),
    "W4A8": (4, 8, None),
    "KV8": (None, None, 8),
    "KV4": (None, None, 4),
    "W4A8KV4": (4, 8, 4),
}
VISION_SCHEMES = ("fp32", "W8", "W4", "A8", "W8A8", "W4A8")
WINDOW = 1024
STRIDE = 512
# amax, scale, round and clamp per element; the dequantizing multiply folds
# into the consumer's output scale.
QUANT_OPS = 4
HEADER = [
    "model",
    "scheme",
    "metric",
    "value",
    "delta",
    "samples",
    "weight_bytes",
    "kv_bytes_per_token",
    "quant_ops_per_token",
    "macs_per_token",
]
Scheme = tuple[int | None, int | None, int | None]
Hook = Callable[[nn.Module, tuple], tuple]


def precision(bits: int | None) -> str:
    return "bf16" if bits is None else f"int{bits}"


def quantize_weights(model: nn.Module, bits: int) -> None:
    with torch.no_grad():
        for _, module in quant.quantizable(model):
            view = quant.channel_view(module)
            if bits == 8:
                view.copy_(quant.quantize_per_channel(view, bits))
            else:
                view.copy_(quant.quantize_group(view, bits))


def activation_hook(bits: int) -> Hook:
    def hook(module: nn.Module, args: tuple) -> tuple:
        x = args[0]
        if x.dim() == 4:
            # Feature maps: one scale per pixel over its channels.
            x = quant.quantize_per_token(x.movedim(1, -1), bits).movedim(-1, 1)
        else:
            x = quant.quantize_per_token(x, bits)
        return (x, *args[1:])

    return hook


def kv_hook(bits: int) -> Hook:
    def hook(module: nn.Module, args: tuple) -> tuple:
        query, key, value, *rest = args
        # (batch, kv_heads, length, head_dim): one scale per token per head.
        return (
            query,
            quant.quantize_per_token(key, bits),
            quant.quantize_per_token(value, bits),
            *rest,
        )

    return hook


def apply(model: nn.Module, scheme: Scheme) -> None:
    weights, activations, kv = scheme
    if weights is not None:
        quantize_weights(model, weights)
    if activations is not None:
        for _, module in quant.quantizable(model):
            module.register_forward_pre_hook(activation_hook(activations))
    if kv is not None:
        for module in model.modules():
            if isinstance(module, GroupedQueryAttention):
                module.register_forward_pre_hook(kv_hook(kv))


def build(family: str, key: str) -> nn.Module:
    module = importlib.import_module(f"models.{family}")
    return getattr(module, family)(key)


def language_model(family: str, key: str) -> tuple[nn.Module, object]:
    repos, reference_class = LANGUAGE[family]
    reference = reference_class.from_pretrained(repos[key], dtype=torch.float32)
    model = build(family, key)
    model.load_state_dict(reference.state_dict(), strict=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(repos[key])
    return model.eval(), tokenizer


def vision_model(family: str, key: str) -> tuple[nn.Module, object]:
    name = importlib.import_module(f"configs.{family}").TORCHVISION_BUILDERS[
        key
    ]
    weights = torchvision.models.get_model_weights(name).DEFAULT
    # torchvision drops the aux heads after loading unless asked to keep them,
    # and our state dict expects them.
    extra = {"aux_logits": True} if family == "googlenet" else {}
    reference = getattr(torchvision.models, name)(weights=weights, **extra)
    model = build(family, key)
    model.load_state_dict(reference.state_dict(), strict=True)
    return model.eval(), weights.transforms()


def googlenet_input(x: torch.Tensor) -> torch.Tensor:
    """Re-normalize to ``[-1, 1]`` as torchvision's ``transform_input`` does."""
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return (x * std + mean - 0.5) / 0.5


def windows(total: int, limit: int | None) -> Iterator[tuple[int, int, int]]:
    """``(begin, end, first scored position)`` for each stride window."""
    previous = 0
    for count, begin in enumerate(range(0, total, STRIDE)):
        if limit is not None and count == limit:
            return
        end = min(begin + WINDOW, total)
        yield begin, end, max(begin + 1, previous)
        previous = end
        if end == total:
            return


def wikitext_ids(tokenizer: object) -> torch.Tensor:
    rows = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
    )
    text = "\n\n".join(rows["text"])
    # verbose=False: the whole test set is one sequence, windowed below.
    return torch.tensor(tokenizer(text, verbose=False).input_ids)[None]


@torch.no_grad()
def perplexity(
    model: nn.Module, ids: torch.Tensor, device: str, limit: int | None
) -> tuple[float, int]:
    nll, scored = 0.0, 0
    for begin, end, first in windows(ids.shape[1], limit):
        chunk = ids[:, begin:end].to(device)
        logits, _ = model(chunk)
        log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        targets = chunk[0, 1:]
        picked = log_probs.gather(1, targets[:, None])[:, 0]
        nll -= picked[first - begin - 1 :].sum().item()
        scored += end - first
    return math.exp(nll / scored), scored


@torch.no_grad()
def top1(
    model: nn.Module,
    transform: object,
    path: Path,
    device: str,
    limit: int | None,
    transform_input: bool,
) -> tuple[float, int]:
    dataset = torchvision.datasets.ImageFolder(path, transform)
    if limit is not None and limit < len(dataset):
        # Evenly spaced so a subset still spans every class.
        step = len(dataset) / limit
        dataset = Subset(dataset, [int(i * step) for i in range(limit)])
    correct = 0
    for images, labels in DataLoader(dataset, batch_size=16):
        if transform_input:
            images = googlenet_input(images)
        predicted = model(images.to(device)).argmax(dim=-1).cpu()
        correct += (predicted == labels).sum().item()
    return correct / len(dataset), len(dataset)


def per_token(family: str, key: str) -> tuple[int, int, int]:
    """MACs, quantizable input elements and KV elements for one token."""
    storage = tracer.build(family, key)
    if family in LANGUAGE:
        records = tracer.decode(family, key, 1, 1)
    else:
        records = tracer.classify(family, key, 1, 224)
    names = {name for name, _ in quant.quantizable(storage)}
    inputs = sum(r.input_numel for r in records if r.path in names)
    kv = quant.kv_elements_per_token(storage)
    return sum(r.macs for r in records), inputs, kv


def rows_for(
    family: str,
    key: str,
    scheme_names: tuple[str, ...],
    args: argparse.Namespace,
) -> list[list]:
    macs, inputs, kv_elements = per_token(family, key)
    rows = []
    baseline = None
    for name in scheme_names:
        scheme = SCHEMES[name]
        if family in LANGUAGE:
            model, tokenizer = language_model(family, key)
            apply(model, scheme)
            model.to(args.device)
            ids = wikitext_ids(tokenizer)
            metric = "perplexity"
            value, samples = perplexity(model, ids, args.device, args.limit)
        else:
            model, transform = vision_model(family, key)
            apply(model, scheme)
            model.to(args.device)
            metric = "top1"
            if args.imagenet is None:
                value, samples = float("nan"), 0
            else:
                value, samples = top1(
                    model,
                    transform,
                    args.imagenet,
                    args.device,
                    args.limit,
                    family == "googlenet",
                )
        baseline = value if baseline is None else baseline
        weights, activations, kv = scheme
        storage = tracer.build(family, key)
        weight_bytes = quant.model_weight_bytes(storage, precision(weights))
        kv_bytes = quant.kv_bytes_per_token(storage, precision(kv))
        ops = QUANT_OPS * (
            (inputs if activations is not None else 0)
            + (kv_elements if kv is not None else 0)
        )
        print(
            f"{key:<18}{name:<10}{metric} {value:>8.3f}"
            f"  weights {weight_bytes / 1e6:>7.1f} MB"
            f"  quant ops/token {ops / 1e6:>7.2f} M"
        )
        rows.append(
            [
                key,
                name,
                metric,
                f"{value:.4f}",
                f"{value - baseline:.4f}",
                samples,
                f"{weight_bytes:.0f}",
                f"{kv_bytes:.0f}",
                ops,
                macs,
            ]
        )
    return rows


def plot(rows: list[list], path: Path) -> None:
    metrics = list(dict.fromkeys(row[2] for row in rows))
    models = list(dict.fromkeys(row[0] for row in rows))
    colors = dict(zip(models, SERIES, strict=False))
    figure, axes = panels("quality against weight storage", 1, len(metrics))
    for ax, metric in zip(axes, metrics, strict=True):
        for row in rows:
            model, scheme, kind, _, delta, _, weight_bytes, *_ = row
            if kind != metric:
                continue
            ax.scatter(
                float(weight_bytes) / 1e6,
                float(delta),
                color=colors[model],
                s=30,
                zorder=3,
            )
            ax.annotate(
                scheme,
                (float(weight_bytes) / 1e6, float(delta)),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.set_xlabel("weight MB")
        ax.set_ylabel(f"{metric} change from fp32")
        ax.set_title(metric, loc="left")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=colors[m], label=m)
        for m in models
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=8)
    figure.savefig(path, dpi=150, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("families", nargs="*", default=[*LANGUAGE, *VISION])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imagenet", type=Path)
    parser.add_argument("--limit", type=int, help="windows or images")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    language_rows, vision_rows = [], []
    for family in args.families:
        configs = importlib.import_module(f"configs.{family}")
        for key in getattr(configs, f"{family.upper()}_CONFIGS"):
            if family in LANGUAGE:
                language_rows += rows_for(family, key, tuple(SCHEMES), args)
            else:
                vision_rows += rows_for(family, key, VISION_SCHEMES, args)
    for name, rows in (("llm", language_rows), ("cnn", vision_rows)):
        if not rows:
            continue
        with open(RESULTS / f"{name}.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(HEADER)
            writer.writerows(rows)
    plot(language_rows + vision_rows, RESULTS / "quality.png")


if __name__ == "__main__":
    main()
