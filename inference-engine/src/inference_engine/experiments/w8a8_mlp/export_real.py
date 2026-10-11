"""Capture and export a pinned TinyStories MLP, without Coral execution."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from inference_engine.experiments.w8a8_mlp.main import quantize
from inference_engine.experiments.w8a8_mlp.real_support import (
    REEF_ROOT,
    external_directory,
    sha256,
)

MODEL = "roneneldan/TinyStories-Instruct-28M"
REVISION = "96c7ddf6012aaec1b227a1a663f09de016de0621"
LAYER = 3


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Use the existing reference generator's FP32 tanh-GELU expression."""
    return (
        0.5 * x * (1 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))
    )


def reference_arrays(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluate exact integer dots and independently ordered FP32 ops."""
    for tensor in (x, w1, b1, w2, b2):
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Reference inputs must be finite CPU FP32")
        if tensor.device.type != "cpu":
            raise ValueError("Reference must run on CPU")
    qx, sx = quantize(x)
    q1, s1 = quantize(w1)
    q2, s2 = quantize(w2)
    a1 = qx.to(torch.int64) @ q1.to(torch.int64).T
    pre = (a1.float() * sx[:, None]) * s1[None, :] + b1
    hidden = gelu(pre)
    qh, sh = quantize(hidden)
    a2 = qh.to(torch.int64) @ q2.to(torch.int64).T
    for accumulator in (a1, a2):
        if torch.any(accumulator < -(2**31)) or torch.any(accumulator >= 2**31):
            raise ValueError("Reference accumulator exceeds INT32")
    output = (a2.float() * sh[:, None]) * s2[None, :] + b2
    fp32_pre = x @ w1.T + b1
    fp32_hidden = gelu(fp32_pre)
    fake_pre = (qx.float() * sx[:, None]) @ (q1.float() * s1[:, None]).T + b1
    fake_hidden = gelu(fake_pre)
    fake_qh, fake_sh = quantize(fake_hidden)
    fake_output = (fake_qh.float() * fake_sh[:, None]) @ (
        q2.float() * s2[:, None]
    ).T + b2
    arrays = {
        "input_fp32": x,
        "w1_fp32": w1,
        "w2_fp32": w2,
        "w1": q1,
        "s1": s1,
        "b1": b1,
        "w2": q2,
        "s2": s2,
        "b2": b2,
        "expected_input_q": qx,
        "expected_input_scale": sx,
        "expected_a1": a1.to(torch.int32),
        "expected_pre": pre,
        "expected_hidden": hidden,
        "expected_hidden_q": qh,
        "expected_hidden_scale": sh,
        "expected_a2": a2.to(torch.int32),
        "expected_output": output,
        "fp32_pre": fp32_pre,
        "fp32_hidden": fp32_hidden,
        "fp32_output": fp32_hidden @ w2.T + b2,
        "fake_pre": fake_pre,
        "fake_hidden_q": fake_qh,
        "fake_output": fake_output,
    }
    for name, tensor in arrays.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite reference stage: {name}")
    for scale in (sx, s1, s2, sh):
        if torch.any(scale < torch.finfo(torch.float32).tiny):
            raise ValueError("Unsupported subnormal or underflowed scale")
    return arrays


def capture(cache: Path | None) -> tuple[dict[str, torch.Tensor], dict]:
    """Capture actual post-ln_2 activations from the fixed FP32 prompt pass."""
    import transformers
    from huggingface_hub import snapshot_download
    from workloads.experiments.story_quality.main import PROMPTS

    snapshot = Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            cache_dir=str(cache) if cache else None,
            allow_patterns=[
                "config.json",
                "pytorch_model.bin",
                "tokenizer*",
                "special_tokens_map.json",
                "merges.txt",
                "vocab.json",
            ],
        )
    )
    if snapshot.name != REVISION:
        raise ValueError("Checkpoint did not resolve to the pinned revision")
    model = transformers.GPTNeoForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
    )
    block = model.transformer.h[LAYER].mlp
    if (tuple(block.c_fc.weight.shape), tuple(block.c_proj.weight.shape)) != (
        (2048, 512),
        (512, 2048),
    ) or model.config.activation_function != "gelu_new":
        raise ValueError(
            "Checkpoint architecture differs from the approved MLP"
        )
    if block.c_fc.bias is None or block.c_proj.bias is None:
        raise ValueError("Expected both checkpoint biases")
    prompt = dict(PROMPTS)["summary_only"]
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    length = encoded["input_ids"].shape[1]
    positions = [0, length // 2, length - 1]
    if len(set(positions)) != 3:
        raise ValueError("Prompt must contain three distinct token positions")
    captured = []

    def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured.append(args[0].detach().clone())

    handle = block.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected one MLP invocation during prompt prefill")
    x = captured[0][0, positions].contiguous()
    arrays = reference_arrays(
        x,
        block.c_fc.weight.detach(),
        block.c_fc.bias.detach(),
        block.c_proj.weight.detach(),
        block.c_proj.bias.detach(),
    )
    with torch.inference_mode():
        arrays["fp32_block_output"] = block(x).detach()
    metadata = {
        "kind": "real",
        "model": MODEL,
        "revision": REVISION,
        "tokenizer_revision": REVISION,
        "layer": LAYER,
        "module": "transformer.h.3.mlp",
        "input_boundary": "post-ln_2",
        "prompt_key": "summary_only",
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_ids": encoded["input_ids"][0].tolist(),
        "attention_mask": encoded["attention_mask"][0].tolist(),
        "positions": positions,
        "selected_tokens": [
            tokenizer.decode(encoded["input_ids"][0, p : p + 1])
            for p in positions
        ],
        "tokenizer_class": type(tokenizer).__name__,
        "checkpoint_files": {
            p.name: sha256(p) for p in snapshot.iterdir() if p.is_file()
        },
        "transformers": transformers.__version__,
        "prompt_source_sha256": sha256(
            REEF_ROOT
            / "workloads/src/workloads/experiments/story_quality/main.py"
        ),
    }
    return arrays, metadata


def synthetic() -> tuple[dict[str, torch.Tensor], dict]:
    """Exercise the export/host path with the original small fixture recipe."""
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(3, 8, generator=generator) * 0.5
    x[-1] = 0
    w1 = torch.randn(16, 8, generator=generator) * 0.15
    w2 = torch.randn(8, 16, generator=generator) * 0.15
    w1[-1] = 0
    w2[-1] = 0
    b1 = torch.randn(16, generator=generator) * 0.05
    b2 = torch.randn(8, generator=generator) * 0.05
    arrays = reference_arrays(x, w1, b1, w2, b2)
    arrays["fp32_block_output"] = arrays["fp32_output"].clone()
    return arrays, {"kind": "synthetic-unit", "seed": 42}


def main() -> None:
    """Write arrays outside Git, with value hashes and checkpoint provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--synthetic-check", action="store_true")
    args = parser.parse_args()
    output = external_directory(args.output)
    if output.exists():
        parser.error(
            "Output must be a new directory; existing fixtures are immutable"
        )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.inference_mode():
        tensors, metadata = (
            synthetic()
            if args.synthetic_check
            else capture(external_directory(args.cache) if args.cache else None)
        )
    arrays = {
        name: tensor.detach().contiguous().numpy()
        for name, tensor in tensors.items()
    }
    output.mkdir(parents=True)
    archive = output / "arrays.npz"
    np.savez(archive, **arrays)
    metadata.update(
        {
            "schema": 1,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "device": "cpu",
            "threads": 1,
            "deterministic_algorithms": True,
            "dimensions": {
                "tokens": arrays["input_fp32"].shape[0],
                "width": arrays["input_fp32"].shape[1],
                "hidden": arrays["w1"].shape[0],
            },
            "quantization": "symmetric [-127,127], ties-even, zero-scale=1; "
            "weights per output row, activations per token",
            "archive_sha256": sha256(archive),
            "arrays": {
                name: {
                    "shape": list(a.shape),
                    "dtype": a.dtype.str,
                    "bytes": a.nbytes,
                    "value_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                }
                for name, a in arrays.items()
            },
            "exporter_sha256": sha256(Path(__file__)),
            "quantizer_source_sha256": sha256(
                Path(__file__).with_name("main.py")
            ),
            "workloads_lock_sha256": sha256(REEF_ROOT / "workloads/uv.lock"),
        }
    )
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported {metadata['kind']} fixture to {output}")


if __name__ == "__main__":
    main()
