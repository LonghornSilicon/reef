"""Greedy TinyStories-Instruct completions per GPT-Neo size and precision."""

import argparse
import copy
import csv
from pathlib import Path

import torch
import transformers
from torch import nn

from workloads.configs.gpt_neo import GPT_NEO_CONFIGS, GPT_NEO_REPOS
from workloads.experiments.quantization import (
    INT4_GROUP,
    PRECISIONS,
    is_quantized,
)
from workloads.models.gpt_neo import gpt_neo

RESULTS = Path(__file__).resolve().parents[4] / "results" / "story_quality"
MAX_NEW_TOKENS = 320
END = "<|endoftext|>"
# int8 and int4 are simulated: values are rounded to the integer grid and
# the arithmetic runs in fp32.

# Field order was randomised in training, but Story always comes last. The
# marker is "Story:" plus a space and a BLANK line, as in the train split; the
# valid split omits that blank line, and prompting without it makes the models
# emit a header of their own to get back on pattern.
PROMPTS = [
    (
        "summary_only",
        "Summary: Lily finds a lost kitten in the garden and helps it find "
        "its mother.\n"
        "Story: \n\n",
    ),
    (
        "summary_words",
        "Summary: Tom builds a sandcastle at the beach and a big wave washes "
        "it away, so he builds a new one further up.\n"
        "Words: build, wave, proud\n"
        "Story: \n\n",
    ),
    (
        "summary_dialogue",
        "Features: Dialogue\n"
        "Summary: Ben asks his grandma why the moon follows them home, and "
        "she explains it is very far away.\n"
        "Story: \n\n",
    ),
    (
        "summary_twist_badending",
        "Features: Twist, BadEnding\n"
        "Summary: Anna trades her red balloon for a shiny rock, and then "
        "discovers the rock is just painted mud.\n"
        "Story: \n\n",
    ),
    (
        "everything_at_once",
        "Features: Dialogue, MoralValue\n"
        "Words: share, lantern, brave\n"
        "Summary: Mia is afraid of the dark until her brother lends her his "
        "lantern, and she learns to share it with a younger child who is "
        "scared too.\n"
        "Random sentence: She held the lantern up high so everyone could "
        "see.\n"
        "Story: \n\n",
    ),
]


def load(key: str) -> tuple[nn.Module, object]:
    """Our GPT-Neo carrying the published weights, and that repo's tokenizer."""
    repo = GPT_NEO_REPOS[key]
    reference = transformers.GPTNeoForCausalLM.from_pretrained(
        repo, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    model = gpt_neo(key)
    model.load_state_dict(reference.state_dict(), strict=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(repo)
    return model.eval(), tokenizer


def quantize(x: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Symmetric rounding with one scale per vector along ``dim``."""
    limit = 2 ** (bits - 1) - 1
    amax = x.abs().amax(dim=dim, keepdim=True)
    # An all-zero vector has amax 0; keep the scale finite so 0 / scale is 0.
    scale = amax.clamp(min=torch.finfo(x.dtype).tiny) / limit
    return torch.clamp(torch.round(x / scale), -limit, limit) * scale


def quantize_input(module: nn.Module, args: tuple) -> tuple:
    return (quantize(args[0], 8, dim=-1), *args[1:])


def convert(model: nn.Module, precision: str) -> nn.Module:
    """A copy of the fp32 ``model`` running at ``precision``."""
    model = copy.deepcopy(model)
    if precision == "bf16":
        return model.to(torch.bfloat16)
    if precision == "fp32":
        return model
    for name, module in model.named_modules():
        if not is_quantized(name, type(module).__name__):
            continue
        weight = module.weight.data
        if precision == "int8":
            # One scale per output channel for the weight, and one per token
            # for the input, as dynamic int8 kernels do.
            weight.copy_(quantize(weight, 8, dim=1))
            module.register_forward_pre_hook(quantize_input)
        else:
            # Weight-only, one scale per 32 inputs of each output channel.
            out_features, in_features = weight.shape
            assert in_features % INT4_GROUP == 0, name
            groups = weight.reshape(out_features, -1, INT4_GROUP)
            weight.copy_(quantize(groups, 4, dim=2).reshape(weight.shape))
    return model


def tell(model: nn.Module, tokenizer: object, prompt: str) -> str:
    """The continuation alone, cut at the first end-of-text marker."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, MAX_NEW_TOKENS)
    # generate() has no EOS stop and always emits MAX_NEW_TOKENS, so the tail
    # runs on into a fabricated next example unless it is cut here.
    text = tokenizer.decode(output[0, input_ids.shape[1] :])
    return text.split(END)[0].strip()


def csv_path(index: int, name: str) -> Path:
    # Numbered so the files sort in the order the prompts add constraints.
    return RESULTS / f"{index}_{name}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "keys",
        nargs="*",
        choices=list(GPT_NEO_CONFIGS),
        metavar="model",
        help=f"any of {', '.join(GPT_NEO_CONFIGS)} (default: all)",
    )
    parser.add_argument(
        "--precision",
        action="append",
        choices=PRECISIONS,
        help="repeatable (default: all of " + ", ".join(PRECISIONS) + ")",
    )
    args = parser.parse_args()
    keys = args.keys or list(GPT_NEO_CONFIGS)
    precisions = args.precision or list(PRECISIONS)
    # Keyed by prompt, because each prompt becomes one file holding every
    # model and precision, which is the comparison worth reading. Models stay
    # the outer loop so each checkpoint is loaded once.
    stories: dict[str, list[list[object]]] = {name: [] for name, _ in PROMPTS}
    for key in keys:
        fp32, tokenizer = load(key)
        for precision in precisions:
            print(f"\n{key} {precision}\n")
            model = convert(fp32, precision)
            for name, prompt in PROMPTS:
                decoded = tokenizer.decode(tokenizer.encode(prompt))
                assert decoded == prompt, name
                story = tell(model, tokenizer, prompt)
                words = len(story.split())
                print(f"  {name:<28}{words:>5} words")
                stories[name].append([key, precision, words, story])
    RESULTS.mkdir(parents=True, exist_ok=True)
    for index, (name, _) in enumerate(PROMPTS, start=1):
        output = csv_path(index, name)
        with open(output, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["model", "precision", "words", "story"])
            writer.writerows(stories[name])
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
