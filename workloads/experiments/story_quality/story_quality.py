"""Greedy TinyStories-Instruct completions for every published GPT-Neo size."""

import csv
import sys
from pathlib import Path

import torch
import transformers
from torch import nn

from configs.gpt_neo import GPT_NEO_CONFIGS, GPT_NEO_REPOS
from models.gpt_neo import gpt_neo

RESULTS = Path(__file__).resolve().parents[2] / "results" / "story_quality"
MAX_NEW_TOKENS = 320
END = "<|endoftext|>"

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


def tell(model: nn.Module, tokenizer: object, prompt: str) -> str:
    """The continuation alone, cut at the first end-of-text marker."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, MAX_NEW_TOKENS)
    # generate() has no EOS stop and always emits MAX_NEW_TOKENS, so the tail
    # runs on into a fabricated next example unless it is cut here.
    text = tokenizer.decode(output[0, input_ids.shape[1] :])
    return text.split(END)[0].strip()


def main() -> None:
    keys = sys.argv[1:] or list(GPT_NEO_CONFIGS)
    # Keyed by prompt, because each prompt becomes one file holding every
    # model, which is the comparison worth reading. Models stay the outer
    # loop so each checkpoint is loaded once.
    stories: dict[str, list[list[object]]] = {name: [] for name, _ in PROMPTS}
    for key in keys:
        print(f"\n{key}\n")
        model, tokenizer = load(key)
        for name, prompt in PROMPTS:
            assert tokenizer.decode(tokenizer.encode(prompt)) == prompt, name
            story = tell(model, tokenizer, prompt)
            words = len(story.split())
            print(f"  {name:<28}{words:>5} words")
            stories[name].append([key, words, story])
    RESULTS.mkdir(parents=True, exist_ok=True)
    for index, (name, _) in enumerate(PROMPTS, start=1):
        # Numbered so the files sort in the order the prompts add constraints.
        output = RESULTS / f"{index}_{name}.csv"
        with open(output, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["model", "words", "story"])
            writer.writerows(stories[name])
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
