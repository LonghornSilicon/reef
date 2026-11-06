"""Test B: tensor-parallel GPT-Neo against one chip on real TinyStories text.

Packs validation stories into fixed-length token sequences, runs tp_compare
on them for every precision and scheme, a few runs at once, and prints a
Markdown table. tp_compare feeds every model the same tokens (teacher
forcing) and compares the logits at every position.

    uv run python dist-reference/test_b.py --model models/tinystories-28m
        --stories models/data/TinyStories-valid.txt
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tokenizers import Tokenizer

# inference-engine/, where `cmake -S . -B build` puts build/bin/tp_compare.
MODULE_ROOT = Path(__file__).resolve().parents[1]
END_OF_TEXT = "<|endoftext|>"

# (precision, scheme, rank counts, extra flags): one tp_compare run each.
# Against fp32, rank count 1 is one chip in W8A8: the error W8A8 itself adds.
RUNS = [
    ("fp32", "megatron", "2,4,8", []),
    ("w8a8", "megatron", "2,4,8", []),
    ("w8a8", "megatron", "1,2", ["--reference", "fp32"]),
    ("w8a8", "megatron", "4,8", ["--reference", "fp32"]),
    ("fp32", "column", "4", []),
    ("w8a8", "column", "4", []),
]
COLUMNS = [
    "precision",
    "scheme",
    "ranks",
    "reference",
    "positions",
    "max_rel",
    "mean_rel",
    "argmax_agree",
    "ppl_reference",
    "ppl_split",
]


def pack_stories(
    text: str, tokenizer: Tokenizer, length: int, count: int
) -> list[list[int]]:
    """Cut consecutive stories, each ended by <|endoftext|>, into sequences.

    Stories are packed back to back, as in training, so a sequence can
    start mid-story; every sequence holds exactly length tokens.
    """
    end = tokenizer.token_to_id(END_OF_TEXT)
    tokens: list[int] = []
    for story in text.split(END_OF_TEXT):
        if story.strip():
            tokens += [*tokenizer.encode(story.strip()).ids, end]
        if len(tokens) >= length * count:
            break
    return [tokens[i * length : (i + 1) * length] for i in range(count)]


def main() -> None:
    """Tokenize, run every comparison in parallel, print the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stories", type=Path, required=True)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--sequences", type=int, default=2)
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="tp_compare runs at once; each needs about 650 MB",
    )
    parser.add_argument(
        "--out", type=Path, default=MODULE_ROOT / "results" / "test_b"
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=MODULE_ROOT / "build" / "bin" / "tp_compare",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    sequences = pack_stories(
        args.stories.read_text(), tokenizer, args.length, args.sequences
    )
    args.out.mkdir(parents=True, exist_ok=True)
    tokens_file = args.out / "tokens.txt"
    tokens_file.write_text(
        "".join(" ".join(map(str, ids)) + "\n" for ids in sequences)
    )

    def run(precision: str, scheme: str, ranks: str, extra: list[str]) -> str:
        return subprocess.run(
            [
                str(args.executable),
                "--model",
                str(args.model),
                "--tokens",
                str(tokens_file),
                "--precision",
                precision,
                "--scheme",
                scheme,
                "--tp",
                ranks,
                *extra,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        outputs = list(pool.map(lambda spec: run(*spec), RUNS))
    rows = [
        line.split("\t") for output in outputs for line in output.splitlines()
    ]

    table = "| " + " | ".join(COLUMNS) + " |\n"
    table += "|" + "---|" * len(COLUMNS) + "\n"
    table += "".join("| " + " | ".join(row) + " |\n" for row in rows)
    (args.out / "results.md").write_text(table)
    print(table, end="")


if __name__ == "__main__":
    main()
