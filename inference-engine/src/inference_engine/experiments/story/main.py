"""Print a prompt and GPT-Neo's greedy continuation, run by the C++ engine."""

import argparse
from pathlib import Path

from inference_engine.gpt_neo import PRECISIONS, tell

# inference-engine/, where `cmake -S . -B build` puts build/bin/gpt_neo_cli.
MODULE_ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    """Parse the command line and print the story."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="text for the model to continue")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="directory with config.json, model.safetensors, tokenizer.json",
    )
    parser.add_argument("--precision", choices=PRECISIONS, default="fp32")
    parser.add_argument("--new-tokens", type=int, default=100)
    parser.add_argument(
        "--executable",
        type=Path,
        default=MODULE_ROOT / "build" / "bin" / "gpt_neo_cli",
        help="gpt_neo_cli to run (default: build/bin/gpt_neo_cli)",
    )
    args = parser.parse_args()
    story = tell(
        args.prompt,
        args.new_tokens,
        args.model,
        args.executable,
        args.precision,
    )
    print(args.prompt + story)


if __name__ == "__main__":
    main()
