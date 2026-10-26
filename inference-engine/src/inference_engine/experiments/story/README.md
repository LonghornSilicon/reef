# story

Continues a text prompt with GPT-Neo on the host C++ engine. Python
tokenizes the prompt, `gpt_neo_cli` generates greedily, and Python decodes
the result. The output stops before the first end-of-text token, because
past it the model starts an unrelated new story.

## Get a model

A model directory holds a Hugging Face `config.json`, `model.safetensors`
and `tokenizer.json`. TinyStories-28M only has `model.safetensors` on its
`refs/pr/1` branch:

```sh
mkdir -p models/tinystories-28m && cd models/tinystories-28m
hf=https://huggingface.co/roneneldan/TinyStories-28M/resolve
curl -fLO "$hf/main/config.json"
curl -fLO "$hf/main/tokenizer.json"
curl -fLO "$hf/refs%2Fpr%2F1/model.safetensors"   # 242 MB
```

## Run

From `inference-engine/`:

```sh
cmake -S . -B build && cmake --build build --target gpt_neo_cli
uv run python -m inference_engine.experiments.story.main \
    --model models/tinystories-28m "Once upon a time, there was a dog named Max."
```

| Option | Default | Meaning |
|---|---|---|
| `--precision` | `fp32` | `fp32` or `w8a8` |
| `--new-tokens` | `100` | tokens to generate before cutting at end-of-text |
| `--executable` | `build/bin/gpt_neo_cli` | engine binary |

The engine is a readable reference, not a fast one: 120 tokens of the 28M
model take about 30 seconds on an M-series Mac.
