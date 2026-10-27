# Inference Engine

A minimal GPT-Neo inference engine for Reef, in two parts:

- `reference/`: the host C++ engine, which is the **source of truth** for
  the arithmetic, and its Python interface.
- `coral/`: RVV kernels, firmware and benchmarks for the Coral NPU core.
  Each kernel reproduces an engine function bit for bit, and
  `coral/tools/test.sh` fails on any difference. See
  [coral/README.md](coral/README.md).

```
reference/
  cpp/
    include/inference_engine/   the engine
    tests/                      its Google Test cases
    gpt_neo_cli.cpp             runs TinyStories end to end in fp32 or W8A8
  python/
    inference_engine/           Python interface (model.py), for runs and sweeps
    tests/test_reference.py     pytest: C++ tests, Python interface, lint, docs
coral/
  tests/test_coral.py           pytest: every kernel against the engine
```

Each part keeps its tests beside its code; `uv run pytest` runs both pytest
files.

## Requirements

- Python 3.12 and `uv`
- CMake 3.20 or newer and a C++17 compiler
- Doxygen for the full test suite
- Docker for the Coral kernel tests (see below)
- Network access on the first CMake build to fetch Google Test

Run the commands below from `inference-engine/`.

## Build and test the Python and C++ flow

```sh
uv sync --locked
uv run pytest
```

This builds the C++ targets, runs their Google Test cases through CTest, runs GPT-Neo from Python through the C++ engine, checks linting and generated C++ documentation, and checks the Coral kernels against the engine on reef_perf.

To run a subset of tests:

```sh
uv run pytest -m "not slow"
uv run pytest -m "not coral"   # without Docker or the Coral toolchain
```

## Build and test C++ alone

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The build produces the executables `gpt_neo_cli`, `engine_tests` (every
Google Test case), and coral's host tools `gen_golden` and `pack_model`.
Google Test is fetched during the first CMake configuration.

## Run a model

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

Text in, text out, through Python: the package tokenizes the prompt, runs
`gpt_neo_cli` and decodes the result, stopping before the first
end-of-text token because past it the model starts an unrelated new story.

```sh
cmake -S . -B build && cmake --build build --target gpt_neo_cli
uv run python -m inference_engine \
    --model models/tinystories-28m "Once upon a time, there was a dog named Max."
```

| Option | Default | Meaning |
|---|---|---|
| `--precision` | `fp32` | `fp32` or `w8a8` |
| `--new-tokens` | `100` | tokens to generate before cutting at end-of-text |
| `--executable` | `build/bin/gpt_neo_cli` | engine binary |

The engine is a readable reference, not a fast one: 120 tokens of the 28M
model take about 30 seconds on an M-series Mac.

`gpt_neo_cli` is the engine underneath. It takes a model directory (it reads
`config.json` and `model.safetensors`) plus a prompt as token IDs, and
prints the greedy continuation as token IDs:

```sh
build/bin/gpt_neo_cli --model DIR [--precision fp32|w8a8] --new-tokens 20 7454 2402 257 640 11
```

From Python, `inference_engine.model.generate()` calls it with token IDs
and `tell()` with text.

## Check the Coral kernels against the engine

```sh
coral/tools/setup_toolchain.sh   # once
coral/tools/test.sh
```

`uv run pytest` runs `coral/tools/test.sh` through `coral/tests/test_coral.py`.

## Contributing

When adding a component, put it in a header under
`reference/cpp/include/inference_engine/` and cover it with Google Test
cases in a `reference/cpp/tests/test_*.cpp`, listed in `engine_tests` in
`CMakeLists.txt`. Code that runs on the Coral core goes in `coral/` (see its
README). Put importable Python code in `reference/python/inference_engine/`
and add a case to `reference/python/tests/test_reference.py` when Python
calls new C++ functionality.

Within `reference/python/`, only `__main__.py` and experiments run
standalone from the command line. Put each experiment, such as a sweep,
under `reference/python/inference_engine/experiments/` with its own README,
`main.py` for one artifact, `run.py` for a configured sweep, and `plot.py`
for the sweep results. Do not add a main function elsewhere in
`reference/python/`.

The following is a growing list of rules that must be checked before a PR is
complete:

1. `nullptr` represents only unallocated memory (for example, a pointer set
   to `nullptr` after `delete`). It never represents an optional value or
   argument; use `std::optional` for those.
2. New C++ behavior has Google Test coverage, and any new Python-to-C++ path
   has a case in `reference/python/tests/test_reference.py`. Mark
   long-running pytest cases as `slow`.
3. `bash tools/lint.sh` passes. It applies automatic Python fixes and C++
   formatting, then checks both languages. Review its edits before committing.
4. `uv run pytest` passes. For C++ changes, also run the C++-only build and
   CTest commands above. The full pytest run checks Ruff, clang-format,
   clang-tidy, and Doxygen in addition to the tests.
5. The engine is the source of truth for the kernels. A change to a Coral
   kernel, or to any engine function a kernel matches (see
   `coral/include/reef_kernels/kernels.hpp`), passes `coral/tools/test.sh`
   (the `coral` pytest case). To change the arithmetic, change the engine
   first and its tests, then the kernel.
6. C++ code is templated on its value type (`Tensor<Scalar>`,
   `GPTNeo<Scalar>`), with a separate accumulator type where sums need more
   range than the inputs (`matmul<std::int8_t, std::int32_t>`). Hard-code
   `float` only where a data format fixes it, such as the F32 checkpoint
   loader.
7. The engine is the readable, checked reference; speed belongs in the
   kernels. Access tensor elements with `.at()` or a computed index, not
   pointer or iterator arithmetic. Use `safe_add` and `safe_multiply`
   (`util.hpp`) for sums and products that could overflow, so overflow,
   infinity, and NaN throw instead of propagating, and make unsupported
   value types fail to compile.
8. Shared code stays general. A base class or shared helper does not name
   its subclasses, a particular model, or the kernels that mirror it, and
   comments stay true as code is added (the engine *may be* header-only).
   Model code such as `GPTNeo` and `weights.hpp` may name its model.
9. When two classes differ only by a parameter, they share one function
   that takes it (as `causal_visibility` serves global and local attention)
   instead of copying the logic.
10. Every Google Test case has a comment above it saying what it checks.
11. Add plain `assert`s for internal invariants, alongside the exceptions
    that report bad input.
12. Headers under `reference/cpp/include/` are documented with Doxygen, with every
    parameter described. Every `NOLINT` has a comment saying why the check
    does not apply.
13. The PR description follows `.github/pull_request_template.md`, including
    a Test section with the exact commands and roughly how long they take.
