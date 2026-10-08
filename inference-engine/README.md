# Inference Engine

This module is the starting point for a minimal inference engine in Reef.

## Requirements

- Python 3.12 and `uv`
- CMake 3.20 or newer and a C++17 compiler
- Doxygen for the full test suite
- Network access on the first CMake build to fetch Google Test

Run the commands below from `inference-engine/`.

## Build and test the Python and C++ flow

```sh
uv sync --locked
uv run pytest
```

This builds the C++ targets, runs their Google Test cases through CTest, tests the Python-to-C++ matrix call, and checks linting and generated C++ documentation.

To run a subset of tests:

```sh
uv run pytest -m unit
uv run pytest -m "not slow"
```

## Build and test C++ alone

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The build produces the C++ library and `tensor_cli` executable. Google Test is fetched during the first CMake configuration.

## Contributing

When adding a component, put public C++ declarations in
`src/csrc/include/inference_engine/` and their implementations in
`src/csrc/src/`. Add new implementation files to `CMakeLists.txt` and cover
the component with Google Test cases in `tests/cpp/`. Put importable Python
code in `src/inference_engine/` and add an integration test when Python calls
new C++ functionality.

Within `src/`, only experiments should run standalone from the command line.
Put each experiment under `src/inference_engine/experiments/` with its own README,
`main.py` for one artifact, `run.py` for a configured sweep, and `plot.py`
for the sweep results. Do not add a main function elsewhere in `src/`.

The following is a growing list of rules that must be checked before a PR is
complete:

1. `nullptr` represents only unallocated memory (for example, a pointer set
   to `nullptr` after `delete`). It never represents an optional value or
   argument; use `std::optional` for those.
2. New C++ behavior has Google Test coverage, and any new Python-to-C++ path
   has a Python integration test. Mark every pytest case as `unit`,
   `integration`, or `experiment`, and mark long-running cases as `slow`.
3. `bash tools/lint.sh` passes. It applies automatic Python fixes and C++
   formatting, then checks both languages. Review its edits before committing.
4. `uv run pytest` passes. For C++ changes, also run the C++-only build and
   CTest commands above. The full pytest run checks Ruff, clang-format,
   clang-tidy, and Doxygen in addition to the tests.
