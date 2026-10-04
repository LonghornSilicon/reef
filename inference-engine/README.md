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

The build produces the C++ library and `matrix_cli` executable. Google Test is fetched during the first CMake configuration.
