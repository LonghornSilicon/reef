# Tests

`uv run pytest` is the module's single test entry point. The session fixture
builds the C++ targets in a temporary directory.

- `cpp/test_matrix.cpp` checks native matrix multiplication and shape errors;
  pytest launches it through CTest.
- `python/test_matrix.py` sends matrices through the Python-to-C++ command-line
  bridge and checks the returned result.
- `python/test_package.py` checks that the Python package imports.

TODO: Add focused operator tests, a short end-to-end inference experiment,
and simulator tests as those components are implemented. Mark simulator tests
`slow` when they are too expensive for the fast local suite.
