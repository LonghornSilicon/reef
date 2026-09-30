# Tests to add with implementation

- `python/`: pytest checks for host build/run orchestration, reference-output
  comparison, and a short experiment run.
- `cpp/`: focused kernel and model tests using a C++ test framework selected
  with the implementation. A pytest fixture should build and invoke them.
- Mark simulator tests `slow` if they cannot run in the fast local suite.

The current Python test only checks package installation. There is no test of
inference behavior yet.
