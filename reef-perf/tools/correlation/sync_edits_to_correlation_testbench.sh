#!/usr/bin/env bash
# Copies the uncommitted edits of the checkout you run it from onto the RTL
# correlation test bench, rebuilds it and runs the C++ tests.
#
# The test bench is a git worktree inside the reef Docker container whose
# commits are the unmerged work the RTL comparison needs (for example the
# fetch fixes B04-B06). Only uncommitted edits travel: the test bench is reset
# to its own HEAD first, so commit-and-cherry-pick any committed work that
# later edits build on. New files reach the diff only once git knows them
# (`git add -N <file>`); staged renames and deletions don't (unstage them).
#
# Run on the host, from anywhere in the checkout whose edits should travel
# (the script itself may live elsewhere, e.g. a copy taken from the
# orhan/correlation-tools branch):
#   bash /path/to/sync_edits_to_correlation_testbench.sh
# Settings (environment):
#   REEF_CONTAINER   the reef Docker container (default orhan-reef-rebase)
#   TESTBENCH        the test bench path in it
#                    (default /tmp/rtl-correlation-testbench)
set -euo pipefail

CONTAINER=${REEF_CONTAINER:-orhan-reef-rebase}
TESTBENCH=${TESTBENCH:-/tmp/rtl-correlation-testbench}
REPO=$(git rev-parse --show-toplevel)
DIFF=$(mktemp)
trap 'rm -f "$DIFF"' EXIT

git -C "$REPO" diff > "$DIFF"
# Files the diff creates (new or renamed): remove them in the test bench
# first, since the reset keeps untracked files and `git apply` refuses to
# create a file that already exists.
NEW_FILES=$(git -C "$REPO" diff --name-only --diff-filter=AR | tr '\n' ' ')
# git apply rejects an empty diff.
APPLY='git apply -'
[ -s "$DIFF" ] || APPLY=true

docker exec -i "$CONTAINER" bash -c "cd $TESTBENCH && git reset -q --hard \
  && rm -f $NEW_FILES && $APPLY && cd reef-perf \
  && { cmake --build build -j 4 2>&1 | grep -E 'error|warning:' || true; } \
  && cd build && ctest 2>&1 | grep -E 'tests passed|tests failed'" < "$DIFF"
