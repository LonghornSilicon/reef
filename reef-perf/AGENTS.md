# Instructions for coding agents

Shared by everyone who uses an AI assistant on reef-perf (Claude Code reads it
through CLAUDE.md; Codex and others read AGENTS.md). Humans follow the same
rules; README.md → Contributing has the full list.

## Before you change anything

- Read README.md, docs/interfaces.md and docs/glossary.md. Use the glossary's
  terms with its meanings: a **module** is one of the five owner-assigned
  parts, a **unit** is a Sparta unit, a **resource** is one member of a pool.
- Each module has an owner. Change another module only through its interface,
  and change an interface only together with both sides and interfaces.md.
- Don't tune timing to match the RTL unless asked. Calibration belongs to
  each module's owner.
- Everything builds and runs inside the Docker image (`../tools/docker.sh
  shell`). Don't install host packages.

## Code

- C++ follows the LLVM coding standards: `UpperCamelCase` types, variables
  and members, `lowerCamelCase` functions, `REEF_PERF_<DIR>_<FILE>_H` guards,
  `//===- File - Description ---===//` banners. Sparta-defined names keep
  Sparta's spelling, marked `NOLINT`.
- `struct` when every member is public and there is no invariant; `class`
  otherwise. Don't declare copy or move members unless the defaults would be
  wrong.
- No thin wrappers: no class or file that only forwards to another.
- Lean comments. `///` only on public declarations, one line where possible;
  plain `//` only where the code isn't obvious.
- Use the purpose-defined types in `common/Types.h` (`Cycle`, `Address`,
  `RegId`).
- American spelling (behavior, modeled, utilization).
- Python: Ruff-formatted, Google-style docstrings, type annotations.

## Checks

- `bash scripts/lint.sh` and `uv run pytest` must pass.
- A change that only moves code must leave `tests/golden/m3_results.json`
  unchanged. A change meant to alter timing regenerates it and says why in the
  commit message.
- Don't push or open pull requests unless asked.

## Writing (docs, commit messages, PR text, comments)

Write like an engineer explaining the work to a colleague:

- Plain, direct sentences. Lead with the point; give the reason after.
- Short paragraphs. Use lists and tables for genuinely parallel items, not to
  chop up an argument.
- Say what something is or does, concretely: file paths, names, numbers.
  Prefer "the LSU handles one memory instruction at a time" to "the LSU has
  limited concurrency".
- Bold only the few words a reader skimming should catch. No emoji, no
  headings on a three-line answer.
- Don't oversell: no "robust", "seamless", "comprehensive", "powerful", "key
  insight". Don't announce what you're about to say; say it.
- Don't refer to other companies' products or internal tools.
- State limits and uncertainty plainly ("not modeled yet", "an assumption;
  no M3 source").
