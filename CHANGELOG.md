# Changelog

Notable changes to Chimera are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/) once it reaches 1.0.

## [Unreleased]

### Added

- `?` operator for Rust-style error propagation on `Ok`/`Err` values.
- C `#define` constants and function-like macros.
- Kotlin `"key" to value` pairs and `downTo` ranges.
- Lua `local t = {…}` tables as lists.
- `isqrt`, `crc32`, `adler32` and `fnv1a` builtins.

### Changed

- Ranges print as `range(start, end)`, because `..` means different things
  in different dialects.

### Fixed

- `void` methods whose last statement is an expression no longer fail
  their return-type check.
- `//` floor division followed by a trailing `//` comment on the same line.
- Tracebacks raised inside generators point at the call that created the
  generator.

## [0.1.0] - 2026-10-12

The first release.

### Language

- One parser for Python, JavaScript/TypeScript, C, C++, Java, C#, Go, Rust,
  Ruby, Lua, Haskell, OCaml, Kotlin, Swift, SQL, LINQ and English, mixed
  freely in one file.
- Interchangeable block styles (braces, indentation, `do … end`/`end`).
- Classes, structs, records, data classes, enums/ADTs, traits/interfaces,
  `impl` blocks, operator overloading.
- Pattern matching (`match`, `case`, `when`, Haskell equations, destructuring).
- Resumable generators (`yield`, `function*`, `yield*`/`yield from`,
  `send`), Ruby blocks, Go `defer`.
- Arbitrary-precision integers.

### Machine learning

- N-d tensors with broadcasting and reverse-mode autograd.
- `f′`/`f″` (forward-mode duals) and `∇f` operators, plus `f(x) = …`
  definitions.
- `model` declarations and the `train … on … for N epochs using …`
  statement.
- PyTorch-, Keras- and scikit-learn-style APIs.

### AI

- `ai "…"`, `please …`, sentiment, classification, summaries, keywords,
  embeddings and Markov text, all available offline.
- Claude backend through the Messages API, and the Python SDK's
  `anthropic.Anthropic().messages.create(…)` shape.

### Queries

- SQL (`SELECT`/`JOIN`/`GROUP BY`/`HAVING`/`ORDER BY`/`LIMIT`, `INSERT`,
  `UPDATE`, `DELETE`) and C# LINQ over plain collections.

### Tooling

- `chimera` CLI: `run`, `-e`, `repl`, `check`, `explain`, `convert`,
  `tokens`, `ast`.
- Transpiler to Python, JavaScript, Ruby and Lua, verified against real
  interpreters.
