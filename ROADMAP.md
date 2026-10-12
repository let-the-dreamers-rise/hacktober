# Roadmap

Chimera's goal is simple to state and impossible to finish: **any
reasonable line of code, in any popular language, should just run.** This
roadmap lists the next steps. Every item is open to contributors. Comment on
(or open) an issue before starting a big one.

## Now: breadth

- [ ] **Regular expressions**: an in-tree engine behind Python `re`,
      JavaScript `/…/` literals and Ruby `=~`/`gsub`.
- [ ] **More dialects**: Elixir function heads, Scala placeholder lambdas
      (`_ * 2`), Julia/R/MATLAB ranges and matrices, Perl sigils, a Lisp
      S-expression mode.
- [ ] **Fill out standard libraries**: Go `strings`/`strconv`, Java
      streams and `Collectors`, Ruby enumerators, OCaml `Printf`/`List`.
- [ ] **Transpiler**: pattern matching for the JavaScript and Lua targets;
      new targets (Go, Rust, Kotlin).
- [ ] **Better error messages for dialect mix-ups**, for example "this
      looks like Rust's `..` (exclusive) inside a Ruby block".

## Next: depth

- [ ] **Bytecode compiler and VM.** The tree-walker is simple but slow.
      A register VM, with `chimera-core::codec` as the cache format, could
      make Chimera 5–20× faster.
- [ ] **Static analysis**: `chimera check` could warn about undefined
      names, arity mismatches and unreachable code before running.
- [ ] **Optional gradual typing**: enforce the type annotations Chimera
      already parses (`x: int`, `int x`, `x int`, `val x: Int`).
- [ ] **Tensor performance**: blocked matmul, buffer reuse in the autograd
      tape, convolution layers.
- [ ] **Streaming AI**: stream Claude responses into `print` as they
      arrive; a tool-use bridge so models can call Chimera functions.

## Later: reach

- [ ] **Language server (LSP)**: hover shows the dialect and canonical form
      (the `explain` output), go-to-definition across dialects.
- [ ] **WebAssembly build and an online playground.** The interpreter has
      no dependencies, so it should compile to `wasm32` with few changes.
- [ ] **Package imports across dialects**: `import` a `.py`-, `.js`- or
      `.rb`-flavoured file and call into it.
- [ ] **A Jupyter kernel**, since tensors and plots belong in notebooks.

## Non-goals

- **Full compliance with any one language.** If you need all of Python, use
  Python. Chimera aims at the large common core that people actually write.
- **Calling native libraries** (NumPy, npm packages, crates). Chimera stays
  self-contained. API shapes are emulated, not bound.
- **Being a sandbox.** See [SECURITY.md](SECURITY.md).
