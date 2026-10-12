# Contributing to Chimera

Thanks for wanting to help build a language that speaks every language. This
guide covers getting set up, the shape of the code, step-by-step recipes for
the most common contributions, and a list of issue ideas at every level.

If you are here for **Hacktoberfest** 🎃, welcome. Read the
[Hacktoberfest notes](#hacktoberfest) first. They explain what counts and
what doesn't.

## Getting set up

You need Rust 1.88 or newer (`rustup update stable`). There are no other
dependencies.

```bash
git clone https://github.com/<you>/hacktober && cd hacktober
cargo build                       # debug build (already lightly optimised)
cargo test                        # everything: unit, golden, examples, transpiler
cargo run -- examples/hello.chi   # run a program
cargo run -- repl                 # play
```

Before you open a pull request, run the same checks CI runs:

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace
```

The transpiler round-trip test also runs `python3` and `node` when they are
installed. Without them it skips those checks rather than failing.

## Where things live

```text
crates/chimera-core/src/        language-agnostic utilities (RNG, diagnostics, tables, text)
crates/chimera-lang/src/
├── lexer.rs                    context-sensitive tokens (`//`, `#`, backticks, `$`, lifetimes…)
├── parser/
│   ├── mod.rs                  parser state, dialect flags, blocks (indent / braces / `end`)
│   ├── stmt.rs                 statements: every language's def/fn/func, if/for/while, classes…
│   ├── expr.rs                 Pratt expression parser, calls, literals, lambdas, comprehensions
│   ├── pattern.rs              patterns for match/case/destructuring, type annotations
│   ├── query.rs                SQL and LINQ
│   └── ml.rs                   `model` and `train`
├── ast.rs                      the single AST every dialect parses into
├── interp/                     tree-walking interpreter
│   ├── mod.rs                  statements, expressions, calls, closures
│   ├── ops.rs                  operators and their polyglot semantics
│   ├── class.rs                classes, structs, enums, traits, attribute lookup
│   ├── pattern.rs              pattern matching
│   ├── generator.rs            resumable generators
│   ├── format.rs               printing, repr, printf/format strings
│   └── query.rs                SQL execution
├── builtins/
│   ├── core.rs                 print, len, map, range… (hundreds of names and aliases)
│   ├── methods.rs              methods on strings, lists, maps (`.map`, `.each`, `.push_str`…)
│   ├── objects.rs              namespace objects: console, Math, System, fmt, std, IO…
│   ├── modules.rs              importable modules: math, random, json, time, itertools…
│   ├── math.rs, json.rs        numerics and JSON
│   └── prelude.chi             library code written in Chimera itself
├── ml/                         tensors, autograd, dual numbers, layers, optimisers
├── ai/                         offline NLP, the Claude backend, `please`
├── convert.rs                  the transpiler (Python, JavaScript, Ruby, Lua)
└── lib.rs                      the embedding API (`Engine`)
crates/chimera-lang/tests/
├── golden/*.chi + *.out        golden programs and their expected output
├── convert/                    transpiler round-trip program
└── *.rs                        test harnesses
crates/chimera/src/cli.rs       the command-line tool and REPL
examples/                       runnable programs, each one run by the test suite
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how a program travels
through these pieces. [docs/LANGUAGE.md](docs/LANGUAGE.md) lists the
disambiguation rules.

## Recipes

### 1. Add a function written in Chimera

The easiest contribution needs no Rust at all. Everything defined in
`crates/chimera-lang/src/builtins/prelude.chi` becomes a builtin available to
every program:

```python
// Linear interpolation between a and b.
def lerp(a, b, t) = a + (b - a) * t
```

Then add a line that uses it to a golden test (recipe 5) and you're done.

### 2. Add a builtin written in Rust

Use Rust when you need speed or access to the host. Builtins are registered
with `def`, which takes every name the function goes by. Aliases are how one
function serves several languages:

```rust
// crates/chimera-lang/src/builtins/math.rs
def(interp, &["clamp", "coerceIn"], |interp, args| {
    let x = float_arg(interp, &args, 0, "clamp")?;
    let lo = float_arg(interp, &args, 1, "clamp")?;
    let hi = float_arg(interp, &args, 2, "clamp")?;
    Ok(Value::Float(x.max(lo).min(hi)))
});
```

`arg`, `int_arg`, `float_arg` and `str_arg` (in `builtins/mod.rs`) fetch
arguments and report friendly type errors. Return errors with
`interp.type_err(...)` or `interp.value_err(...)`, so users get a proper
traceback.

**Methods** (`xs.chunked(2)`, `s.capitalize()`) live in
`builtins/methods.rs`, which dispatches on the receiver's type and the method
name. If a method takes no arguments and Ruby or Kotlin users would write it
without parentheses (`xs.first`, `s.length`), add it to the bare-property list
at the top of that file.

**Namespace objects** (`fmt.Println`, `console.log`, `Math.max`) are built in
`builtins/objects.rs` with `module(name, alias(interp, &[("Println",
"println"), …]))`. **Importable modules** (`import math`, `from itertools
import …`) go in `builtins/modules.rs`.

### 3. Teach the parser a new construct

This is where the interesting work is. The pattern almost always goes like
this:

1. **Find the decision point.** Statements start in
   `Parser::statement_inner` (`parser/stmt.rs`), which dispatches on the first
   word. Expressions go through the Pratt loop in `parser/expr.rs`
   (`expr_bp`): prefix operators in `prefix`, atoms in `primary` (with
   keyword-led forms in `word_primary`), infix operators in `binary_op` with
   binding powers from the `bp` module, and postfix forms (calls, indexing,
   `.method`) in `postfix_ops`.
2. **Recognise your construct from local context.** Use `self.at_word("x")`,
   `self.nth_is_op(1, "(")`, `self.same_line()` and similar helpers. When you
   can't tell from a fixed lookahead, try a parse with `self.attempt(|p| …)`.
   It rewinds on failure, so a speculative parse costs nothing when it doesn't
   match.
3. **Desugar into the existing AST** whenever you can. Elixir's `x |> f(y)`
   becomes the call `f(x, y)`, and Kotlin's `listOf(…)` is a prelude function.
   A new `ExprKind`/`StmtKind` is only needed for genuinely new semantics, and
   then the interpreter (`interp/mod.rs`) and the transpiler (`convert.rs`) need
   a case too.
4. **Run the whole test suite.** Every syntax shares one grammar, so a new
   rule can steal tokens from another language. The golden tests exist to
   catch exactly that.
5. **Document the rule** in [docs/LANGUAGE.md](docs/LANGUAGE.md) if it
   resolves an ambiguity.

Parser flags in `parser/mod.rs` (`no_brace`, `eq_is_cmp`, `rust_ranges`, …)
carry context such as "we're inside a Rust function". Save a flag before you
change it and restore it afterwards, the way `fn_decl` does with
`rust_ranges`. `in_header` and `nested` do this for the common header and
bracket cases.

`chimera explain file.chi`, `chimera ast file.chi` and `chimera tokens
file.chi` show you exactly how something was understood.

### 4. Make a runtime behaviour match a language

Operator semantics are in `interp/ops.rs`. Printing (Python lists print with
quotes, Ruby's `p` vs `puts`, printf verbs) is in `interp/format.rs`. When two
languages disagree (what does `"3" + 4` do?), Chimera picks the reading that
is least surprising across languages and documents it. Open an issue to
discuss before changing an existing choice.

### 5. Add a golden test

Golden tests are Chimera programs with their expected output next to them:

```text
crates/chimera-lang/tests/golden/my_feature.chi
crates/chimera-lang/tests/golden/my_feature.out
```

Write the `.chi`, then generate the `.out`:

```bash
BLESS=1 cargo test -p chimera-lang --test golden
git diff crates/chimera-lang/tests/golden/   # READ the new output. Is it right?
```

`BLESS=1` writes down whatever the program printed, bugs included, so check
it by hand. Golden programs run with the deterministic offline AI backend, so
their output never depends on the network.

Programs in `examples/` are run by the test suite too (they must not fail),
but their output isn't compared.

### 6. Add a transpiler target

`convert.rs` holds an `Emitter` that walks the AST, with one branch per
`Target` for the places where languages differ. To add a target, add a `Target`
variant (`from_name`, `extension`), handle it in the emitter, and add a case
to `tests/convert_roundtrip.rs` if the target language's interpreter is
commonly installed. Constructs a target can't express are emitted as a call
to `chimera("what", "original source")`, so the output always parses and
the gaps are easy to find.

## Code guidelines

- **No dependencies.** The workspace builds with the standard library alone.
  That's a deliberate constraint. Please don't add crates.
- **No `unsafe`.** It's forbidden at the workspace level.
- **Clippy clean** with `-D warnings`, and `cargo fmt` formatted.
- **Doc comments** on public items and on any non-obvious heuristic. Explain
  *why* a rule exists and give an example of the syntax it handles, like
  ``// `x = f(y)   // note`: aligned trailing comments``.
- **Errors are for users.** Say what was expected and what was found, point
  at the right span, and suggest a fix when you can.
- **Tests with every change.** A behaviour change without a golden or unit
  test will probably come back later as a bug.

## Pull requests

- Keep each PR to one feature or fix. Small PRs get reviewed fast.
- Describe what you changed and show a snippet that works now and didn't
  before.
- If your change resolves a syntax ambiguity, explain which reading you chose
  and why.
- CI must pass. If it fails in a way you don't understand, say so in the PR
  and we'll help.

## Hacktoberfest

- Pull requests merged, approved, or labelled `hacktoberfest-accepted` during
  October count toward Hacktoberfest.
- **Quality over quantity.** PRs that only fix whitespace, rename things
  arbitrarily, or add your name to a list will be closed and labelled
  `spam`/`invalid`, which counts against you on the Hacktoberfest site.
- Comment on an issue before you start, so two people don't build the same
  thing. If you've gone quiet for a week, someone else may pick it up.
- First time contributing to open source? Look for the `good first issue`
  label or pick something from the **good first issues** below. Ask questions
  in the issue; there are no silly ones.

## Issue ideas

These are real gaps, checked against the current interpreter. Each one makes
a satisfying pull request.

### Good first issues (builtins and prelude, no parser work)

- **Go's `strings` and `strconv` packages**: `strings.ToUpper`,
  `strings.Split`, `strings.Contains`, `strings.Repeat`, `strconv.Itoa`,
  `strconv.Atoi`. A namespace object in `builtins/objects.rs`, aliasing
  existing builtins.
- **OCaml's printing functions**: `print_int`, `print_float`,
  `print_endline`, `string_of_int`, `int_of_string`.
- **MATLAB/Octave helpers**: `numel`, `isempty`, `fliplr`.
- **Ruby enumerators without a block**: `10.times.to_a`, `3.times.map { … }`,
  `"abc".each_char.to_a`.
- **`lerp`, `clamp`, `remap`** in the prelude.
- **More Ruby string methods**: `tr`, `squeeze`, `succ`. Check
  `methods.rs` for what already exists.
- **More examples**: a small game, a JSON pretty-printer, a matrix library,
  a text adventure in English syntax. Put them in `examples/`.

### Intermediate (parser work in one place)

- **Scala placeholder lambdas**: `xs.map(_ * 2)`, `xs.reduce(_ + _)`.
- **Julia/R/MATLAB ranges**: `1:5` outside slices and dict literals.
- **MATLAB matrices**: `[1 2 3; 4 5 6]` (space-separated elements).
- **Elixir function heads**: `def f(x), do: x * 2` inside `defmodule … end`.
- **Scala's `object Main extends App { … }`**.
- **Java streams**: `list.stream().filter(…).map(…).collect(Collectors.toList())`.
- **Transpiler**: support `match`/pattern matching in the JavaScript and
  Lua targets.

### Advanced

- **A Lisp dialect**: `(defn sq [x] (* x x))`. S-expressions are easy to
  parse but hard to tell apart from parenthesised expressions. Design first,
  then open an issue.
- **Perl sigils**: `my @xs = (1, 2, 3); print scalar(@xs);`
- **A Go or Rust transpiler target.**
- **A bytecode compiler.** The tree-walker is the slowest part of the system.
  `chimera-core::codec` was written for a bytecode cache.
- **Tensor performance**: blocked matrix multiply, or reusing buffers in the
  autograd tape.
- **Streaming `ai` responses** in the Claude backend.

If you find a program that should work and doesn't, the smallest example
that fails is a great issue on its own.

## Code of conduct

Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Be kind and assume good faith.
