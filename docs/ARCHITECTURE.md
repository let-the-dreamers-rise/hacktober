# Architecture

This is a guided tour of how Chimera turns a file written in a dozen
languages into running code. It's written for contributors: read it once,
then use it as a map.

```text
            ┌──────────┐   ┌──────────┐   ┌─────┐   ┌─────────────┐
 source ───►│  lexer   │──►│  parser  │──►│ AST │──►│ interpreter │──► output
            └──────────┘   └──────────┘   └─────┘   └─────────────┘
              tokens with     one grammar     │       values, envs, flows
              context-        for every       │
              sensitive       dialect         ├──► convert.rs  (Python / JS / Ruby / Lua source)
              decisions                       └──► explain     (dialect + canonical form)
```

There are three crates:

| Crate | Role |
|---|---|
| `chimera-core` | Language-agnostic building blocks: `Span`/`Diagnostic` and rustc-style rendering, a seedable xoshiro256\*\* RNG, checksums, box-drawn tables, edit-distance suggestions. |
| `chimera-lang` | The language: lexer, parser, AST, interpreter, builtins, ML, AI, queries and the transpiler. Exposes `Engine`. |
| `chimera` | The command-line tool: argument parsing, `run`, `repl`, `explain`, `convert`, `check`, `tokens` and `ast`. |

## The embedding API

Everything goes through `chimera_lang::Engine`:

```rust
let mut engine = Engine::offline();        // or Engine::new() for env-configured AI
engine.capture_output();                   // collect output instead of printing it
engine.run("demo.chi", source)?;           // lex + parse + run
let printed = engine.take_output();
```

`Engine::run` parses the whole file, reports every syntax error at once if
there are any, and otherwise executes the statements in order. If the
program defines `main` and never calls it, `main` runs afterwards (see
`run_main` in `lib.rs`). Errors come back as `chimera_lang::Error`, with a
pre-rendered, colourable report.

## 1. Lexing (`lexer.rs`)

The lexer produces a flat `Vec<Token>`. Every token records its span, line,
column, the indentation of its line, whether it starts a line, and whether
whitespace preceded it. The parser relies heavily on the last two: `f (x)`
and `f(x)` can mean different things, and so can `a -b` and `a - b`.

The lexer **never reserves words**. `end`, `select`, `to`, `times` and
`model` are all plain identifiers, and the parser decides what they mean.

The lexer *does* decide character-level ambiguities, because only it can see
raw text:

- `//`: floor division or comment (`at_floor_div`)
- `#`: length operator, `#define`, or comment (`at_length_op`, `at_define`)
- `--`: decrement or comment (`is_dash_comment`)
- `'`: Rust lifetime, Haskell prime (`foldl'`), or character/string literal
  (`at_lifetime`)
- `` ` ``: Haskell infix or JavaScript template string (`at_backtick_infix`)
- `(*`: OCaml comment or a Haskell section `(* 3)` (`at_ocaml_comment`)
- `%w[…]`: Ruby word array or modulo (`at_word_array`)
- `?`/`!` after a method name: part of a Ruby method name (`xs.empty?`) or
  an operator

Strings are lexed into `StrLit { parts }`, where each part is text or an
interpolated expression's source range. One lexer handles Python f-strings,
JS templates, Ruby `#{}`, C# `$"{}"`, Kotlin/Swift `"${}"`/`"\()"` and
soft `$name` interpolation. The parser later parses each interpolated range
with `parse_expr_range`.

## 2. Parsing (`parser/`)

The parser is a hand-written recursive-descent parser with a Pratt core for
expressions. Three ideas keep "any syntax" manageable:

**Interchangeable block styles.** After any header (`if c`, `def f()`,
`while x`, `class A`), `Parser::body` accepts `{ … }`, `:` with an indented
block, `do`/`then`/`begin` … `end`, `=> expr`, or a newline followed by
statements up to an end keyword (`end`, `fi`, `done`, `endif`, …). Python's
offside rule is enforced by comparing token **columns**, not with INDENT
tokens, so indented blocks nest correctly inside braces and vice versa.
Terminator classes (`terms::END`, `ELSE`, `CATCH`, `CASE`, `WHEN`, `UNTIL`)
tell a body which words end it.

**Context flags.** Some meanings depend on where you are rather than on the
next few tokens. `Parser` carries flags such as:

| Flag | Meaning |
|---|---|
| `no_brace` | in a statement header, `{` opens the body, not a map |
| `eq_is_cmp` | in an unparenthesised condition, `=` compares |
| `rust_ranges` | inside a Rust `fn`, `a..b` excludes `b` |
| `kotlin_fun` | inside a Kotlin `fun`, `print` adds no newline |
| `no_juxtapose` | in a SQL select list, `f x` is not a call |
| `no_bitor` | in Haskell guards, `\|` is not bitwise or |
| `in_index` | in `[…]`, `a:b` is a slice |

**Backtracking.** `Parser::attempt(|p| …)` snapshots the token position and
error count, runs a speculative parse, and rewinds if it fails. This
resolves the hard collisions:

- `(a, b) => a + b` (a lambda) versus `(a, b)` (a tuple)
- `int x = 5` (a C declaration) versus `print x` (a command call)
- `Point { x: 1 }` (a struct literal) versus `if p { … }` (a block)
- `f(x) = x^2` (a math definition) versus `f(x) == …` (a comparison)

`statement_inner` (`stmt.rs`) dispatches on the first word: declarations in
every language's spelling (`def`, `function`, `fn`, `func`, `fun`, `proc`,
`sub`), control flow, classes, structs, enums, traits, `impl` blocks, SQL
statements, `model`, `train` and English sentences. Anything else becomes an
expression statement, a command call (`puts x`), or an assignment.

`expr.rs` holds the Pratt loop (`expr_bp`). `binary_op` maps tokens to
operators with binding powers from the `bp` module, `prefix`/`primary` parse
atoms and prefix forms, and `postfix_ops` handles calls, indexing, member
access, trailing blocks (`xs.map { … }`, `do |x| … end`), Rust `?`, Swift
`!`, struct literals and more.

The parser **recovers from errors**: it records a `Diagnostic` and skips to
the next statement boundary, so one run reports all syntax errors.

## 3. The AST (`ast.rs`)

Every dialect lowers to one tree. Statements (`StmtKind`) are `Let`,
`Assign`, `Fn`, `Class`, `Enum`, `If`, `While`, `For`, `Match`, `Try`,
`Return`, `Model`, `Train`, `Query` and so on. Expressions (`ExprKind`) cover
literals, names, calls, lambdas, comprehensions, ranges, `if`/`match` as
expressions, `Yield`, `Jump` (`return`/`break`/`raise` usable as
expressions), `Let` conditions (`if let`) and more.

Much of the "polyglot" work happens by **desugaring into this shared
tree**:

- Haskell sections `(+ 1)` become calls to `__section_left`/`__section_right`.
- Go's `v, ok := m[k]` becomes `__comma_ok`.
- Lua's numeric `for i = 1, n` becomes an inclusive range.
- Kotlin's `listOf`, Rust's `vec!` and C#'s collection initialisers become
  list literals or prelude calls.

Because everything lands in one tree, `explain` and `convert` work on every
dialect for free.

## 4. The interpreter (`interp/`)

A tree-walking evaluator. The main pieces are:

**Values** (`value.rs`). `Value` is a cheap-to-clone enum: `Nil`, `Bool`,
`Int(i64)`, `Big(Rc<BigInt>)`, `Float`, `Str(Rc<str>)`, `List`/`Map`/`Set`
(shared and mutable through `Rc<RefCell<…>>`), `Tuple`, `Range`, `Func`
(closures), `Native` (builtins), `Bound` methods, `Class`/`Instance`,
`Enum`/`Variant`, `Module`, `Tensor`, `Dual`, and `Object(Rc<dyn Object>)`
for native objects such as models, optimisers and the AI client. Integers
promote to `BigInt` (`bigint.rs`, schoolbook and Karatsuba multiplication)
on overflow.

**Names** (`symbol.rs`). Every identifier is interned into a 4-byte `Sym`,
so lookups compare integers.

**Environments.** `Env` is a chain of `Rc` scopes (`Global`, `Function`,
`Block`). Small scopes are a vector, and large ones (the global scope) add a
hash map. Closures capture their defining `Env`.

**Control flow.** Evaluation returns `R<Value> = Result<Value, Flow>`, where
`Flow` is `Return(v)`, `Break`, `Continue` or `Throw(exception)`. Rust's `?`
operator does all the unwinding: `return` deep inside a loop inside a
`match` is just an `Err(Flow::Return(v))` propagating up to the function
call. Exceptions carry a stack of `Frame`s for tracebacks.

**Generators** (`generator.rs`). A recursive tree-walker can't pause in the
middle of a Rust stack. So a function containing `yield` is compiled into a
small `GStmt` tree and run by a *stepper* that keeps its position in an
explicit stack of frames (`Block`, `While`, `For`, `CFor`, `Loop`, `Repeat`,
`Delegate`). Statements without a `yield` run atomically on the normal
interpreter. The same mechanism powers Python generators, JavaScript
`function*`, `yield*`/`yield from` delegation and `send()`. A Ruby method
that `yield`s to a block runs eagerly with the block bound instead.

**Classes** (`class.rs`). Classes, structs, records, data classes and
`impl` blocks all build a `Class` with fields, methods, static methods and
optional superclasses and traits. Attribute lookup checks instance fields,
then the class chain, then built-in methods, and suggests a near match
("did you mean …") when nothing is found. Inside methods, fields can be used
without `self.`/`this.` (`implicit_member`).

**Patterns** (`pattern.rs`). One matcher serves Python `match`, Rust
`match`, Haskell function heads, Scala/Kotlin `when`, destructuring
assignment and `catch` clauses. It handles literals, bindings, wildcards,
tuples and lists (with `*rest`/`..rest`), maps, enum variants, struct
patterns, ranges, type tests, guards and alternatives.

**Operators** (`ops.rs`) implement the cross-language semantics documented
in [LANGUAGE.md](LANGUAGE.md). They dispatch tensors and dual numbers to the
ML module, and call operator overloads (`__add__`, `operator+`, `plus`) on
instances.

**Formatting** (`format.rs`) implements `print`/`repr`, Python format specs,
printf verbs, Rust `{:?}`/`{:>8.2}` and C# composite formatting.

## 5. Builtins (`builtins/`)

Builtins register with every name they go by, so `len`, `length`, `count`,
`size` and `strlen` are one function. The pieces are:

- `core.rs`: the bulk of global functions (`print`, `range`, `map`, `zip`,
  `sorted`, `input`, conversions, …).
- `methods.rs`: methods on built-in types. `"abc".upper()`, `xs.push(…)`,
  `xs.each { … }`, `s.len()` and `xs.iter().map(…).collect()` all go through
  one dispatcher keyed on receiver type and method name.
- `objects.rs`: namespace objects like `console`, `Math`, `System`, `fmt`,
  `std`, `IO`, `Arrays`, `Printf` and `string`.
- `modules.rs`: importable modules (`math`, `random`, `json`, `time`, `os`,
  `sys`, `string`, `statistics`, `itertools`, `functools`, `collections`,
  `operator`, Go's `strings`/`strconv`, …). `import` also loads `.chi` files
  next to the program.
- `prelude.chi`: library code **written in Chimera**, parsed once at engine
  start-up: `Result` and its methods, Kotlin collection builders, Haskell
  helpers, `StringBuilder` and more.

## 6. Machine learning (`ml/`)

- `tensor.rs`: `Tensor` is an `Rc`'d n-d `f64` array plus an optional
  autograd node (the producing op and its inputs). Elementwise ops broadcast
  with NumPy rules. `backward()` walks the graph in reverse topological
  order, iteratively, so deep graphs don't overflow the stack. All indexing
  is expressed as a gather of flat indices, whose backward is a
  scatter-add, so every kind of slice is differentiable.
- `dual.rs`: dual numbers for forward mode. Each differentiation gets a
  fresh tag, which avoids perturbation confusion, so `f″` is exact nested
  forward mode.
- `autodiff.rs`: `f′`, `derivative(f, n)`, `∇f` and `grad(f)`.
  Reverse-mode gradients of plain functions run them on tensors.
- `nn.rs`: layers (dense, dropout, flatten, activations), losses, the
  optimisers (SGD with momentum, RMSProp, Adam), `model` declarations, the
  `train` statement, and API shapes for PyTorch (`nn.Linear`, `optim.Adam`,
  `loss.backward()`), Keras (`Sequential`, `compile`, `fit`, `predict`) and
  scikit-learn (`LinearRegression`, `KMeans`, …).

## 7. AI (`ai/`)

- `text.rs`: deterministic, offline NLP. It has tokenisation, embeddings
  built by feature-hashing words and character n-grams (FNV-1a from
  `chimera-core`), cosine similarity, lexicon sentiment with negation and
  intensifiers, keyword extraction, extractive summaries, few-shot and topic
  classification, entity extraction and Markov text generation.
- `offline.rs`: the offline assistant behind `ai "…"` when no key is set. It
  recognises arithmetic, summarisation, sentiment, translation-style
  transforms, haiku, jokes and more.
- `claude.rs`: the Claude backend. It calls the Messages API through a
  `curl` subprocess (Chimera has no dependencies, so it has no TLS stack).
  The API key goes to curl on stdin, never on the command line. The request
  body goes in a temporary file. Refusals and API errors become Chimera
  exceptions.
- `nl.rs`: `please`. A pattern table maps English instructions to actions
  on variables. Steps are split on "and then", pronouns resolve to the last
  object mentioned, and "N times" repeats a step. With the Claude backend,
  unknown instructions are sent to the model, which writes Chimera code, and
  the code runs in the current scope.
- `mod.rs`: the `ai` object (configuration, history, caching, `mock`) and
  the `anthropic.Anthropic().messages.create(…)` SDK shape.

## 8. Queries (`parser/query.rs`, `interp/query.rs`)

SQL and LINQ parse into a `Query` node: projections, sources with aliases,
joins, filters, grouping, `HAVING`, ordering, limits, and the mutations
`INSERT`/`UPDATE`/`DELETE`. The executor works on lists of maps or objects.
Rows are bound to their alias (or to their bare column names) in a child
environment, and ordinary expressions are evaluated against them, so any
Chimera expression works inside `WHERE`.

## 9. The transpiler (`convert.rs`)

`convert(stmts, src, target)` walks the AST and emits Python, JavaScript,
Ruby or Lua. Two notable pieces are:

- a precedence table, so the output carries exactly the parentheses it
  needs;
- per-target lowering of constructs that don't exist everywhere: ranges,
  comprehensions (to `map`/`filter` chains in JS, Ruby and Lua),
  `continue` (a `goto continue` label in Lua), classes (metatables in Lua,
  `attr_accessor` in Ruby), string interpolation and `match`.

Anything a target can't express is emitted as
`chimera("what", "original source")`, so the output always parses and the
gaps are easy to find. `detect_dialect` uses scored heuristics to label each
statement for `explain`.

## 10. Testing

| Test | What it checks |
|---|---|
| unit tests (`cargo test --lib`) | lexer rules, bigint arithmetic, tensors, text tools, the transpiler |
| `tests/golden.rs` | every `tests/golden/*.chi` program prints exactly its `.out` file (offline AI) |
| `tests/examples.rs` | every program in `examples/` runs without error |
| `tests/parse_smoke.rs` | one representative program per dialect parses cleanly |
| `tests/convert_roundtrip.rs` | the Python and JavaScript translations, run by real `python3`/`node`, print what Chimera prints |
| `chimera` CLI tests | REPL multi-line entry detection |

## Design constraints

- **Zero dependencies.** Everything from bigints to tensors to JSON is
  written here. It keeps the build instant and the code readable end to end.
- **No `unsafe`.** `Rc`/`RefCell` throughout. The interpreter is
  single-threaded by design. The CLI runs it on a thread with a 1 GiB stack,
  so deeply recursive programs don't overflow.
- **Determinism.** The offline AI, the RNG (when seeded) and hash-based
  embeddings give the same output on every machine. The golden tests depend
  on this.
