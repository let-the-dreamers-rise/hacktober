# Chimera

**A programming language that has no syntax of its own, so it accepts all of them.**

Write a line of Python, then a JavaScript arrow function, a Rust `match`, a Go
`:=`, a Haskell list comprehension, a SQL query, or a plain English sentence.
Chimera parses each one, works out what it means, and runs the whole file on
a single interpreter with one shared set of values. Machine learning and AI
are part of the grammar too. Derivatives are operators (`f′`, `∇f`), neural
networks are declarations (`model … train … on`), and `ai "…"` is an
expression.

```python
def fizzbuzz(n):                                  # Python
    return "FizzBuzz" if n % 15 == 0 else "Fizz" if n % 3 == 0 else "Buzz" if n % 5 == 0 else str(n)

const words = [1, 2, 3, 4, 5, 15].map(fizzbuzz);  // JavaScript
puts words.join(" ")                              # Ruby

fn collatz(mut n: u64) -> u64 {                   // Rust
    let mut steps = 0;
    while n != 1 {
        n = if n % 2 == 0 { n / 2 } else { 3 * n + 1 };
        steps += 1;
    }
    steps
}

func longest(limit int) (int, int) {              // Go
    best, arg := 0, 0
    for i := 1; i < limit; i++ {
        if s := collatz(i); s > best {
            best, arg = s, i
        }
    }
    return arg, best
}

n, steps = longest(1000)
printf("Longest Collatz chain under 1000: %d (%d steps)\n", n, steps)   // C

squares = [x * x | x <- [1..10], even x]          -- Haskell
local total = 0                                   -- Lua
for _, s in ipairs(squares) do total = total + s end
print("sum of even squares:", total)

data class Planet(val name: String, val moons: Int)       // Kotlin
planets = listOf(Planet("Earth", 1), Planet("Mars", 2), Planet("Jupiter", 95))
busy = SELECT name FROM planets WHERE moons > 1 ORDER BY moons DESC   -- SQL
print(busy.map(p => p["name"]))

f(x) = x^2 * sin(x)                               # math
print("f′(π) ≈", round(f′(π), 4))

set greeting to "hello"                           # English
please capitalize greeting
say greeting, "from", len(planets), "planets"

big = 2 ** 200                                    # arbitrary precision
print(len(str(big)), "digits")
print(sentiment("Chimera is a wonderful, delightful mess").label)   # AI
```

```console
$ chimera examples/hello.chi
1 2 Fizz 4 Buzz FizzBuzz
Longest Collatz chain under 1000: 871 (178 steps)
sum of even squares: 220
["Jupiter", "Mars"]
f′(π) ≈ -9.8696
Hello from 3 planets
61 digits
positive
```

That is a single file, and none of it is preprocessed or split by language.
Every line goes through the same lexer, parser and interpreter.

---

## Quick start

You need a stable Rust toolchain (Rust 1.88 or newer). The
project has **no dependencies** apart from the standard library.

```bash
git clone https://github.com/let-the-dreamers-rise/hacktober
cd hacktober
cargo run --release -- examples/hello.chi     # run a program
cargo run --release -- repl                   # interactive session
cargo install --path crates/chimera           # put `chimera` on your PATH
```

```console
$ chimera -e 'print([x ** 2 for x in 1..5].filter(n => n % 2 == 1).sum())'
35
```

## What's inside

| | |
|---|---|
| **Polyglot parser** | Python, JavaScript/TypeScript, C, C++, Java, C#, Go, Rust, Ruby, Lua, Haskell, OCaml, Kotlin, Swift, SQL, LINQ and English, all in one grammar with context-sensitive disambiguation |
| **One runtime** | Closures, classes, structs, enums, traits/interfaces, pattern matching, exceptions, generators (resumable, not threaded), Ruby blocks, Go `defer`, arbitrary-precision integers |
| **Machine learning in the syntax** | N-d tensors with broadcasting, reverse-mode autograd, forward-mode dual numbers, `f′`/`f″`/`∇f` operators, `model` declarations, a `train … on … for N epochs using adam` statement, PyTorch/Keras/scikit-learn-style APIs |
| **AI in the syntax** | `ai "prompt"`, `please <English instruction>`, sentiment, classification, summarisation, embeddings, Markov text, all available offline; with `ANTHROPIC_API_KEY` set, `ai` talks to Claude |
| **Queries in the syntax** | `SELECT … FROM … JOIN … GROUP BY … HAVING … ORDER BY … LIMIT` over lists, maps and objects; `INSERT`/`UPDATE`/`DELETE`; C# LINQ `from … where … select` |
| **Tooling** | REPL with multi-line entry, `explain` (shows the dialect and canonical form of every statement), `convert` (translates to Python, JavaScript, Ruby or Lua), rustc-style error messages with "did you mean" hints |
| **Engineering** | About 44k lines of Rust, zero dependencies, `unsafe` forbidden, golden tests, translations checked against real `python3` and `node` |

## A tour

### Every dialect at once

```rust
#[derive(Debug, Clone, PartialEq)]
enum Shape { Circle(f64), Rect { w: f64, h: f64 } }

impl Shape {
    fn area(&self) -> f64 {
        match self {
            Shape::Circle(r) => 3.14159 * r * r,
            Shape::Rect { w, h } => w * h,
        }
    }
}
```

```ruby
shapes = [Shape::Circle(1.0), Shape::Rect { w: 2.0, h: 3.0 }]
shapes.each_with_index do |s, i|
  puts "#{i}: #{s.area().round(2)}"
end
```

```haskell
quicksort []     = []
quicksort (p:xs) = quicksort [x | x <- xs, x < p] ++ [p] ++ quicksort [x | x <- xs, x >= p]
main = print $ quicksort [3, 1, 4, 1, 5, 9, 2, 6]
```

Definitions from one dialect can be called from any other. A Rust enum can
be matched in a Python `match`, a Go function can return into a Lua loop, and
a Haskell function can be passed to JavaScript's `.map`.

### Generators and blocks

```javascript
function* naturals() { let n = 0; while (true) yield n++; }
```

```python
def primes():
    for n in naturals():
        if n > 1 and all(n % d for d in range(2, isqrt(n) + 1)):
            yield n
print(take(10, primes()))      # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```

Generators run on an explicit, resumable stepper. They are not threads or
callbacks, so infinite generators, `yield*`/`yield from` delegation and Ruby
`yield` to a block all share one mechanism.

### Calculus as syntax

```python
f(x) = x^3 - 2x^2 + x
print(f′(2), f″(2))                         # 5 8

rosenbrock(a, b) = (1 - a)^2 + 100 * (b - a^2)^2
print(∇rosenbrock(0, 0))                    # [-2.0, 0.0]

w = tensor([1.0, -2.0, 3.0], requires_grad=True)
(w * w).sum().backward()
print(w.grad.tolist())                      # [2.0, -4.0, 6.0]
```

`f′` uses forward-mode dual numbers, `∇f` uses reverse-mode autodiff, and
tensors record a computation graph the way PyTorch does. Implicit
multiplication (`2x^2`) and `^` for powers work in math-style definitions.

### Neural networks as declarations

```python
seed(7)
X = tensor [[0, 0], [0, 1], [1, 0], [1, 1]]
y = tensor [[0], [1], [1], [0]]

model XorNet:
    dense 2 -> 16, tanh
    dense 16 -> 1, sigmoid

net = XorNet()
history = train net on X, y for 1500 epochs using adam(lr=0.03) with loss bce reporting every 500
```

```console
epoch   500  loss 0.000236
epoch  1000  loss 0.000069
epoch  1500  loss 0.000032
[0.0, 1.0] -> 1.000 (want 1.0)
```

Keras-style `Sequential([...]).compile(...).fit(...)`, PyTorch-style
`nn.Linear` with `optim.Adam`, and scikit-learn's
`LinearRegression().fit(X, y)` work as well.

### SQL over your own data

```sql
by_dept = SELECT dept, COUNT(*) AS n, AVG(salary) AS avg_pay
          FROM employees GROUP BY dept HAVING COUNT(*) > 1 ORDER BY avg_pay DESC
console.table(by_dept)
```

```console
┌──────┬───┬─────────┐
│ dept │ n │ avg_pay │
├──────┼───┼─────────┤
│ eng  │ 3 │ 135.0   │
│ ops  │ 2 │ 98.0    │
└──────┴───┴─────────┘
```

Queries are ordinary expressions. They read plain lists of maps or objects,
support joins, `BETWEEN`, `LIKE` and aggregates, and `INSERT`/`UPDATE`/`DELETE`
change the underlying list in place.

### AI as syntax

```python
answer = ai "what is 17 * 23"                # 391
print(sentiment("the build is broken").label) # negative
print(classify("the striker scored twice", ["sports", "politics", "cooking"]))  # sports

todo = ["write docs", "fix bug", "ship it"]
please reverse todo and then print todo
please add "celebrate" to todo
```

Without an API key, a deterministic offline assistant answers what it can:
arithmetic, summaries, sentiment, keywords, haiku, jokes and text
transformations. When `ANTHROPIC_API_KEY` is set, `ai` calls Claude through
the Messages API, and `please` asks the model to write Chimera code for
instructions it doesn't recognise, then runs that code. Programs that use AI
stay testable, because `ai.mock(...)` queues scripted replies.

| Variable | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` | enables the Claude backend |
| `CHIMERA_AI` | force a backend: `offline`, `claude` or `mock` |
| `CHIMERA_AI_MODEL` | override the Claude model |

### English

```text
set price to 20
set quantity to 3
let total be price times quantity
if total is greater than 50 then say "big order:", total
repeat 3 times: say "thanks!"
```

## The command line

```text
chimera [OPTIONS] <file.chi> [ARGS...]    run a program
chimera -e <code> [ARGS...]               run code given on the command line
chimera repl                              start an interactive session
chimera check <file>...                   report syntax errors without running
chimera explain <file>                    show how each statement was understood
chimera convert <file> --to <language>    translate to python, js, ruby or lua
chimera tokens <file>                     print the token stream
chimera ast <file>                        print the syntax tree
```

`explain` is the best way to see what the parser is doing:

```console
$ chimera explain examples/hello.chi
── line 9 · Rust · function definition
   │ fn collatz(mut n: u64) -> u64 {
   │     ...
   ⇒ canonical form:
     def collatz(n):
         steps = 0
         while n != 1:
             n = n / 2 if n % 2 == 0 else 3 * n + 1
             steps += 1
         return steps
```

`convert` turns a Chimera program, in whatever mix of syntaxes, into one
real language. The test suite runs the Python and JavaScript output through
`python3` and `node` and checks that they print the same thing Chimera does.

Inside the REPL, `:help` lists the commands. Among them are `:explain`, `:ast`,
`:py`/`:js`/`:rb`/`:lua <code>` (translate a snippet), `:vars`, `:load` and
`:ai`.

## How it works

```text
source ──► lexer ──► parser ──► AST ──► interpreter ──► output
           (context-   (Pratt + backtracking,           (tree-walking, Rc<Env>
           sensitive   dialect flags, one AST           scopes, generators as
           tokens)     for every language)              resumable steppers)
```

The hard part is ambiguity. Is `//` a comment or floor division? Is `#` a
comment or Lua's length operator? Is `a..b` inclusive (Ruby, Kotlin) or
exclusive (Rust)? Is `{` a block, a map or a struct literal? The lexer and
parser answer these from local context: what came before, what comes after,
and which dialect's construct the current function is written in. Each rule
is documented in [docs/LANGUAGE.md](docs/LANGUAGE.md). The code tour is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```text
crates/
├── chimera-core/   deterministic RNG, checksums, binary codec, diagnostics, tables
├── chimera-lang/   lexer, parser, interpreter, builtins, ML, AI, queries, transpiler
└── chimera/        the `chimera` command-line tool
examples/           runnable programs (all run by the test suite)
```

## Hacktoberfest 🎃

This repo welcomes Hacktoberfest contributions. A language that accepts
every syntax always has another syntax missing, so there's plenty to do:

- **Teach Chimera a construct it doesn't know yet.** Elixir pipes, PHP `$vars`,
  Perl, Zig, Julia, R, APL, COBOL. If it parses unambiguously, it can probably go in.
- **Add a builtin or prelude function.** Many can be written in Chimera itself
  in `crates/chimera-lang/src/builtins/prelude.chi`.
- **Add a transpiler target.** Go, Rust, C, Kotlin.
- **Write a golden test or an example program.**
- **Find a program that should run and doesn't.** That's a bug report, and
  bug reports are very welcome.

Start with [CONTRIBUTING.md](CONTRIBUTING.md). It explains how to add a
dialect feature in about 20 lines and how golden tests work, and it lists
issue ideas for every level.

## Status

Chimera is an experiment, and it's honest about it:

- It is a tree-walking interpreter, so it's slow compared with CPython or V8,
  and a lot slower than compiled Rust or Go.
- "Any syntax" means a large and growing subset of each language, not full
  compliance. Some constructs genuinely conflict. In those cases Chimera picks
  the most common reading and documents it.
- There is no static type checking. Type annotations are parsed and used for
  defaults and conversions, not enforced.

See [ROADMAP.md](ROADMAP.md) for what's next.

## License

[MIT](LICENSE)
