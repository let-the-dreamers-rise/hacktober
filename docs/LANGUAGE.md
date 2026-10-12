# The Chimera language

Chimera has one set of values and one runtime, but it accepts many surface
syntaxes. This document covers the shared semantics, then the rules Chimera
uses when two languages read the same characters differently.

If you're unsure how a line was understood, run `chimera explain file.chi`.
It prints the dialect, the kind of statement, and a canonical (Python-like)
rendering of every statement.

## Values

| Value | Written as | Notes |
|---|---|---|
| nothing | `nil`, `None`, `null`, `undefined`, `nullptr` | one value with many names |
| booleans | `true`/`false`, `True`/`False` | |
| integers | `42`, `0xff`, `0b1010`, `1_000_000` | 64-bit, **promoted to arbitrary precision on overflow** |
| floats | `3.14`, `1e-9` | IEEE-754 double |
| strings | `"…"`, `'…'`, `` `…` ``, `"""…"""`, `r"…"`, `f"…"` | immutable, Unicode |
| lists | `[1, 2, 3]`, `vec![…]`, `listOf(…)`, `{1, 2}` in a typed C/Java array or a Lua `local` | mutable, 0-indexed, negative indices count from the end |
| tuples | `(1, "a")`, `Pair(a, b)` | immutable |
| maps | `{"a": 1}`, `{a: 1}`, `{a = 1}`, `mapOf("a" to 1)` | insertion-ordered |
| sets | `{1, 2}`, `set([…])`, `setOf(…)` | |
| ranges | `1..5`, `0..<5`, `range(5)`, `1 to 5` | lazy; print as `range(start, end)` |
| functions | `def`, `function`, `fn`, `func`, `fun`, `x => …`, `\|x\| …`, `\x -> …`, `lambda x: …`, `{ it * 2 }` | closures |
| objects | classes, structs, records, data classes, enum variants | |
| tensors | `tensor [[1, 2], [3, 4]]`, `torch.tensor(…)`, `np.array(…)` | n-dimensional floats with autograd |
| errors | `Exception("…")`, `ValueError(…)`, `Err(e)` | thrown, caught, or matched |

**Truthiness:** `nil`, `false`, `0`, `0.0`, `""` and empty collections are
false. Everything else is true. (Ruby treats `0` as true; Chimera follows
Python, JavaScript and C here.)

**Equality:** `==` compares by value, and `3 == 3.0` is true. `"3" == 3` is
false. JavaScript's `===` behaves like `==`, because Chimera never coerces
types for comparison.

**Integers never overflow.** `2 ** 200` and `factorial(50)` are exact.
`9223372036854775807 + 1` is `9223372036854775808`.

## Arithmetic

| Expression | Result | Why |
|---|---|---|
| `7 / 2` | `3.5` | true division, as in Python 3, JavaScript and Ruby with floats |
| `7 / 2` inside a function with C, Java, Go or Rust integer types | `3` | typed integer functions keep integer division |
| `7 // 2` | `3` | floor division (Python, Lua) |
| `-7 // 2` | `-4` | floors toward negative infinity |
| `-7 % 3` | `2` | the result takes the sign of the divisor, as in Python |
| `2 ^ 3`, `2 ** 3` | `8` | `^` is exponentiation, as in maths, Lua and R (not XOR) |
| `"a" + 1` | `"a1"` | strings concatenate with anything |
| `"a" * 3` | `"aaa"` | |
| `[1, 2] + [3]`, `[1, 2] * 2` | `[1, 2, 3]`, `[1, 2, 1, 2]` | |
| `"lua" .. "!"` | `"lua!"` | Lua concatenation (spaced `..`, see below) |
| `1 / 0`, `1 / 0.0` | `ZeroDivisionError` | Chimera raises instead of producing `Infinity` |
| `x++`, `x--`, `++x` | increment/decrement | C family |

Math-style definitions allow implicit multiplication: `f(x) = 3x^2 + 2x`.

## Disambiguation rules

Every rule below exists because two languages assign different meanings to
the same characters. Each one was chosen to keep real programs in every
dialect working. If you find a program that breaks a rule, please open an
issue.

### `//`: comment or floor division?

`//` is **floor division** when all of the following hold:

- the previous token on the line is an operand (a number, a name, `)` or `]`);
- it is not preceded by two or more spaces or a tab (aligned trailing
  comments like `x = f(y)   // note` stay comments);
- the rest of the line starts like an expression (a digit, a name, `(` or `-`);
- the rest of the line (up to any later ` //` comment) doesn't read as prose.
  Two consecutive plain words (`// set x to 5`), or quotes, colons, `!` or
  `?`, mean a comment.

Otherwise `//` starts a comment.

```python
half = n // 2               # floor division
x = 10 // 3   // floor division, then a C-style comment
y = compute(x) // the answer to everything
```

### `#`: comment, length, directive or attribute?

- `#` followed by a name, `(`, `{` or a quote, **after an operator or after
  `return`/`print`/`and`/`or`/`not`/`in`/`then`/`do` on the same line**, is
  Lua's length operator: `print(#t)`, `n = #xs + 1`.
- `#define NAME value` defines a constant, and `#define SQ(x) ((x) * (x))`
  defines a function. Other preprocessor lines (`#include`, `#pragma`,
  `#ifdef`) are skipped.
- `#[derive(…)]` and `#![…]` are Rust attributes, accepted and ignored.
- `#{…}` inside a double-quoted string is Ruby interpolation.
- Everything else starts a comment.

### `--`: comment or decrement?

`--` starts a comment (Lua, SQL, Haskell) when it begins a line or follows
whitespace **and** is followed by whitespace, the end of the line, or another
`-`. `x--` and `--x` are decrements.

### `..`, `...`, `..<` and `..=`: which ranges include their end?

| Syntax | Meaning |
|---|---|
| `a..b` | **inclusive** (Ruby, Kotlin, Haskell `[a..b]`, Perl) |
| `a..b` inside a Rust `fn` body or a brace-style `for x in a..b {` | **exclusive** (Rust) |
| `a..=b`, `a...b` | inclusive (Rust, Swift) |
| `a..<b`, `a until b` | exclusive (Swift, Kotlin) |
| `a .. b` with spaces around `..` and non-numeric operands | Lua string concatenation |
| `a to b`, `a downTo b`, `a..b step 2` | Kotlin/Scala ranges |

### Backticks: template string or infix function?

`` a `div` b ``, a backtick-quoted name directly between two operands, is
Haskell's infix application. Any other backtick starts a JavaScript template
string with `${…}` interpolation.

### `$`: Haskell application, PHP variables, Go/Perl interpolation

- `f $ g x` is Haskell's low-precedence application: `f(g(x))`.
- `$name` as a variable is PHP/Perl/shell syntax for `name`.
- `"$name"` and `"${expr}"` inside strings interpolate (Kotlin, Groovy,
  Dart). This is **soft** interpolation: if `name` isn't defined, the text
  stays literally `$name`, so prices like `"$5"` and regexes are safe.

### `{`: block, map, set, struct literal or lambda?

- **After a statement header** (`if x > 0 {`, `while …{`, `fn f() {`), `{`
  opens a block.
- **After a capitalised name** in an expression (`Point { x: 1, y: 2 }`), it
  is a struct literal (Rust, Go, Swift).
- **After a call or method name** (`xs.map { |x| … }`, `xs.each do |x| … end`,
  `xs.filter { it > 2 }`), it is a trailing block or lambda (Ruby, Kotlin,
  Swift, Groovy). Kotlin's implicit `it` and `x -> …` parameters both work.
- **In expression position**, `{k: v}` / `{"k": v}` / `{k = v}` is a map,
  `{a, b}` is a set, and `{}` is an empty map.
- In a typed C/Java/C# declaration (`int xs[] = {1, 2}`) or a Lua `local`,
  `{1, 2}` is a list.

### `=` in conditions

Inside an `if`/`while` header written without parentheses, VB, Pascal,
SQL and English-style `=` means comparison: `if x = 3 then … end`. Inside
parentheses, and everywhere else, `=` is assignment.

### Function application by juxtaposition

`f x y` (Haskell/OCaml/F#) is a call when `f` is a name and the next token
starts an operand on the same line. Command calls such as `puts "hi"`,
`print x` and `say greeting` use the same rule. Inside SQL select lists
juxtaposition is disabled, so `SELECT name n` still treats `n` as an alias.

### `print`

`print`, `puts`, `println`, `console.log`, `System.out.println`, `fmt.Println`,
`say` and `echo` all print with a trailing newline. `print` inside a Kotlin
`fun` doesn't add a newline (Kotlin's `println` does). Other no-newline forms
are `print_string`, `io.write`, `System.out.print`, `Console.Write`,
`fmt.Print` and `printf`.

### `main`

If a program defines `main` (a function, or a `static main` method on a
class, as in Java and C#) and never calls it, Chimera calls it after the rest
of the file has run. Command-line arguments are passed when `main` takes
parameters. `--no-main` turns this off.

### Results and options

`Some(x)` is just `x`, and `None` is `nil`. `Ok(v)` and `Err(e)` are
variants of a prelude `Result` enum. In `match` patterns, `Ok(v)` also
matches any plain non-error value and `Err(e)` matches an exception value, so
Rust-style matching works on functions that simply return or throw. The `?`
operator propagates errors.

### Lua tables and indexing

Lists are **0-indexed in every dialect**. `t[1]` is the second element even
in Lua code, because one value can't be 0-indexed for Python and 1-indexed
for Lua at the same time. `ipairs(t)` yields 1-based positions, so idiomatic
`for i, v in ipairs(t)` loops print what Lua programmers expect.

### Implicit `this`

Inside methods, fields can be read and assigned without `self.`/`this.`/`@`
(Java, C#, C++, Kotlin style) when no local variable shadows them.

## Machine-learning syntax

```text
f(x) = <expr>                 math-style definition
f′(x), f″(x)                  derivatives (forward-mode dual numbers)
∇f(a, b)                      gradient (reverse-mode autodiff); also grad(f)(a, b)

model Name:                   layer lines: dense IN -> OUT[, activation]
    dense 2 -> 8, relu        also: dropout P, flatten, and bare activations (relu, tanh, …)
    dense 8 -> 1, sigmoid

train NET on X, Y [for N epochs] [using OPT(lr=…)] [with loss L | minimizing L]
                  [in batches of B] [reporting every K | silently]
```

`train` is an expression. It returns the list of per-epoch losses.
Optimisers are `sgd`, `momentum`, `rmsprop` and `adam` (`adamw` is an alias).
Losses are `mse`, `mae`, `bce`, `bce_with_logits`, `cross_entropy` and `nll`,
and Keras/PyTorch spellings such as `MSELoss` and
`categorical_crossentropy` work too.

## AI syntax

```text
ai "prompt"                   ask the configured backend, returns a string
ai.ask(prompt, system=…)      the same, with options
ai.chat(message)              multi-turn conversation (ai.reset() clears it)
ai.code("description")        generate Chimera code (Claude backend; returns source text)
ai.mock("reply", …)           script replies for tests
please <instruction>          perform an English instruction
sentiment(text)               .label, .score
classify(text, labels)        best label (offline: topic vocabularies + similarity)
summarize(text, n)            extractive summary
keywords(text, n)             TF-weighted keywords
embed(text)                   feature-hashed embedding vector
similarity(a, b)              cosine similarity of embeddings
```

The backend is chosen by `CHIMERA_AI` (`offline`, `claude`, `mock`), by
`--ai` on the command line, or automatically: Claude when
`ANTHROPIC_API_KEY` is set, otherwise offline. `anthropic.Anthropic()
.messages.create(model=…, max_tokens=…, messages=[…])` works too, so Python
SDK snippets run unchanged.

## Query syntax

```sql
SELECT [DISTINCT] cols | *  FROM source [alias] [JOIN other alias ON cond]
       [WHERE cond] [GROUP BY keys] [HAVING cond] [ORDER BY key [ASC|DESC], …]
       [LIMIT n] [OFFSET n]
INSERT INTO list VALUES {record}, …
UPDATE list SET field = expr, … [WHERE cond]
DELETE FROM list [WHERE cond]
```

Sources are any lists of maps or objects in scope. Aggregates are `COUNT`,
`SUM`, `AVG`, `MIN` and `MAX`. Conditions accept `AND`/`OR`/`NOT`, `IN`,
`BETWEEN`, `LIKE` and `IS [NOT] NULL`. Keywords may be upper- or lowercase.
C# LINQ (`from x in xs where … orderby … select …`) is also an expression.

## English

Statements that read as English sentences are parsed as such:

```text
set x to 5                    let total be price times quantity
add 3 to x                    increase x by 2
if x is greater than 3 then say "big" otherwise say "small"
repeat 3 times: say "hi"      for each item in list: say item
say the square root of 144    say the sum of numbers
```

Comparisons (`is greater than`, `is at least`, `is not`) and arithmetic
words (`plus`, `minus`, `times`, `divided by`) work in any expression.
`please` takes broader instructions (`please sort xs descending`,
`please add 99 to scores and then print scores`, `please keep the even
numbers in scores`). It understands common list, string and arithmetic
tasks offline, and with the Claude backend it falls back to having the model
write code.
