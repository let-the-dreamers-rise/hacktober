//! Command-line parsing and the subcommands.

use std::io::{BufRead, IsTerminal, Read, Write};

use chimera_lang::ast::{ExprKind, StmtKind};
use chimera_lang::convert::{self, Target};
use chimera_lang::{Engine, Value, ai};

const USAGE: &str = "\
chimera — the shapeshifting programming language

USAGE:
    chimera [OPTIONS] <file.chi> [ARGS...]    run a program
    chimera run <file> [ARGS...]              run a program
    chimera -e <code> [ARGS...]               run code given on the command line
    chimera repl                              start an interactive session
    chimera check <file>...                   report syntax errors without running
    chimera explain <file>                    show how each statement was understood
    chimera convert <file> --to <language>    translate to python, js, ruby or lua
    chimera tokens <file>                     print the token stream
    chimera ast <file>                        print the syntax tree
    chimera version                           print the version

OPTIONS:
    --offline           use the deterministic offline AI backend
    --ai <backend>      choose the AI backend: offline, claude or mock
    --model <id>        Claude model for the claude backend
    --no-main           don't run main() automatically
    --color / --no-color    force colored or plain error messages
    -o <file>           write `convert` output to a file

Programs may mix Python, JavaScript, C, Java, Go, Rust, Ruby, Lua, Haskell,
OCaml, Kotlin, Swift, SQL and plain English. Run `chimera repl` and type
`:help` to explore.";

#[derive(Debug, Default)]
struct Options {
    offline: bool,
    ai: Option<String>,
    model: Option<String>,
    no_main: bool,
    color: Option<bool>,
    output: Option<String>,
    to: Option<String>,
}

/// Runs the CLI with `args` (without the program name); returns the exit
/// code.
pub fn main(args: Vec<String>) -> u8 {
    let mut opts = Options::default();
    let mut rest: Vec<String> = Vec::new();
    let mut it = args.into_iter();
    // Options come before the program; everything after a program file
    // belongs to the program.
    while let Some(a) = it.next() {
        let tool = rest.first().is_some_and(|c| {
            matches!(
                c.as_str(),
                "convert" | "check" | "explain" | "tokens" | "ast"
            )
        });
        if !rest.is_empty() && !tool {
            rest.push(a);
            continue;
        }
        match a.as_str() {
            "--offline" => opts.offline = true,
            "--ai" => opts.ai = it.next(),
            "--model" => opts.model = it.next(),
            "--no-main" => opts.no_main = true,
            "--color" => opts.color = Some(true),
            "--no-color" => opts.color = Some(false),
            "-o" | "--output" => opts.output = it.next(),
            "--to" | "-t" => opts.to = it.next(),
            "-h" | "--help" | "help" if rest.is_empty() => {
                println!("{USAGE}");
                return 0;
            }
            "-V" | "--version" | "version" if rest.is_empty() => {
                println!("chimera {}", env!("CARGO_PKG_VERSION"));
                return 0;
            }
            _ => rest.push(a),
        }
    }
    let color = opts
        .color
        .unwrap_or_else(|| std::io::stderr().is_terminal());
    let make_engine = |opts: &Options| -> Engine {
        let mut e = if opts.offline {
            Engine::offline()
        } else {
            Engine::new()
        };
        if let Some(b) = opts.ai.as_deref().and_then(ai::Backend::from_name) {
            e.interp().ai.backend = b;
        }
        if let Some(m) = &opts.model {
            e.interp().ai.model = m.clone();
        }
        e.set_color(color);
        e.set_auto_main(!opts.no_main);
        e
    };

    match rest.first().map(String::as_str) {
        None => {
            if std::io::stdin().is_terminal() {
                return repl(make_engine(&opts));
            }
            let mut src = String::new();
            if std::io::stdin().read_to_string(&mut src).is_err() {
                eprintln!("error: could not read standard input");
                return 1;
            }
            run_source(make_engine(&opts), "<stdin>", &src, vec!["-".into()])
        }
        Some("-e" | "eval") => {
            let Some(code) = rest.get(1).cloned() else {
                eprintln!("error: `-e` needs some code");
                return 2;
            };
            let mut argv = vec!["-e".to_owned()];
            argv.extend(rest[2..].iter().cloned());
            run_source(make_engine(&opts), "<eval>", &code, argv)
        }
        Some("repl") => repl(make_engine(&opts)),
        Some("run") => match rest.get(1) {
            Some(path) => run_file(make_engine(&opts), path, rest[2..].to_vec()),
            None => {
                eprintln!("error: `run` needs a file");
                2
            }
        },
        Some("check") => {
            if rest.len() < 2 {
                eprintln!("error: `check` needs at least one file");
                return 2;
            }
            let mut status = 0;
            for path in &rest[1..] {
                let Some(src) = read(path) else { return 1 };
                let mut engine = make_engine(&opts);
                match engine.parse(path, &src) {
                    Ok(stmts) => println!("{path}: ok ({} statements)", stmts.len()),
                    Err(e) => {
                        eprint!("{}", e.report);
                        status = 1;
                    }
                }
            }
            status
        }
        Some("explain") => match rest.get(1) {
            Some(path) => explain(make_engine(&opts), path),
            None => {
                eprintln!("error: `explain` needs a file");
                2
            }
        },
        Some("convert") => {
            let Some(path) = rest.get(1) else {
                eprintln!("error: `convert` needs a file");
                return 2;
            };
            let lang = opts
                .to
                .clone()
                .or_else(|| rest.get(2).cloned())
                .unwrap_or_else(|| "python".into());
            let Some(target) = Target::from_name(&lang) else {
                eprintln!(
                    "error: unknown language `{lang}` (try python, js, ruby, lua or chimera)"
                );
                return 2;
            };
            let Some(src) = read(path) else { return 1 };
            let mut engine = make_engine(&opts);
            match engine.convert(path, &src, target) {
                Ok(code) => match &opts.output {
                    Some(out) => match std::fs::write(out, code) {
                        Ok(()) => {
                            eprintln!("wrote {out}");
                            0
                        }
                        Err(e) => {
                            eprintln!("error: cannot write {out}: {e}");
                            1
                        }
                    },
                    None => {
                        print!("{code}");
                        0
                    }
                },
                Err(e) => {
                    eprint!("{}", e.report);
                    1
                }
            }
        }
        Some("tokens") => {
            let Some(path) = rest.get(1) else {
                eprintln!("error: `tokens` needs a file");
                return 2;
            };
            let src = std::fs::read_to_string(path).unwrap_or_else(|_| path.clone());
            let (toks, errors) = chimera_lang::lexer::lex(&src);
            for t in toks {
                println!("{:>4}:{:<3} {:?}", t.line + 1, t.col + 1, t.tok);
            }
            if errors.is_empty() { 0 } else { 1 }
        }
        Some("ast") => {
            let Some(path) = rest.get(1) else {
                eprintln!("error: `ast` needs a file");
                return 2;
            };
            let src = std::fs::read_to_string(path).unwrap_or_else(|_| path.clone());
            let mut engine = make_engine(&opts);
            match engine.parse("<ast>", &src) {
                Ok(stmts) => {
                    println!("{stmts:#?}");
                    0
                }
                Err(e) => {
                    eprint!("{}", e.report);
                    1
                }
            }
        }
        Some(path) => run_file(make_engine(&opts), path, rest[1..].to_vec()),
    }
}

fn read(path: &str) -> Option<String> {
    if path == "-" {
        let mut s = String::new();
        return std::io::stdin().read_to_string(&mut s).ok().map(|_| s);
    }
    match std::fs::read_to_string(path) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            None
        }
    }
}

fn run_file(mut engine: Engine, path: &str, args: Vec<String>) -> u8 {
    let Some(src) = read(path) else { return 1 };
    if let Some(dir) = std::path::Path::new(path).parent() {
        engine.set_base_dir(dir);
    }
    let mut argv = vec![path.to_owned()];
    argv.extend(args);
    run_source(engine, path, &src, argv)
}

fn run_source(mut engine: Engine, name: &str, src: &str, argv: Vec<String>) -> u8 {
    engine.set_args(argv);
    match engine.run(name, src) {
        Ok(_) => 0,
        Err(e) => {
            let _ = std::io::stdout().flush();
            eprint!("{}", e.report);
            1
        }
    }
}

/// `chimera explain`: each top-level statement with the dialect it reads
/// as and its canonical (Python-flavoured) form.
fn explain(mut engine: Engine, path: &str) -> u8 {
    let Some(src) = read(path) else { return 1 };
    let stmts = match engine.parse(path, &src) {
        Ok(s) => s,
        Err(e) => {
            eprint!("{}", e.report);
            return 1;
        }
    };
    for s in &stmts {
        let text = src.get(s.span.start..s.span.end).unwrap_or("").trim_end();
        if text.is_empty() {
            continue;
        }
        let line = src[..s.span.start].matches('\n').count() + 1;
        let dialect = convert::detect_dialect(text);
        let kind = describe_stmt(&s.kind);
        println!("── line {line} · {dialect} · {kind}");
        for l in text.lines().take(8) {
            println!("   │ {l}");
        }
        if text.lines().count() > 8 {
            println!("   │ …");
        }
        let canonical = convert::convert(std::slice::from_ref(s), &src, Target::Chimera);
        let body: Vec<&str> = canonical
            .lines()
            .skip_while(|l| l.starts_with('#') || l.is_empty())
            .filter(|l| !l.trim().is_empty())
            .collect();
        if !body.is_empty() && body.join("\n").trim() != text.trim() {
            println!("   ⇒ canonical form:");
            for l in body.iter().take(8) {
                println!("     {l}");
            }
        }
        println!();
    }
    0
}

fn describe_stmt(k: &StmtKind) -> &'static str {
    match k {
        StmtKind::Expr(e) => match &e.kind {
            ExprKind::Assign { .. } | ExprKind::CompoundAssign { .. } | ExprKind::IncDec { .. } => {
                "assignment"
            }
            ExprKind::Call { .. } => "call",
            ExprKind::If { .. } => "conditional",
            ExprKind::Match { .. } => "pattern match",
            ExprKind::Query(_) => "query",
            ExprKind::Please(_) => "natural-language instruction",
            _ => "expression",
        },
        StmtKind::Let { .. } => "declaration",
        StmtKind::Fn(_) => "function definition",
        StmtKind::Class(_) => "class definition",
        StmtKind::Impl { .. } => "methods for a type",
        StmtKind::Enum(_) => "enum definition",
        StmtKind::While { .. }
        | StmtKind::Loop(_)
        | StmtKind::Repeat { .. }
        | StmtKind::For { .. }
        | StmtKind::CFor { .. } => "loop",
        StmtKind::Return(_) => "return",
        StmtKind::Break | StmtKind::Continue => "loop control",
        StmtKind::Throw(_) => "raise",
        StmtKind::Try { .. } => "error handling",
        StmtKind::Switch { .. } => "switch",
        StmtKind::Import(_) => "import",
        StmtKind::Defer(_) => "deferred block",
        StmtKind::Delete(_) => "deletion",
        StmtKind::Model(_) => "neural network definition",
        StmtKind::Train(_) => "training run",
        StmtKind::Pass => "no-op",
    }
}

// ----- REPL ---------------------------------------------------------------------------

const REPL_HELP: &str = "\
Type code in any dialect. Blocks continue until they are complete (after a
line ending in `:`, until a blank line).

  :help                     this text
  :quit                     leave (also Ctrl-D)
  :vars                     list the variables you've defined
  :ast <code>               show the syntax tree of <code>
  :explain <code>           show which dialect <code> reads as
  :py/:js/:rb/:lua <code>   translate <code>
  :load <file>              run a file into this session
  :ai [backend]             show or set the AI backend (offline, claude, mock)
  :reset                    start a fresh session";

fn repl(mut engine: Engine) -> u8 {
    engine.set_auto_main(false);
    println!(
        "Chimera {} — any syntax, one language. :help for help, :quit to leave.",
        env!("CARGO_PKG_VERSION")
    );
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut n = 0usize;
    loop {
        let Some(input) = read_entry(&mut lines) else {
            println!();
            return 0;
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(cmd) = trimmed.strip_prefix(':') {
            if !repl_command(&mut engine, cmd) {
                return 0;
            }
            continue;
        }
        if matches!(trimmed, "exit" | "quit" | "exit()" | "quit()") {
            return 0;
        }
        n += 1;
        let name = format!("<repl:{n}>");
        let show = ends_with_expression(&mut engine, &input);
        match engine.run(&name, &input) {
            Ok(v) => {
                if show && !matches!(v, Value::Nil) {
                    let shown = engine.repr(&v);
                    println!("=> {shown}");
                }
            }
            Err(e) => {
                let _ = std::io::stdout().flush();
                eprint!("{}", e.report);
            }
        }
    }
}

/// Reads one complete entry, asking for more lines while it is unfinished.
fn read_entry(lines: &mut impl Iterator<Item = std::io::Result<String>>) -> Option<String> {
    let mut buf = String::new();
    let mut python_block = false;
    loop {
        print!("{}", if buf.is_empty() { "chi> " } else { "...> " });
        let _ = std::io::stdout().flush();
        let line = lines.next()?.ok()?;
        if !buf.is_empty() && line.trim().is_empty() {
            return Some(buf);
        }
        if buf.is_empty() && line.trim_start().starts_with(':') {
            return Some(line);
        }
        buf.push_str(&line);
        buf.push('\n');
        if line.trim_end().ends_with(':') {
            python_block = true;
        }
        if python_block {
            continue;
        }
        if is_complete(&buf) {
            return Some(buf);
        }
    }
}

/// Whether `src` is ready to run: brackets are balanced and it parses (or
/// fails somewhere other than at its very end, which needs more input).
pub fn is_complete(src: &str) -> bool {
    let mut depth = 0i32;
    let mut in_str: Option<char> = None;
    let mut prev = ' ';
    for c in src.chars() {
        match in_str {
            Some(q) => {
                if c == q && prev != '\\' {
                    in_str = None;
                }
            }
            None => match c {
                '"' | '`' => in_str = Some(c),
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                _ => {}
            },
        }
        prev = c;
    }
    if depth > 0 || in_str == Some('`') || needs_end(src) {
        return false;
    }
    let (_, errors) = chimera_lang::parser::parse(src, 0);
    if errors.is_empty() {
        return true;
    }
    // Unfinished blocks (`def f` … `end`, `do`, `then`) fail at the end.
    let end = src.trim_end().len();
    !errors
        .iter()
        .all(|d| d.primary_span().is_none_or(|s| s.start + 1 >= end))
}

/// Ruby/Lua/English blocks still waiting for their `end` (or `until`).
fn needs_end(src: &str) -> bool {
    let mut open: Vec<&str> = Vec::new();
    for line in src.lines() {
        let t = line.split(" #").next().unwrap_or("").trim();
        let first = t.split_whitespace().next().unwrap_or("");
        // Lua's `repeat … until cond` closes on `until`.
        if first == "until" && open.last() == Some(&"repeat") {
            open.pop();
            continue;
        }
        if first == "end" {
            open.pop();
            continue;
        }
        let block_kw = matches!(
            first,
            "def"
                | "class"
                | "module"
                | "if"
                | "unless"
                | "while"
                | "until"
                | "for"
                | "case"
                | "begin"
                | "function"
                | "repeat"
        );
        let one_line = t.ends_with(':')
            || t.contains('{')
            || t.ends_with(" end")
            || first == "def" && t.contains(" = ");
        let modifier =
            !block_kw && (t.contains(" if ") || t.contains(" unless ") || t.contains(" while "));
        let trailing_block =
            t.ends_with(" do") || t.contains(" do |") && t.ends_with('|') || t.ends_with(" then");
        if block_kw && !one_line {
            open.push(first);
        } else if !block_kw && !modifier && trailing_block {
            open.push("do");
        }
    }
    !open.is_empty()
}

fn ends_with_expression(engine: &mut Engine, src: &str) -> bool {
    match engine.parse("<repl-check>", src) {
        Ok(stmts) => match stmts.last().map(|s| &s.kind) {
            Some(StmtKind::Expr(e)) => !matches!(
                e.kind,
                ExprKind::Assign { .. } | ExprKind::CompoundAssign { .. } | ExprKind::IncDec { .. }
            ),
            _ => false,
        },
        Err(_) => false,
    }
}

/// Handles a `:command`; returns `false` to quit.
fn repl_command(engine: &mut Engine, cmd: &str) -> bool {
    let (name, arg) = match cmd.split_once(char::is_whitespace) {
        Some((n, a)) => (n, a.trim()),
        None => (cmd, ""),
    };
    match name {
        "q" | "quit" | "exit" => return false,
        "h" | "help" | "?" => println!("{REPL_HELP}"),
        "vars" | "v" => {
            let names = engine.global_names();
            if names.is_empty() {
                println!("(no variables yet)");
            }
            for n in names {
                if let Ok(v) = engine.run("<vars>", &n) {
                    let shown: String = engine.repr(&v).chars().take(70).collect();
                    println!("  {n} = {shown}");
                }
            }
        }
        "ast" => match engine.parse("<ast>", arg) {
            Ok(stmts) => println!("{stmts:#?}"),
            Err(e) => eprint!("{}", e.report),
        },
        "explain" => println!("reads as: {}", convert::detect_dialect(arg)),
        "py" | "python" | "js" | "javascript" | "rb" | "ruby" | "lua" | "chimera" => {
            let target = Target::from_name(name).unwrap_or(Target::Python);
            match engine.convert("<convert>", arg, target) {
                Ok(code) => {
                    for l in code.lines().skip(1) {
                        if !l.trim().is_empty() {
                            println!("{l}");
                        }
                    }
                }
                Err(e) => eprint!("{}", e.report),
            }
        }
        "load" | "l" => match std::fs::read_to_string(arg) {
            Ok(src) => {
                if let Err(e) = engine.run(arg, &src) {
                    eprint!("{}", e.report);
                }
            }
            Err(e) => eprintln!("error: cannot read {arg}: {e}"),
        },
        "ai" => {
            if let Some(b) = ai::Backend::from_name(arg) {
                engine.interp().ai.backend = b;
            }
            let cfg = &engine.interp().ai;
            println!("AI backend: {} (model {})", cfg.backend.name(), cfg.model);
        }
        "reset" => {
            *engine = Engine::new();
            engine.set_auto_main(false);
            println!("(fresh session)");
        }
        other => println!("unknown command `:{other}` (try :help)"),
    }
    true
}

#[cfg(test)]
mod tests {
    use super::is_complete;

    #[test]
    fn detects_unfinished_input() {
        assert!(is_complete("x = 1\n"));
        assert!(!is_complete("foo(1,\n"));
        assert!(!is_complete("def f\n  1\n"));
        assert!(is_complete("def f\n  1\nend\n"));
        assert!(!is_complete("fn main() {\n"));
        assert!(!is_complete("repeat\n  x = x + 1\n"));
        assert!(is_complete("repeat\n  x = x + 1\nuntil x > 3\n"));
        assert!(!is_complete("[1, 2].each do |x|\n"));
    }
}
