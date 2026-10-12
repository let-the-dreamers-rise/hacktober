//! # chimera-lang
//!
//! The Chimera programming language: write in any syntax.
//!
//! Chimera reads Python, JavaScript, C, Java, Go, Rust, Ruby, Lua, OCaml,
//! Haskell, Kotlin, Swift and plain English — mixed freely in one file —
//! and runs it on one tree-walking interpreter with arbitrary-precision
//! integers, tensors with automatic differentiation, neural networks, SQL
//! queries over collections and AI built into the syntax.
//!
//! ```
//! use chimera_lang::Engine;
//!
//! let mut engine = Engine::offline();
//! engine.capture_output();
//! engine.run("demo.chi", r#"
//!     def square(x): return x * x        # Python
//!     const xs = [1, 2, 3].map(square);  // JavaScript
//!     puts xs.sum                        # Ruby
//! "#).unwrap();
//! assert_eq!(engine.take_output(), "14\n");
//! ```
//!
//! Layout: [`lexer`] → [`parser`] (one recursive-descent parser for every
//! dialect) → [`ast`] → [`interp`] (evaluation) with [`value`] (runtime
//! values), [`builtins`] (the standard library), [`ml`] (tensors, autograd,
//! models) and [`ai`] (prompts, `please`, text tools).

pub mod ai;
pub mod ast;
pub mod bigint;
pub mod builtins;
pub mod interp;
pub mod lexer;
pub mod ml;
pub mod parser;
pub mod symbol;
pub mod value;

use std::path::PathBuf;

use chimera_core::{Diagnostic, Span};

pub use ast::FileId;
pub use interp::{Flow, Interp, Output};
pub use value::Value;

/// A failed run.
#[derive(Debug, Clone)]
pub struct Error {
    /// What kind of failure.
    pub kind: ErrorKind,
    /// One-line message (`ZeroDivisionError: division by zero`).
    pub message: String,
    /// The full report with source excerpts and stack trace.
    pub report: String,
}

/// Kinds of [`Error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// The program did not parse.
    Syntax,
    /// An exception escaped the program.
    Runtime,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.report)
    }
}

impl std::error::Error for Error {}

/// A Chimera interpreter session.
///
/// Definitions persist across [`Engine::run`] calls, which is what the
/// REPL relies on.
#[derive(Debug)]
pub struct Engine {
    interp: Interp,
    color: bool,
    auto_main: bool,
}

impl Default for Engine {
    fn default() -> Self {
        Engine::new()
    }
}

impl Engine {
    /// A fresh session with the standard library loaded. The AI backend is
    /// chosen from the environment (Claude when `ANTHROPIC_API_KEY` is set).
    pub fn new() -> Engine {
        Engine {
            interp: Interp::new(),
            color: false,
            auto_main: true,
        }
    }

    /// A session whose AI backend is the deterministic offline one.
    pub fn offline() -> Engine {
        let mut e = Engine::new();
        e.interp.ai = ai::AiConfig::offline();
        e
    }

    /// Colours error reports with ANSI escapes.
    pub fn set_color(&mut self, color: bool) {
        self.color = color;
    }

    /// Whether a `main` function that the program defines but never calls
    /// runs after the top level (on by default; the REPL turns it off).
    pub fn set_auto_main(&mut self, on: bool) {
        self.auto_main = on;
    }

    /// Buffers program output instead of writing it to stdout.
    pub fn capture_output(&mut self) {
        self.interp.output = Output::Buffer(String::new());
    }

    /// Takes the buffered output.
    pub fn take_output(&mut self) -> String {
        match &mut self.interp.output {
            Output::Buffer(b) => std::mem::take(b),
            Output::Stdout => String::new(),
        }
    }

    /// Sets the directory file imports resolve against.
    pub fn set_base_dir(&mut self, dir: impl Into<PathBuf>) {
        self.interp.base_dir = dir.into();
    }

    /// Sets `sys.argv` / `process.argv` / `ARGV`.
    pub fn set_args(&mut self, args: Vec<String>) {
        let list = Value::list(args.iter().map(|a| Value::str(a)).collect());
        self.interp
            .builtins
            .insert(symbol::Sym::new("__argv"), list);
    }

    /// The underlying interpreter.
    pub fn interp(&mut self) -> &mut Interp {
        &mut self.interp
    }

    /// Parses and runs `src`, returning the value of the last statement.
    pub fn run(&mut self, name: &str, src: &str) -> Result<Value, Error> {
        let file = self.interp.add_file(name, src);
        let (stmts, errors) = parser::parse(src, file);
        if !errors.is_empty() {
            return Err(self.syntax_error(file, &errors));
        }
        let result = self.interp.run_program(&stmts, file);
        let result = result.and_then(|v| self.interp.flush_defers().map(|()| v));
        let result = match result {
            Ok(v) if self.auto_main && !self.interp.main_called => {
                self.run_main(file).map(|r| r.unwrap_or(v))
            }
            other => other,
        };
        match result {
            Ok(v) => Ok(v),
            Err(flow) => Err(self.runtime_error(flow)),
        }
    }

    /// Calls the program's `main` — a top-level function or a class's
    /// static `main` (Java, C#) — if it was defined but not called.
    fn run_main(&mut self, file: FileId) -> Result<Option<Value>, Flow> {
        let main = symbol::Sym::new("main");
        let mut target = match self.interp.globals.get(main) {
            Some(f @ Value::Func(_)) => Some(f),
            _ => None,
        };
        if target.is_none() {
            for (_, v) in self.interp.globals.local_bindings() {
                if let Value::Class(c) = &v
                    && let Some(f) = c.find_static(main)
                {
                    target = Some(f);
                    break;
                }
            }
        }
        let Some(f) = target else { return Ok(None) };
        let wants_args = matches!(&f, Value::Func(c) if !c.def.params.is_empty());
        let args = if wants_args {
            let argv = self
                .interp
                .builtins
                .get(&symbol::Sym::new("__argv"))
                .cloned()
                .unwrap_or(Value::list(Vec::new()));
            // `main(String[] args)` gets the arguments after the program name.
            let rest = match &argv {
                Value::List(l) => Value::list(l.borrow().iter().skip(1).cloned().collect()),
                other => other.clone(),
            };
            vec![rest]
        } else {
            Vec::new()
        };
        let saved = self.interp.cur_file;
        self.interp.cur_file = file;
        let r = self
            .interp
            .call_value(f, interp::Args::new(args, Span::default()));
        self.interp.cur_file = saved;
        let r = r.and_then(|v| self.interp.flush_defers().map(|()| v));
        r.map(Some)
    }

    /// Parses `src` and returns the syntax tree (for `chimera explain`).
    pub fn parse(&mut self, name: &str, src: &str) -> Result<Vec<ast::Stmt>, Error> {
        let file = self.interp.add_file(name, src);
        let (stmts, errors) = parser::parse(src, file);
        if errors.is_empty() {
            Ok(stmts)
        } else {
            Err(self.syntax_error(file, &errors))
        }
    }

    /// `str(v)`.
    pub fn display(&mut self, v: &Value) -> String {
        self.interp
            .to_str(v)
            .unwrap_or_else(|_| interp::format::display_plain(v, false))
    }

    /// `repr(v)`.
    pub fn repr(&mut self, v: &Value) -> String {
        self.interp
            .repr(v)
            .unwrap_or_else(|_| interp::format::repr_plain(v))
    }

    /// Names defined at the top level (for REPL completion).
    pub fn global_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .interp
            .globals
            .local_names()
            .iter()
            .map(|s| s.as_str().to_string())
            .collect();
        names.sort();
        names
    }

    /// Names of built-ins (for REPL completion).
    pub fn builtin_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .interp
            .builtins
            .keys()
            .map(|s| s.as_str().to_string())
            .collect();
        names.sort();
        names
    }

    fn syntax_error(&self, file: FileId, errors: &[Diagnostic]) -> Error {
        let source = &self.interp.files[file as usize];
        let mut report = String::new();
        for d in errors.iter().take(10) {
            report.push_str(&d.render(source, self.color));
            report.push('\n');
        }
        if errors.len() > 10 {
            report.push_str(&format!("… and {} more errors\n", errors.len() - 10));
        }
        Error {
            kind: ErrorKind::Syntax,
            message: errors
                .first()
                .map(|d| d.message.clone())
                .unwrap_or_default(),
            report,
        }
    }

    fn runtime_error(&mut self, flow: Flow) -> Error {
        let thrown = match flow {
            Flow::Throw(t) => t,
            Flow::Return(_) | Flow::Break | Flow::Continue => {
                return Error {
                    kind: ErrorKind::Runtime,
                    message: "control flow escaped the program".into(),
                    report:
                        "error: `break`, `continue` or `return` outside of a loop or function\n"
                            .into(),
                };
            }
        };
        let message = self.exception_message(&thrown.value);
        let mut d = Diagnostic::error(message.clone());
        let mut file_for_render = None;
        if let Some((file, span)) = thrown.loc {
            d = d.with_label(span, "raised here");
            file_for_render = Some(file);
        }
        for entry in thrown.trace.iter().rev().take(12) {
            let f = &self.interp.files[entry.file as usize];
            let (line, col) = f.line_col(entry.span.start);
            d = d.with_note(format!("in {} at {}:{line}:{col}", entry.name, f.name()));
        }
        if thrown.trace.len() > 12 {
            d = d.with_note(format!("… {} more frames", thrown.trace.len() - 12));
        }
        let report = match file_for_render {
            Some(file) => d.render(&self.interp.files[file as usize], self.color),
            None => {
                let empty = chimera_core::SourceFile::new("<unknown>", "");
                Diagnostic::error(message.clone()).render(&empty, self.color)
            }
        };
        let _ = Span::default();
        Error {
            kind: ErrorKind::Runtime,
            message,
            report,
        }
    }

    /// `TypeError: …` for built-in errors, `Uncaught 42` for other values.
    fn exception_message(&mut self, v: &Value) -> String {
        match v {
            Value::Instance(inst) => {
                let class = inst.class.name.as_str();
                match inst.get(symbol::Sym::new("message")) {
                    Some(m) => {
                        let m = self.display(&m);
                        if m.is_empty() {
                            class.to_string()
                        } else {
                            format!("{class}: {m}")
                        }
                    }
                    None => {
                        let shown = self.display(v);
                        format!("Uncaught {shown}")
                    }
                }
            }
            other => {
                let shown = self.repr(other);
                format!("Uncaught {shown}")
            }
        }
    }
}
