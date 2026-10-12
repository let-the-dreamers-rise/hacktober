//! The tree-walking interpreter.
//!
//! Evaluation works directly on the core AST. Control flow that unwinds —
//! `return`, `break`, `continue` and exceptions — travels as the error side
//! of [`R`] in a [`Flow`], so ordinary `?` propagation does the right thing
//! and `finally`/`defer` blocks can intercept it.
//!
//! Scoping: functions get a fresh [`Env`]; blocks get one only when they
//! declare variables (`let`, `var`, …). Plain assignment (`x = 1`) updates
//! the nearest existing binding and otherwise creates a local in the current
//! function, which gives closures JavaScript/Ruby semantics while keeping
//! Python code working.

pub mod class;
pub mod format;
pub mod generator;
pub mod iter;
pub mod modules;
pub mod ops;
pub mod pattern;
pub mod query;

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use chimera_core::{Rng, SourceFile, Span};

use crate::ast::*;
use crate::symbol::Sym;
use crate::value::*;

/// Arguments to a call.
#[derive(Clone, Debug, Default)]
pub struct Args {
    /// Positional arguments.
    pub pos: Vec<Value>,
    /// Named arguments.
    pub named: Vec<(Sym, Value)>,
    /// Call site.
    pub span: Span,
}

impl Args {
    /// Positional-only arguments.
    pub fn new(pos: Vec<Value>, span: Span) -> Args {
        Args {
            pos,
            named: Vec::new(),
            span,
        }
    }

    /// The `i`th positional argument.
    pub fn get(&self, i: usize) -> Option<&Value> {
        self.pos.get(i)
    }

    /// A named argument.
    pub fn named(&self, name: &str) -> Option<Value> {
        self.named
            .iter()
            .find(|(n, _)| n.is(name))
            .map(|(_, v)| v.clone())
    }

    /// Positional argument `i` or the named argument `name`.
    pub fn arg(&self, i: usize, name: &str) -> Option<Value> {
        self.named(name).or_else(|| self.pos.get(i).cloned())
    }

    /// Number of positional arguments.
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// Whether there are no positional arguments.
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }
}

/// An unwinding control transfer.
#[derive(Debug)]
pub enum Flow {
    /// `return value`
    Return(Value),
    /// `break`
    Break,
    /// `continue`
    Continue,
    /// An exception.
    Throw(Box<Thrown>),
}

/// A thrown exception with its stack trace.
#[derive(Debug, Clone)]
pub struct Thrown {
    /// The exception value (usually an error instance).
    pub value: Value,
    /// Call stack at the throw, outermost first.
    pub trace: Vec<TraceEntry>,
    /// Where it was thrown.
    pub loc: Option<(FileId, Span)>,
}

/// One frame of a stack trace.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    /// Function name.
    pub name: Rc<str>,
    /// File of the call site.
    pub file: FileId,
    /// Call site.
    pub span: Span,
}

/// Kinds of environments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvKind {
    /// Module/program scope.
    Global,
    /// A function call.
    Function,
    /// A block that declares variables.
    Block,
}

/// A scope of variables.
pub struct Env {
    vars: RefCell<Vec<(Sym, Value)>>,
    map: Option<RefCell<HashMap<Sym, Value>>>,
    parent: Option<Rc<Env>>,
    kind: EnvKind,
}

impl std::fmt::Debug for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<env {:?}>", self.kind)
    }
}

impl Env {
    /// A module-level scope (hash-backed).
    pub fn global() -> Rc<Env> {
        Rc::new(Env {
            vars: RefCell::new(Vec::new()),
            map: Some(RefCell::new(HashMap::new())),
            parent: None,
            kind: EnvKind::Global,
        })
    }

    /// A child scope.
    pub fn child(parent: &Rc<Env>, kind: EnvKind) -> Rc<Env> {
        Rc::new(Env {
            vars: RefCell::new(Vec::new()),
            map: None,
            parent: Some(parent.clone()),
            kind,
        })
    }

    fn get_local(&self, s: Sym) -> Option<Value> {
        if let Some(map) = &self.map {
            return map.borrow().get(&s).cloned();
        }
        self.vars
            .borrow()
            .iter()
            .rev()
            .find(|(n, _)| *n == s)
            .map(|(_, v)| v.clone())
    }

    /// Looks a name up through the scope chain.
    pub fn get(&self, s: Sym) -> Option<Value> {
        let mut env = self;
        loop {
            if let Some(v) = env.get_local(s) {
                return Some(v);
            }
            env = env.parent.as_ref()?;
        }
    }

    /// Defines (or redefines) a name in this scope.
    pub fn define(&self, s: Sym, v: Value) {
        if let Some(map) = &self.map {
            map.borrow_mut().insert(s, v);
            return;
        }
        let mut vars = self.vars.borrow_mut();
        if let Some(slot) = vars.iter_mut().find(|(n, _)| *n == s) {
            slot.1 = v;
        } else {
            vars.push((s, v));
        }
    }

    fn set_local(&self, s: Sym, v: &Value) -> bool {
        if let Some(map) = &self.map {
            let mut m = map.borrow_mut();
            if let Some(slot) = m.get_mut(&s) {
                *slot = v.clone();
                return true;
            }
            return false;
        }
        let mut vars = self.vars.borrow_mut();
        if let Some(slot) = vars.iter_mut().find(|(n, _)| *n == s) {
            slot.1 = v.clone();
            true
        } else {
            false
        }
    }

    /// Updates an existing binding anywhere in the chain.
    pub fn assign(&self, s: Sym, v: &Value) -> bool {
        let mut env = self;
        loop {
            if env.set_local(s, v) {
                return true;
            }
            match &env.parent {
                Some(p) => env = p,
                None => return false,
            }
        }
    }

    /// Removes a binding from the nearest scope that has it.
    pub fn remove(&self, s: Sym) -> bool {
        let mut env = self;
        loop {
            let removed = if let Some(map) = &env.map {
                map.borrow_mut().remove(&s).is_some()
            } else {
                let mut vars = env.vars.borrow_mut();
                let before = vars.len();
                vars.retain(|(n, _)| *n != s);
                vars.len() != before
            };
            if removed {
                return true;
            }
            match &env.parent {
                Some(p) => env = p,
                None => return false,
            }
        }
    }

    /// The nearest function or global scope (where implicit locals live).
    pub fn function_scope(self: &Rc<Self>) -> Rc<Env> {
        let mut env = self.clone();
        while env.kind == EnvKind::Block {
            match &env.parent {
                Some(p) => env = p.clone(),
                None => break,
            }
        }
        env
    }

    /// Names defined directly in this scope.
    pub fn local_names(&self) -> Vec<Sym> {
        if let Some(map) = &self.map {
            return map.borrow().keys().copied().collect();
        }
        self.vars.borrow().iter().map(|(n, _)| *n).collect()
    }

    /// Names visible from this scope (innermost first).
    pub fn all_names(&self) -> Vec<Sym> {
        let mut out = self.local_names();
        if let Some(p) = &self.parent {
            out.extend(p.all_names());
        }
        out
    }

    /// Bindings defined directly in this scope.
    pub fn local_bindings(&self) -> Vec<(Sym, Value)> {
        if let Some(map) = &self.map {
            return map.borrow().iter().map(|(k, v)| (*k, v.clone())).collect();
        }
        self.vars.borrow().clone()
    }
}

/// Where `print` output goes.
#[derive(Debug)]
pub enum Output {
    /// The process's standard output.
    Stdout,
    /// An in-memory buffer (tests, embedding).
    Buffer(String),
}

struct Frame {
    name: Rc<str>,
    file: FileId,
    span: Span,
    class: Option<Rc<Class>>,
    receiver: Option<Value>,
}

/// Interpreter state.
pub struct Interp {
    /// Program-level variables.
    pub globals: Rc<Env>,
    /// Built-in functions, types and modules.
    pub builtins: HashMap<Sym, Value>,
    /// Output sink.
    pub output: Output,
    /// Seeded random number generator (`random`, weight init, dropout).
    pub rng: Rng,
    /// Registered source files (for error messages).
    pub files: Vec<SourceFile>,
    /// Maximum call depth before `RecursionError`.
    pub max_depth: usize,
    /// Directory used to resolve file imports.
    pub base_dir: PathBuf,
    /// AI configuration.
    pub ai: crate::ai::AiConfig,
    /// Number of expressions evaluated (a rough work counter).
    pub steps: u64,
    depth: usize,
    frames: Vec<Frame>,
    pub(crate) cur_file: FileId,
    handling: Vec<Value>,
    defers: Vec<Vec<(Block, Rc<Env>)>>,
    pub(crate) modules: HashMap<String, Value>,
    pub(crate) error_classes: HashMap<&'static str, Rc<Class>>,
    pub(crate) extensions: HashMap<TypeTag, HashMap<Sym, Value>>,
    pub(crate) rows: Vec<Vec<(Option<Sym>, Value)>>,
    pub(crate) aggregates: Vec<HashMap<usize, Value>>,
    /// Environment of the innermost native call site (for `eval`, `please`).
    pub(crate) call_env: Option<Rc<Env>>,
    /// Whether a function named `main` has been called (for auto-running
    /// `main` in Go, Rust, C, Java, Kotlin and Haskell programs).
    pub main_called: bool,
}

impl std::fmt::Debug for Interp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<interpreter>")
    }
}

/// Shorthand for a symbol.
pub(crate) fn sym(s: &str) -> Sym {
    Sym::new(s)
}

impl Interp {
    /// A fresh interpreter with the standard library loaded.
    pub fn new() -> Interp {
        let mut interp = Interp {
            globals: Env::global(),
            builtins: HashMap::new(),
            output: Output::Stdout,
            rng: Rng::new(0xC41_3E5A),
            files: Vec::new(),
            max_depth: 2_000,
            base_dir: PathBuf::from("."),
            ai: crate::ai::AiConfig::from_env(),
            steps: 0,
            depth: 0,
            frames: Vec::new(),
            cur_file: 0,
            handling: Vec::new(),
            defers: vec![Vec::new()],
            modules: HashMap::new(),
            error_classes: HashMap::new(),
            extensions: HashMap::new(),
            rows: Vec::new(),
            aggregates: Vec::new(),
            call_env: None,
            main_called: false,
        };
        crate::builtins::install(&mut interp);
        interp
    }

    /// Writes program output.
    pub fn write_out(&mut self, s: &str) {
        match &mut self.output {
            Output::Stdout => {
                use std::io::Write;
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(s.as_bytes());
                let _ = out.flush();
            }
            Output::Buffer(buf) => buf.push_str(s),
        }
    }

    /// Registers a source file, returning its id.
    pub fn add_file(&mut self, name: &str, text: &str) -> FileId {
        self.files.push(SourceFile::new(name, text));
        (self.files.len() - 1) as FileId
    }

    // ----- errors -------------------------------------------------------

    /// Builds an exception of a built-in class (`TypeError`, …).
    pub fn error(&mut self, kind: &str, msg: impl Into<String>, span: Span) -> Flow {
        let class = self
            .error_classes
            .get(kind)
            .or_else(|| self.error_classes.get("Error"))
            .cloned();
        let msg: String = msg.into();
        let value = match class {
            Some(c) => {
                let inst = Instance {
                    class: c,
                    fields: RefCell::new(vec![(sym("message"), Value::str(&msg))]),
                };
                Value::Instance(Rc::new(inst))
            }
            None => Value::str(&msg),
        };
        self.throw(value, span)
    }

    /// Throws `value` from `span`, capturing the stack trace.
    pub fn throw(&mut self, value: Value, span: Span) -> Flow {
        Flow::Throw(Box::new(Thrown {
            value,
            trace: self.trace(),
            loc: Some((self.cur_file, span)),
        }))
    }

    fn trace(&self) -> Vec<TraceEntry> {
        self.frames
            .iter()
            .map(|f| TraceEntry {
                name: f.name.clone(),
                file: f.file,
                span: f.span,
            })
            .collect()
    }

    /// `Err(TypeError …)` helper.
    pub fn type_err<T>(&mut self, msg: impl Into<String>, span: Span) -> R<T> {
        Err(self.error("TypeError", msg, span))
    }

    /// `Err(ValueError …)` helper.
    pub fn value_err<T>(&mut self, msg: impl Into<String>, span: Span) -> R<T> {
        Err(self.error("ValueError", msg, span))
    }

    // ----- programs and blocks -------------------------------------------

    /// Runs top-level statements in the global scope.
    pub fn run_program(&mut self, stmts: &[Stmt], file: FileId) -> R {
        let globals = self.globals.clone();
        self.run_in(stmts, file, &globals)
    }

    /// Runs statements in `env`, then any pending top-level `defer`s.
    pub fn run_in(&mut self, stmts: &[Stmt], file: FileId, env: &Rc<Env>) -> R {
        let saved = self.cur_file;
        self.cur_file = file;
        self.hoist(stmts, env);
        let mut last = Value::Nil;
        let mut result = Ok(Value::Nil);
        for stmt in stmts {
            match self.exec(stmt, env, true) {
                Ok(v) => last = v,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        let deferred = std::mem::take(self.defers.last_mut().unwrap_or(&mut Vec::new()));
        let result = self.run_defers(deferred, result.map(|_| last));
        self.cur_file = saved;
        match result {
            Err(Flow::Return(v)) => Ok(v),
            Err(Flow::Break) | Err(Flow::Continue) => Ok(Value::Nil),
            other => other,
        }
    }

    /// Runs generated or `eval`'d statements in an existing scope, without
    /// touching the enclosing function's `defer` stack.
    pub(crate) fn exec_snippet(&mut self, stmts: &[Stmt], file: FileId, env: &Rc<Env>) -> R {
        let saved = self.cur_file;
        self.cur_file = file;
        self.hoist(stmts, env);
        let mut last = Ok(Value::Nil);
        for stmt in stmts {
            last = self.exec(stmt, env, true);
            if last.is_err() {
                break;
            }
        }
        self.cur_file = saved;
        last
    }

    fn hoist(&mut self, stmts: &[Stmt], env: &Rc<Env>) {
        for stmt in stmts {
            if let StmtKind::Fn(def) = &stmt.kind
                && let Some(name) = def.name
                && def.decorators.is_empty()
            {
                let f = self.make_closure(def, env);
                env.define(name, f);
            }
        }
    }

    pub(crate) fn make_closure(&self, def: &Rc<FnDef>, env: &Rc<Env>) -> Value {
        Value::Func(Rc::new(Closure {
            def: def.clone(),
            env: env.clone(),
            class: RefCell::new(None),
        }))
    }

    /// Evaluates a block; its value is that of its last expression statement.
    pub fn eval_block(&mut self, block: &Block, env: &Rc<Env>) -> R {
        let env = if block.scoped {
            Env::child(env, EnvKind::Block)
        } else {
            env.clone()
        };
        self.hoist(&block.stmts, &env);
        let mut last = Value::Nil;
        let n = block.stmts.len();
        for (i, stmt) in block.stmts.iter().enumerate() {
            last = self.exec(stmt, &env, i + 1 < n)?;
        }
        Ok(last)
    }

    // ----- statements ------------------------------------------------------

    /// Executes one statement. `auto_call` allows a bare function name used
    /// as a statement to be called (Ruby/Perl style).
    pub fn exec(&mut self, stmt: &Stmt, env: &Rc<Env>, auto_call: bool) -> R {
        self.steps += 1;
        match &stmt.kind {
            StmtKind::Expr(e) => {
                let v = self.eval(e, env)?;
                if auto_call
                    && matches!(e.kind, ExprKind::Name(_) | ExprKind::Member { .. })
                    && self.zero_arity_callable(&v)
                {
                    return self.call_value(v, Args::new(Vec::new(), e.span));
                }
                Ok(v)
            }
            StmtKind::Let { pattern, ty, value } => {
                let mut v = match value {
                    Some(e) => self.eval(e, env)?,
                    None => Value::Nil,
                };
                if let Some(t) = ty {
                    v = self.coerce_declared(v, t, stmt.span)?;
                }
                self.bind_pattern(pattern, v.clone(), env, stmt.span)?;
                Ok(v)
            }
            StmtKind::Fn(def) => {
                let mut f = self.make_closure(def, env);
                for d in def.decorators.iter().rev() {
                    let dec = self.eval(d, env)?;
                    f = self.call_value(dec, Args::new(vec![f], d.span))?;
                }
                if let Some(name) = def.name {
                    env.define(name, f.clone());
                }
                Ok(Value::Nil)
            }
            StmtKind::Class(def) => {
                let class = self.define_class(def, env)?;
                env.define(def.name, class);
                Ok(Value::Nil)
            }
            StmtKind::Impl {
                target,
                methods,
                statics,
            } => {
                self.exec_impl(target, methods, statics, env, stmt.span)?;
                Ok(Value::Nil)
            }
            StmtKind::Enum(def) => {
                let e = self.define_enum(def, env)?;
                env.define(def.name, e);
                Ok(Value::Nil)
            }
            StmtKind::While {
                cond,
                body,
                negate,
                post,
            } => {
                let mut first = true;
                loop {
                    if !(*post && first) {
                        let c = self.eval(cond, env)?.truthy();
                        if c == *negate {
                            break;
                        }
                    }
                    first = false;
                    match self.eval_block(body, env) {
                        Ok(_) | Err(Flow::Continue) => {}
                        Err(Flow::Break) => break,
                        Err(e) => return Err(e),
                    }
                }
                Ok(Value::Nil)
            }
            StmtKind::Loop(body) => {
                loop {
                    match self.eval_block(body, env) {
                        Ok(_) | Err(Flow::Continue) => {}
                        Err(Flow::Break) => break,
                        Err(e) => return Err(e),
                    }
                }
                Ok(Value::Nil)
            }
            StmtKind::Repeat { count, body } => {
                let n = self.eval(count, env)?;
                let Some(n) = n.as_int() else {
                    return self.type_err(
                        format!("`repeat` needs a whole number, got {}", n.type_name()),
                        count.span,
                    );
                };
                for _ in 0..n.max(0) {
                    match self.eval_block(body, env) {
                        Ok(_) | Err(Flow::Continue) => {}
                        Err(Flow::Break) => break,
                        Err(e) => return Err(e),
                    }
                }
                Ok(Value::Nil)
            }
            StmtKind::For {
                pattern,
                iter,
                body,
                fresh_scope,
            } => {
                let mut it = self.eval(iter, env)?;
                // `for k, v in dict` iterates entries, like `.items()`.
                if let (Value::Map(m), Pattern::Tuple(p)) = (&it, pattern)
                    && p.len() == 2
                {
                    let entries = m
                        .borrow()
                        .iter()
                        .map(|(k, v)| Value::tuple(vec![k.clone(), v.clone()]))
                        .collect();
                    it = Value::list(entries);
                }
                let mut items = self.iterate(&it, iter.span)?;
                let target = env.clone();
                while let Some(item) = items.next(self)? {
                    let scope = if *fresh_scope {
                        Env::child(env, EnvKind::Block)
                    } else {
                        target.clone()
                    };
                    self.bind_pattern(pattern, item, &scope, iter.span)?;
                    match self.eval_block(body, &scope) {
                        Ok(_) | Err(Flow::Continue) => {}
                        Err(Flow::Break) => break,
                        Err(e) => return Err(e),
                    }
                }
                Ok(Value::Nil)
            }
            StmtKind::CFor {
                init,
                cond,
                step,
                body,
            } => {
                let scope = Env::child(env, EnvKind::Block);
                if let Some(init) = init {
                    self.exec(init, &scope, false)?;
                }
                loop {
                    if let Some(c) = cond
                        && !self.eval(c, &scope)?.truthy()
                    {
                        break;
                    }
                    match self.eval_block(body, &scope) {
                        Ok(_) | Err(Flow::Continue) => {}
                        Err(Flow::Break) => break,
                        Err(e) => return Err(e),
                    }
                    if let Some(s) = step {
                        self.eval(s, &scope)?;
                    }
                }
                Ok(Value::Nil)
            }
            StmtKind::Return(e) => {
                let v = match e {
                    Some(e) => self.eval(e, env)?,
                    None => Value::Nil,
                };
                Err(Flow::Return(v))
            }
            StmtKind::Break => Err(Flow::Break),
            StmtKind::Continue => Err(Flow::Continue),
            StmtKind::Throw(e) => {
                let v = self.eval(e, env)?;
                let v = match v {
                    Value::Nil => match self.handling.last() {
                        Some(current) => current.clone(),
                        None => {
                            return Err(self.error(
                                "RuntimeError",
                                "re-raised with no active exception",
                                stmt.span,
                            ));
                        }
                    },
                    Value::Str(s) => {
                        let s = s.to_string();
                        return Err(self.error("RuntimeError", s, e.span));
                    }
                    Value::Class(c) => self.instantiate(&c, Args::new(Vec::new(), e.span))?,
                    other => other,
                };
                Err(self.throw(v, e.span))
            }
            StmtKind::Try {
                body,
                catches,
                else_block,
                finally,
            } => self.exec_try(body, catches, else_block.as_ref(), finally.as_ref(), env),
            StmtKind::Switch {
                subject,
                cases,
                default,
            } => {
                let v = self.eval(subject, env)?;
                let mut chosen = None;
                'outer: for case in cases {
                    for value in &case.values {
                        if self.case_eq(value, &v, env)? {
                            chosen = Some(&case.body);
                            break 'outer;
                        }
                    }
                }
                let body = chosen.or(default.as_ref());
                if let Some(body) = body {
                    match self.eval_block(body, env) {
                        Ok(v) => return Ok(v),
                        Err(Flow::Break) => {}
                        Err(e) => return Err(e),
                    }
                }
                Ok(Value::Nil)
            }
            StmtKind::Import(imp) => {
                self.exec_import(imp, env, stmt.span)?;
                Ok(Value::Nil)
            }
            StmtKind::Defer(block) => {
                // Go evaluates a deferred call's function and arguments now
                // and calls it later; other `defer` blocks run as written.
                let (block, denv) = match block.stmts.as_slice() {
                    [
                        Stmt {
                            kind:
                                StmtKind::Expr(Expr {
                                    kind: ExprKind::Call { callee, args },
                                    span,
                                }),
                            ..
                        },
                    ] if args.iter().all(|a| matches!(a, Arg::Pos(_))) => {
                        let denv = Env::child(env, EnvKind::Block);
                        let f = self.eval(callee, env)?;
                        denv.define(sym("__deferred_fn"), f);
                        let mut new_args = Vec::with_capacity(args.len());
                        for (i, a) in args.iter().enumerate() {
                            if let Arg::Pos(e) = a {
                                let v = self.eval(e, env)?;
                                let name = sym(&format!("__deferred_{i}"));
                                denv.define(name, v);
                                new_args.push(Arg::Pos(Expr::new(ExprKind::Name(name), e.span)));
                            }
                        }
                        let call = Expr::new(
                            ExprKind::Call {
                                callee: Box::new(Expr::new(
                                    ExprKind::Name(sym("__deferred_fn")),
                                    callee.span,
                                )),
                                args: new_args,
                            },
                            *span,
                        );
                        (Block::of_expr(call), denv)
                    }
                    _ => (block.clone(), env.clone()),
                };
                if let Some(list) = self.defers.last_mut() {
                    list.push((block, denv));
                }
                Ok(Value::Nil)
            }
            StmtKind::Delete(target) => {
                self.delete(target, env)?;
                Ok(Value::Nil)
            }
            StmtKind::Model(def) => {
                let model = crate::ml::nn::define_model(self, def, env)?;
                env.define(def.name, model);
                Ok(Value::Nil)
            }
            StmtKind::Train(spec) => crate::ml::nn::train_statement(self, spec, env),
            StmtKind::Pass => Ok(Value::Nil),
        }
    }

    fn zero_arity_callable(&self, v: &Value) -> bool {
        match v {
            Value::Func(c) => c.required_params() == 0 && c.def.name.is_some(),
            Value::Bound(b) => match &b.func {
                Value::Func(c) => c.required_params() == 0,
                _ => false,
            },
            // A bare `puts` / `print` / `println` statement prints a newline.
            Value::Native(n) => matches!(&*n.name, "puts" | "println" | "print"),
            _ => false,
        }
    }

    fn exec_try(
        &mut self,
        body: &Block,
        catches: &[Catch],
        else_block: Option<&Block>,
        finally: Option<&Block>,
        env: &Rc<Env>,
    ) -> R {
        let mut result = self.eval_block(body, env);
        if let Err(Flow::Throw(thrown)) = &result {
            let thrown = thrown.clone();
            let mut handled = None;
            for c in catches {
                if self.catch_matches(c, &thrown.value, env)? {
                    handled = Some(c);
                    break;
                }
            }
            if let Some(c) = handled {
                let scope = Env::child(env, EnvKind::Block);
                if let Some(name) = c.binding {
                    scope.define(name, thrown.value.clone());
                }
                self.handling.push(thrown.value.clone());
                result = self.eval_block(&c.body, &scope);
                self.handling.pop();
            }
        } else if result.is_ok()
            && let Some(e) = else_block
        {
            result = self.eval_block(e, env);
        }
        if let Some(f) = finally {
            match self.eval_block(f, env) {
                Ok(_) => {}
                Err(e) => return Err(e),
            }
        }
        result
    }

    fn catch_matches(&mut self, c: &Catch, value: &Value, env: &Rc<Env>) -> R<bool> {
        if c.classes.is_empty() {
            return Ok(true);
        }
        for class_expr in &c.classes {
            let class = self.eval(class_expr, env)?;
            if self.exception_matches(value, &class) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether a thrown `value` is caught by a handler for `class`.
    pub fn exception_matches(&self, value: &Value, class: &Value) -> bool {
        const BASES: &[&str] = &[
            "Exception",
            "Error",
            "StandardError",
            "BaseException",
            "Throwable",
            "RuntimeException",
            "object",
        ];
        match (value, class) {
            (Value::Instance(i), Value::Class(c)) => {
                i.class.is_subclass_of(c)
                    || BASES.contains(&&*c.name.as_str()) && i.class.has_ancestor_named("Error")
            }
            (_, Value::Class(c)) => BASES.contains(&&*c.name.as_str()),
            (_, Value::Type(TypeTag::Any)) => true,
            (v, Value::Type(t)) => self.type_tag_matches(v, *t),
            _ => false,
        }
    }

    fn run_defers(&mut self, deferred: Vec<(Block, Rc<Env>)>, mut result: R) -> R {
        for (block, env) in deferred.into_iter().rev() {
            if let Err(e) = self.eval_block(&block, &env)
                && !matches!(e, Flow::Return(_))
            {
                result = Err(e);
            }
        }
        result
    }

    fn delete(&mut self, target: &Expr, env: &Rc<Env>) -> R<()> {
        match &target.kind {
            ExprKind::Name(n) => {
                if !env.remove(*n) {
                    return Err(self.error(
                        "NameError",
                        format!("name `{n}` is not defined"),
                        target.span,
                    ));
                }
                Ok(())
            }
            ExprKind::Index { object, index } => {
                let obj = self.eval(object, env)?;
                let idx = self.eval(index, env)?;
                match &obj {
                    Value::Map(m) | Value::Set(m) => {
                        if m.borrow_mut().remove(&idx).is_none() {
                            let k = format::repr_plain(&idx);
                            return Err(self.error("KeyError", k, index.span));
                        }
                        Ok(())
                    }
                    Value::List(l) => {
                        let len = l.borrow().len();
                        let i = self.list_index(&idx, len, index.span)?;
                        l.borrow_mut().remove(i);
                        Ok(())
                    }
                    other => {
                        let t = other.type_name();
                        self.type_err(format!("cannot delete items from {t}"), target.span)
                    }
                }
            }
            ExprKind::Member { object, name, .. } => {
                let obj = self.eval(object, env)?;
                match &obj {
                    Value::Instance(i) => {
                        i.remove(*name);
                        Ok(())
                    }
                    Value::Map(m) => {
                        m.borrow_mut().remove(&Value::Str(name.as_str()));
                        Ok(())
                    }
                    _ => self.type_err("cannot delete this attribute", target.span),
                }
            }
            _ => self.type_err("cannot delete this expression", target.span),
        }
    }

    // ----- expressions --------------------------------------------------------

    /// Evaluates an expression.
    pub fn eval(&mut self, e: &Expr, env: &Rc<Env>) -> R {
        self.steps += 1;
        match &e.kind {
            ExprKind::Nil => Ok(Value::Nil),
            ExprKind::Bool(b) => Ok(Value::Bool(*b)),
            ExprKind::Int(i) => Ok(Value::Int(*i)),
            ExprKind::BigInt(b) => Ok(Value::Big(b.clone())),
            ExprKind::Float(f) => Ok(Value::Float(*f)),
            ExprKind::Str(s) => Ok(Value::Str(s.clone())),
            ExprKind::Interp(parts) => {
                let mut out = String::new();
                for part in parts {
                    match part {
                        InterpPart::Lit(s) => out.push_str(s),
                        InterpPart::Expr(expr, Some(spec))
                            if &**spec == crate::lexer::SOFT_INTERPOLATION =>
                        {
                            match &expr.kind {
                                ExprKind::Name(n)
                                    if env.get(*n).is_none()
                                        && self.implicit_member(*n).is_none()
                                        && !self.builtins.contains_key(n) =>
                                {
                                    out.push('$');
                                    out.push_str(&n.as_str());
                                }
                                _ => {
                                    let v = self.eval(expr, env)?;
                                    out.push_str(&self.to_str(&v)?);
                                }
                            }
                        }
                        InterpPart::Expr(expr, spec) => {
                            let v = self.eval(expr, env)?;
                            let s = match spec {
                                Some(spec) => self.format_spec(&v, spec, expr.span)?,
                                None => self.to_str(&v)?,
                            };
                            out.push_str(&s);
                        }
                    }
                }
                Ok(Value::str(&out))
            }
            ExprKind::Name(n) => {
                let v = self.lookup(*n, env, e.span)?;
                if let Value::Func(c) = &v
                    && c.def.bare
                    && c.required_params() == 0
                {
                    return self.call_value(v, Args::new(Vec::new(), e.span));
                }
                Ok(v)
            }
            ExprKind::SelfRef => match env.get(sym("self")) {
                Some(v) => Ok(v),
                None => Err(self.error("NameError", "`self` used outside of a method", e.span)),
            },
            ExprKind::Super => Err(self.error(
                "TypeError",
                "`super` must be called or used with `.method`",
                e.span,
            )),
            ExprKind::List(items) => Ok(Value::list(self.eval_items(items, env)?)),
            ExprKind::Tuple(items) => {
                let v = self.eval_items(items, env)?;
                if v.is_empty() && items.is_empty() {
                    return Ok(Value::tuple(Vec::new()));
                }
                Ok(Value::tuple(v))
            }
            ExprKind::Set(items) => {
                let mut d = Dict::new();
                for v in self.eval_items(items, env)? {
                    if !d.insert(v.clone(), Value::Nil) {
                        let t = v.type_name();
                        return self.type_err(format!("unhashable type: {t}"), e.span);
                    }
                }
                Ok(Value::Set(Rc::new(RefCell::new(d))))
            }
            ExprKind::Map(entries) => {
                let mut d = Dict::new();
                for (k, v) in entries {
                    if let ExprKind::Spread(inner) = &k.kind {
                        let other = self.eval(inner, env)?;
                        match &other {
                            Value::Map(m) => {
                                for (k, v) in m.borrow().iter() {
                                    d.insert(k.clone(), v.clone());
                                }
                            }
                            Value::Instance(i) => {
                                for (k, v) in i.fields.borrow().iter() {
                                    d.insert(Value::Str(k.as_str()), v.clone());
                                }
                            }
                            other => {
                                let t = other.type_name();
                                return self
                                    .type_err(format!("cannot spread {t} into a map"), k.span);
                            }
                        }
                        continue;
                    }
                    let key = self.eval(k, env)?;
                    let val = self.eval(v, env)?;
                    if !d.insert(key.clone(), val) {
                        let t = key.type_name();
                        return self.type_err(format!("unhashable key type: {t}"), k.span);
                    }
                }
                Ok(Value::Map(Rc::new(RefCell::new(d))))
            }
            ExprKind::Spread(inner) => self.eval(inner, env),
            ExprKind::Range {
                start,
                end,
                inclusive,
                step,
            } => self.eval_range(
                start.as_deref(),
                end.as_deref(),
                *inclusive,
                step.as_deref(),
                env,
                e.span,
            ),
            ExprKind::Unary(op, operand) => {
                let v = self.eval(operand, env)?;
                self.unary_op(*op, v, e.span)
            }
            ExprKind::Binary(op, a, b) => {
                if *op == BinOp::Pipe {
                    let x = self.eval(a, env)?;
                    return self.pipe(x, b, env, e.span);
                }
                let x = self.eval(a, env)?;
                let y = self.eval(b, env)?;
                self.binary_op(*op, x, y, e.span)
            }
            ExprKind::Compare(first, rest) => {
                let mut left = self.eval(first, env)?;
                for (op, rhs) in rest {
                    let right = self.eval(rhs, env)?;
                    let r = self.compare_op(*op, &left, &right, rhs.span)?;
                    if rest.len() == 1 {
                        return Ok(r);
                    }
                    if !r.truthy() {
                        return Ok(Value::Bool(false));
                    }
                    left = right;
                }
                Ok(Value::Bool(true))
            }
            ExprKind::And(a, b) => {
                let x = self.eval(a, env)?;
                if !x.truthy() {
                    return Ok(x);
                }
                self.eval(b, env)
            }
            ExprKind::Or(a, b) => {
                let x = self.eval(a, env)?;
                if x.truthy() {
                    return Ok(x);
                }
                self.eval(b, env)
            }
            ExprKind::Coalesce(a, b) => {
                let x = self.eval(a, env)?;
                if !matches!(x, Value::Nil) {
                    return Ok(x);
                }
                self.eval(b, env)
            }
            ExprKind::Call { callee, args } => {
                if !self.aggregates.is_empty()
                    && let Some(v) = query::precomputed(self, e)
                {
                    return Ok(v);
                }
                self.eval_call(callee, args, env, e.span)
            }
            ExprKind::Member {
                object,
                name,
                optional,
            } => {
                if matches!(object.kind, ExprKind::Super) {
                    return self.super_member(*name, e.span);
                }
                let obj = self.eval(object, env)?;
                if *optional && matches!(obj, Value::Nil) {
                    return Ok(Value::Nil);
                }
                self.get_attr(&obj, *name, e.span)
            }
            ExprKind::Index { object, index } => {
                let obj = self.eval(object, env)?;
                self.eval_index(&obj, index, env, e.span)
            }
            ExprKind::Slice { .. } => self.type_err("a slice can only appear inside `[ ]`", e.span),
            ExprKind::Lambda(def) => Ok(self.make_closure(def, env)),
            ExprKind::If {
                cond,
                then,
                otherwise,
            } => {
                if self.eval(cond, env)?.truthy() {
                    self.eval_block(then, env)
                } else if let Some(o) = otherwise {
                    self.eval_block(o, env)
                } else {
                    Ok(Value::Nil)
                }
            }
            ExprKind::Match { subject, arms } => {
                self.eval_match(subject.as_deref(), arms, env, e.span)
            }
            ExprKind::Block(b) => self.eval_block(b, env),
            ExprKind::Comprehension(c) => self.eval_comprehension(c, env),
            ExprKind::Assign { target, value } => {
                let v = self.eval(value, env)?;
                self.assign(target, v.clone(), env)?;
                Ok(v)
            }
            ExprKind::CompoundAssign { op, target, value } => {
                let current = self.eval(target, env)?;
                let rhs = self.eval(value, env)?;
                let new = self.compound_value(*op, current, rhs, e.span)?;
                self.assign(target, new.clone(), env)?;
                Ok(new)
            }
            ExprKind::IncDec {
                target,
                delta,
                prefix,
            } => {
                let current = self.eval(target, env)?;
                let new =
                    self.binary_op(BinOp::Add, current.clone(), Value::Int(*delta), e.span)?;
                self.assign(target, new.clone(), env)?;
                Ok(if *prefix { new } else { current })
            }
            ExprKind::Query(q) => self.eval_query(q, env, e.span),
            ExprKind::Derivative { func, order } => {
                let f = self.eval(func, env)?;
                Ok(crate::ml::autodiff::derivative_fn(f, *order))
            }
            ExprKind::Gradient(func) => {
                let f = self.eval(func, env)?;
                Ok(crate::ml::autodiff::gradient_fn(f))
            }
            ExprKind::StructLit { ty, fields } => self.eval_struct_lit(ty, fields, env, e.span),
            ExprKind::Please(text) => crate::ai::nl::please(self, text, env, e.span),
            ExprKind::Jump(stmt) => self.exec(stmt, env, true),
            ExprKind::Let { pattern, value } => {
                let v = self.eval(value, env)?;
                // A bare name only binds a present value (Swift `if let x = opt`).
                if matches!(**pattern, Pattern::Bind(_)) && matches!(v, Value::Nil) {
                    return Ok(Value::Bool(false));
                }
                Ok(Value::Bool(self.match_into(pattern, &v, env, e.span)?))
            }
            ExprKind::Yield { values, delegate } => {
                // Outside the generator stepper, `yield` calls the Ruby block
                // the method was given.
                let Some(block) = env.get(sym("__block__")) else {
                    return Err(self.error(
                        "SyntaxError",
                        "`yield` outside of a generator (or a method called without a block)",
                        e.span,
                    ));
                };
                let mut vals = Vec::with_capacity(values.len());
                for v in values {
                    vals.push(self.eval(v, env)?);
                }
                if *delegate {
                    let src = vals.pop().unwrap_or(Value::Nil);
                    let mut last = Value::Nil;
                    for item in self.collect(&src, e.span)? {
                        last = self.call_value(block.clone(), Args::new(vec![item], e.span))?;
                    }
                    return Ok(last);
                }
                self.call_value(block, Args::new(vals, e.span))
            }
        }
    }

    fn eval_items(&mut self, items: &[Expr], env: &Rc<Env>) -> R<Vec<Value>> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            if let ExprKind::Spread(inner) = &item.kind {
                let v = self.eval(inner, env)?;
                let mut it = self.iterate(&v, inner.span)?;
                while let Some(x) = it.next(self)? {
                    out.push(x);
                }
            } else {
                out.push(self.eval(item, env)?);
            }
        }
        Ok(out)
    }

    /// Looks a variable up (scopes, then built-ins), with suggestions.
    pub fn lookup(&mut self, n: Sym, env: &Rc<Env>, span: Span) -> R {
        // Inside a query, columns of the current row shadow variables.
        if !self.rows.is_empty()
            && let Some(v) = self.row_lookup(n)
        {
            return Ok(v);
        }
        if let Some(v) = env.get(n) {
            return Ok(v);
        }
        if let Some(v) = self.implicit_member(n) {
            return Ok(v);
        }
        if let Some(v) = self.builtins.get(&n) {
            return Ok(v.clone());
        }
        let name = n.as_str();
        let names: Vec<Rc<str>> = env
            .all_names()
            .into_iter()
            .chain(self.builtins.keys().copied())
            .map(Sym::as_str)
            .collect();
        let suggestion = chimera_core::text::suggest(&name, names.iter().map(|s| &**s));
        let msg = match suggestion {
            Some(s) => format!("name `{name}` is not defined (did you mean `{s}`?)"),
            None => format!("name `{name}` is not defined"),
        };
        Err(self.error("NameError", msg, span))
    }

    fn eval_range(
        &mut self,
        start: Option<&Expr>,
        end: Option<&Expr>,
        inclusive: bool,
        step: Option<&Expr>,
        env: &Rc<Env>,
        span: Span,
    ) -> R {
        let s = match start {
            Some(e) => self.eval(e, env)?,
            None => Value::Int(0),
        };
        let Some(end) = end else {
            return self.value_err("this range has no end", span);
        };
        let e = self.eval(end, env)?;
        let st = match step {
            Some(x) => Some(self.eval(x, env)?),
            None => None,
        };
        self.make_range(&s, &e, inclusive, st.as_ref(), span)
    }

    /// Builds a range value (integers) or a list (floats/characters).
    pub fn make_range(
        &mut self,
        s: &Value,
        e: &Value,
        inclusive: bool,
        step: Option<&Value>,
        span: Span,
    ) -> R {
        if let (Value::Str(a), Value::Str(b)) = (s, e)
            && a.chars().count() == 1
            && b.chars().count() == 1
        {
            let (a, b) = (
                a.chars().next().unwrap_or('a') as u32,
                b.chars().next().unwrap_or('a') as u32,
            );
            let hi = if inclusive { b + 1 } else { b };
            let items = (a..hi)
                .filter_map(char::from_u32)
                .map(|c| Value::str(&c.to_string()))
                .collect();
            return Ok(Value::list(items));
        }
        match (s.as_int(), e.as_int(), step.map(Value::as_int)) {
            (Some(a), Some(b), None | Some(Some(_)))
                if matches!(s, Value::Int(_))
                    && matches!(e, Value::Int(_) | Value::Big(_) | Value::Float(_))
                    || s.as_int().is_some()
                        && e.as_int().is_some()
                        && !matches!(s, Value::Float(f) if f.fract() != 0.0) =>
            {
                let step = step
                    .and_then(Value::as_int)
                    .unwrap_or(if b < a && !inclusive && false { -1 } else { 1 });
                if step == 0 {
                    return self.value_err("range step cannot be zero", span);
                }
                let end = if inclusive {
                    if step > 0 {
                        b.saturating_add(1)
                    } else {
                        b.saturating_sub(1)
                    }
                } else {
                    b
                };
                Ok(Value::Range(Rc::new(RangeVal {
                    start: a,
                    end,
                    step,
                })))
            }
            _ => {
                let (Some(a), Some(b)) = (s.as_f64(), e.as_f64()) else {
                    return self.type_err(
                        format!(
                            "cannot make a range from {} to {}",
                            s.type_name(),
                            e.type_name()
                        ),
                        span,
                    );
                };
                let step = step.and_then(Value::as_f64).unwrap_or(1.0);
                if step == 0.0 {
                    return self.value_err("range step cannot be zero", span);
                }
                let mut out = Vec::new();
                let mut x = a;
                let n_max = 10_000_000;
                while (step > 0.0 && (x < b || inclusive && x <= b + 1e-12))
                    || (step < 0.0 && (x > b || inclusive && x >= b - 1e-12))
                {
                    out.push(Value::Float(x));
                    x = a + step * out.len() as f64;
                    if out.len() > n_max {
                        return self.value_err("range is too large", span);
                    }
                }
                Ok(Value::list(out))
            }
        }
    }

    fn pipe(&mut self, x: Value, rhs: &Expr, env: &Rc<Env>, span: Span) -> R {
        // `x |> f(a, b)` → `f(x, a, b)`; `x |> f` → `f(x)`.
        if let ExprKind::Call { callee, args } = &rhs.kind {
            let mut a = self.eval_args(args, env)?;
            a.pos.insert(0, x);
            a.span = span;
            if let ExprKind::Member { object, name, .. } = &callee.kind {
                let obj = self.eval(object, env)?;
                return self.call_method(&obj, *name, a, span);
            }
            let f = self.eval(callee, env)?;
            return self.call_value(f, a);
        }
        let f = self.eval(rhs, env)?;
        self.call_value(f, Args::new(vec![x], span))
    }

    fn compound_value(&mut self, op: BinOp, current: Value, rhs: Value, span: Span) -> R {
        // In-place list extension keeps aliasing semantics (`xs += [1]`).
        if op == BinOp::Add
            && let (Value::List(l), Value::List(r)) = (&current, &rhs)
        {
            let extra = r.borrow().clone();
            l.borrow_mut().extend(extra);
            return Ok(current);
        }
        self.binary_op(op, current, rhs, span)
    }

    // ----- calls -------------------------------------------------------------------

    /// Evaluates call arguments.
    pub fn eval_args(&mut self, args: &[Arg], env: &Rc<Env>) -> R<Args> {
        let mut out = Args::default();
        for a in args {
            match a {
                Arg::Pos(e) => out.pos.push(self.eval(e, env)?),
                Arg::Named(n, e) => {
                    let v = self.eval(e, env)?;
                    out.named.push((*n, v));
                }
                Arg::Spread(e) => {
                    let v = self.eval(e, env)?;
                    let mut it = self.iterate(&v, e.span)?;
                    while let Some(x) = it.next(self)? {
                        out.pos.push(x);
                    }
                }
                Arg::KwSpread(e) => {
                    let v = self.eval(e, env)?;
                    let Value::Map(m) = &v else {
                        return self.type_err("`**` needs a map", e.span);
                    };
                    for (k, v) in m.borrow().iter() {
                        let name = self.to_str(k)?;
                        out.named.push((sym(&name), v.clone()));
                    }
                }
            }
        }
        Ok(out)
    }

    fn eval_call(&mut self, callee: &Expr, args: &[Arg], env: &Rc<Env>, span: Span) -> R {
        match &callee.kind {
            ExprKind::Member {
                object,
                name,
                optional,
            } => {
                if matches!(object.kind, ExprKind::Super)
                    || matches!(&object.kind, ExprKind::Call { callee, args } if matches!(callee.kind, ExprKind::Super) && args.is_empty())
                {
                    let mut a = self.eval_args(args, env)?;
                    a.span = span;
                    return self.call_super(Some(*name), a, span);
                }
                let obj = self.eval(object, env)?;
                if *optional && matches!(obj, Value::Nil) {
                    return Ok(Value::Nil);
                }
                let mut a = self.eval_args(args, env)?;
                a.span = span;
                self.call_env = Some(env.clone());
                self.call_method_or_ufcs(&obj, *name, a, env, span)
            }
            ExprKind::Super => {
                let mut a = self.eval_args(args, env)?;
                a.span = span;
                self.call_super(None, a, span)
            }
            _ => {
                let f = self.eval_callee(callee, env)?;
                let mut a = self.eval_args(args, env)?;
                a.span = span;
                self.call_env = Some(env.clone());
                self.call_value(f, a)
            }
        }
    }

    /// Evaluates a callee without auto-calling bare functions.
    fn eval_callee(&mut self, callee: &Expr, env: &Rc<Env>) -> R {
        match &callee.kind {
            ExprKind::Name(n) => self.lookup(*n, env, callee.span),
            _ => self.eval(callee, env),
        }
    }

    /// Calls any callable value.
    pub fn call_value(&mut self, f: Value, args: Args) -> R {
        let span = args.span;
        match f {
            Value::Func(c) => self.call_closure(&c, args, None),
            Value::Native(n) => (n.f)(self, args),
            Value::Bound(b) => match &b.func {
                Value::Func(c) => self.call_closure(c, args, Some(b.receiver.clone())),
                Value::Native(n) => {
                    let mut args = args;
                    args.pos.insert(0, b.receiver.clone());
                    (n.f)(self, args)
                }
                other => {
                    let other = other.clone();
                    self.call_value(other, args)
                }
            },
            Value::Class(c) => self.instantiate(&c, args),
            Value::Type(t) => crate::builtins::convert(self, t, args),
            Value::Object(o) => {
                let this = Value::Object(o.clone());
                match o.call(self, args, &this) {
                    Some(r) => r,
                    None => {
                        let t = o.type_name().to_owned();
                        self.type_err(format!("`{t}` object is not callable"), span)
                    }
                }
            }
            Value::Instance(i) => {
                for name in ["__call__", "call", "invoke", "apply"] {
                    if let Some(m) = i.class.find_method(sym(name)) {
                        let recv = Value::Instance(i.clone());
                        return self.call_value(
                            Value::Bound(Rc::new(Bound {
                                receiver: recv,
                                func: m,
                            })),
                            args,
                        );
                    }
                }
                let n = i.class.name;
                self.type_err(format!("`{n}` object is not callable"), span)
            }
            Value::Map(m) => {
                // A map used as a function looks up its argument (Clojure).
                let key = args.get(0).cloned().unwrap_or(Value::Nil);
                Ok(m.borrow().get(&key).unwrap_or(Value::Nil))
            }
            Value::Tensor(t) => {
                let t = Value::Tensor(t);
                self.call_method(&t, sym("__call__"), args, span)
            }
            Value::Str(s) => self.call_symbol(&s, args),
            other => {
                let t = other.type_name();
                self.type_err(format!("`{t}` value is not callable"), span)
            }
        }
    }

    /// Ruby's symbol-to-proc: `xs.map(&:upcase)`, `xs.reduce(:+)`. An
    /// operator symbol applies the operator; a name calls that method on
    /// the first argument.
    fn call_symbol(&mut self, s: &str, mut args: Args) -> R {
        let span = args.span;
        let op = match s {
            "+" => Some(BinOp::Add),
            "-" => Some(BinOp::Sub),
            "*" => Some(BinOp::Mul),
            "/" => Some(BinOp::Div),
            "%" => Some(BinOp::Mod),
            "**" => Some(BinOp::Pow),
            "&" => Some(BinOp::BitAnd),
            "|" => Some(BinOp::BitOr),
            "^" => Some(BinOp::BitXor),
            "<<" => Some(BinOp::Shl),
            ">>" => Some(BinOp::Shr),
            _ => None,
        };
        if let Some(op) = op
            && args.pos.len() == 2
        {
            let b = args.pos.pop().unwrap_or(Value::Nil);
            let a = args.pos.pop().unwrap_or(Value::Nil);
            return self.binary_op(op, a, b, span);
        }
        if args.pos.len() == 2 {
            let b = args.pos.pop().unwrap_or(Value::Nil);
            let a = args.pos.pop().unwrap_or(Value::Nil);
            let cmp = match s {
                "<" => Some(CmpOp::Lt),
                ">" => Some(CmpOp::Gt),
                "<=" => Some(CmpOp::Le),
                ">=" => Some(CmpOp::Ge),
                "==" => Some(CmpOp::Eq),
                "!=" | "/=" => Some(CmpOp::Ne),
                _ => None,
            };
            if let Some(c) = cmp {
                return self.compare_op(c, &a, &b, span);
            }
            match s {
                "++" | "<>" => return self.binary_op(BinOp::Concat, a, b, span),
                "&&" => return Ok(if a.truthy() { b } else { a }),
                "||" => return Ok(if a.truthy() { a } else { b }),
                "!!" => return self.index_value(&a, &b, span),
                "." => {
                    // Composition: `(f . g) x == f (g x)`.
                    return Ok(Value::native("composed", move |interp, args| {
                        let inner = interp.call_value(b.clone(), args)?;
                        interp.call_value(a.clone(), Args::new(vec![inner], Span::default()))
                    }));
                }
                _ => {}
            }
            args.pos.push(a);
            args.pos.push(b);
        }
        if args.pos.is_empty() {
            return self.type_err(
                format!(
                    "the string {} is not callable",
                    format::repr_plain(&Value::str(s))
                ),
                span,
            );
        }
        let recv = args.pos.remove(0);
        self.call_method(&recv, sym(s), args, span)
    }

    /// Calls a user function, binding parameters.
    pub fn call_closure(&mut self, c: &Rc<Closure>, args: Args, receiver: Option<Value>) -> R {
        if c.def.is_generator() {
            return self.call_generator(c, args, receiver);
        }
        self.call_closure_with(c, args, receiver, None)
    }

    /// Calls a closure; `block` is the Ruby block that `yield` calls.
    pub(crate) fn call_closure_with(
        &mut self,
        c: &Rc<Closure>,
        args: Args,
        receiver: Option<Value>,
        block: Option<Value>,
    ) -> R {
        if self.depth >= self.max_depth {
            let span = args.span;
            return Err(self.error(
                "RecursionError",
                format!("maximum recursion depth ({}) exceeded", self.max_depth),
                span,
            ));
        }
        let def = &c.def;
        if def.name.is_some_and(|n| n.is("main")) {
            self.main_called = true;
        }
        let env = Env::child(&c.env, EnvKind::Function);
        let class = c.class.borrow().as_ref().and_then(std::rc::Weak::upgrade);
        if let Some(recv) = &receiver {
            env.define(sym("self"), recv.clone());
            env.define(sym("this"), recv.clone());
            if let Some(r) = def.receiver {
                env.define(r, recv.clone());
            }
        }
        if let Some(b) = block {
            env.define(sym("__block__"), b);
        }
        let span = args.span;
        let saved_file = self.cur_file;
        self.bind_params(def, args, &env)?;
        self.frames.push(Frame {
            name: c.name(),
            file: saved_file,
            span,
            class,
            receiver,
        });
        self.depth += 1;
        self.cur_file = def.file;
        self.defers.push(Vec::new());
        let result = self.eval_block(&def.body, &env);
        let deferred = self.defers.pop().unwrap_or_default();
        let result = self.run_defers(deferred, result);
        self.cur_file = saved_file;
        self.depth -= 1;
        self.frames.pop();
        let value = match result {
            Ok(v) | Err(Flow::Return(v)) => v,
            Err(Flow::Break) | Err(Flow::Continue) => Value::Nil,
            Err(e) => return Err(e),
        };
        if let Some(ret) = &def.ret {
            return self
                .coerce_declared(value, ret, def.span)
                .map_err(|e| match e {
                    Flow::Throw(mut t) => {
                        t.loc = Some((def.file, def.span));
                        Flow::Throw(t)
                    }
                    other => other,
                });
        }
        Ok(value)
    }

    pub(crate) fn bind_params(&mut self, def: &FnDef, args: Args, env: &Rc<Env>) -> R<()> {
        let span = args.span;
        let Args { pos, mut named, .. } = args;
        if def.implicit_params {
            // `{ it * 2 }` / Swift `$0`.
            env.define(sym("it"), pos.first().cloned().unwrap_or(Value::Nil));
            for (i, v) in pos.iter().enumerate() {
                env.define(sym(&format!("${i}")), v.clone());
            }
            return Ok(());
        }
        let mut pos = pos.into_iter();
        let fname = def
            .name
            .map_or_else(|| "<lambda>".into(), |n| n.as_str().to_string());
        for p in &def.params {
            match p.kind {
                ParamKind::Normal => {
                    let name = p.name();
                    let named_val = name.and_then(|n| {
                        named
                            .iter()
                            .position(|(k, _)| *k == n || Some(*k) == p.label)
                            .map(|i| named.remove(i).1)
                    });
                    let v = match (named_val, pos.next()) {
                        (Some(v), Some(extra)) => {
                            // Positional and named both supplied; keep the
                            // positional for this slot and fail loudly.
                            let _ = (v, extra);
                            return self.type_err(
                                format!(
                                    "`{fname}` got multiple values for `{}`",
                                    name.map(|n| n.as_str()).unwrap_or_default()
                                ),
                                span,
                            );
                        }
                        (Some(v), None) | (None, Some(v)) => v,
                        (None, None) => match &p.default {
                            Some(d) => self.eval(d, env)?,
                            None => {
                                let pname = name
                                    .map(|n| n.as_str().to_string())
                                    .unwrap_or_else(|| "?".into());
                                return self.type_err(
                                    format!("`{fname}` is missing the argument `{pname}`"),
                                    span,
                                );
                            }
                        },
                    };
                    let v = match &p.ty {
                        Some(t) => self.coerce_declared(v, t, span)?,
                        None => v,
                    };
                    self.bind_pattern(&p.pattern, v, env, span)?;
                }
                ParamKind::Rest => {
                    let rest: Vec<Value> = pos.by_ref().collect();
                    self.bind_pattern(&p.pattern, Value::list(rest), env, span)?;
                }
                ParamKind::KwRest => {
                    let mut d = Dict::new();
                    for (k, v) in named.drain(..) {
                        d.insert(Value::Str(k.as_str()), v);
                    }
                    self.bind_pattern(&p.pattern, Value::Map(Rc::new(RefCell::new(d))), env, span)?;
                }
            }
        }
        let extra = pos.count();
        if extra > 0 {
            let n = def.params.len();
            return self.type_err(
                format!(
                    "`{fname}` takes {n} argument{} but {} were given",
                    if n == 1 { "" } else { "s" },
                    n + extra
                ),
                span,
            );
        }
        if let Some((k, _)) = named.first() {
            return self.type_err(format!("`{fname}` has no parameter named `{k}`"), span);
        }
        Ok(())
    }

    // ----- assignment ------------------------------------------------------------

    /// Assigns to a target expression (name, member, index or destructuring).
    pub fn assign(&mut self, target: &Expr, value: Value, env: &Rc<Env>) -> R<()> {
        match &target.kind {
            ExprKind::Name(n) => {
                if !env.assign(*n, &value) && !self.assign_implicit_member(*n, &value) {
                    env.function_scope().define(*n, value);
                }
                Ok(())
            }
            ExprKind::Member { object, name, .. } => {
                let obj = self.eval(object, env)?;
                self.set_attr(&obj, *name, value, target.span)
            }
            ExprKind::Index { object, index } => {
                let obj = self.eval(object, env)?;
                let idx = match (&obj, &index.kind) {
                    // Ruby `h[:key] = v`.
                    (
                        Value::Map(_),
                        ExprKind::Slice {
                            start: None,
                            end: Some(end),
                            step: None,
                        },
                    ) if matches!(end.kind, ExprKind::Name(_)) => {
                        let ExprKind::Name(n) = &end.kind else {
                            unreachable!()
                        };
                        Value::Str(n.as_str())
                    }
                    _ => self.eval(index, env)?,
                };
                self.set_index(&obj, idx, value, target.span)
            }
            ExprKind::Tuple(items) | ExprKind::List(items) => {
                let mut it = self.iterate(&value, target.span)?;
                let mut values = Vec::new();
                while let Some(v) = it.next(self)? {
                    values.push(v);
                }
                let spread = items
                    .iter()
                    .position(|e| matches!(e.kind, ExprKind::Spread(_)));
                match spread {
                    None => {
                        if values.len() != items.len() {
                            return self.value_err(
                                format!(
                                    "cannot unpack {} values into {} targets",
                                    values.len(),
                                    items.len()
                                ),
                                target.span,
                            );
                        }
                        for (t, v) in items.iter().zip(values) {
                            self.assign(t, v, env)?;
                        }
                    }
                    Some(si) => {
                        let after = items.len() - si - 1;
                        if values.len() < si + after {
                            return self.value_err("not enough values to unpack", target.span);
                        }
                        let tail = values.split_off(values.len() - after);
                        let middle = values.split_off(si);
                        for (t, v) in items[..si].iter().zip(values) {
                            self.assign(t, v, env)?;
                        }
                        if let ExprKind::Spread(inner) = &items[si].kind {
                            self.assign(inner, Value::list(middle), env)?;
                        }
                        for (t, v) in items[si + 1..].iter().zip(tail) {
                            self.assign(t, v, env)?;
                        }
                    }
                }
                Ok(())
            }
            _ => self.type_err("cannot assign to this expression", target.span),
        }
    }

    /// Applies a declared type to a value: numeric conversions, then a check.
    pub fn coerce_declared(&mut self, v: Value, ty: &TypeExpr, span: Span) -> R {
        if let TypeExpr::Named(n, _) = ty {
            match TypeTag::from_name(&n.as_str()) {
                Some(TypeTag::Float) => {
                    if let Value::Int(i) = v {
                        return Ok(Value::Float(i as f64));
                    }
                }
                Some(TypeTag::Int) => {
                    if let Value::Float(f) = v
                        && f.is_finite()
                    {
                        return Ok(Value::Int(f.trunc() as i64));
                    }
                }
                _ => {}
            }
        }
        if self.type_matches(&v, ty)? {
            Ok(v)
        } else {
            let want = format::type_expr_name(ty);
            let got = v.type_name();
            self.type_err(
                format!("expected a value of type `{want}`, got `{got}`"),
                span,
            )
        }
    }

    /// Whether `v` satisfies a type annotation.
    pub fn type_matches(&mut self, v: &Value, ty: &TypeExpr) -> R<bool> {
        Ok(match ty {
            TypeExpr::Any => true,
            TypeExpr::Optional(inner) => matches!(v, Value::Nil) || self.type_matches(v, inner)?,
            TypeExpr::Union(alts) => {
                for a in alts {
                    if self.type_matches(v, a)? {
                        return Ok(true);
                    }
                }
                false
            }
            TypeExpr::List(_) => matches!(v, Value::List(_) | Value::Tuple(_) | Value::Range(_)),
            TypeExpr::Tuple(items) => match v {
                Value::Tuple(t) => t.len() == items.len(),
                _ => false,
            },
            TypeExpr::Named(n, _) => {
                let name = n.as_str();
                if let Some(tag) = TypeTag::from_name(&name) {
                    return Ok(self.type_tag_matches(v, tag));
                }
                match self
                    .globals
                    .get(*n)
                    .or_else(|| self.builtins.get(n).cloned())
                {
                    Some(Value::Class(c)) => match v {
                        Value::Instance(i) => i.class.is_subclass_of(&c),
                        _ => matches!(v, Value::Nil) && false,
                    },
                    Some(Value::Enum(e)) => {
                        matches!(v, Value::Variant(var) if Rc::ptr_eq(&var.ty, &e))
                    }
                    Some(Value::Type(t)) => self.type_tag_matches(v, t),
                    // Unknown/foreign type names (`size_t`, `Self`) are accepted.
                    _ => true,
                }
            }
        })
    }

    /// Whether `v` is of built-in type `t`.
    pub fn type_tag_matches(&self, v: &Value, t: TypeTag) -> bool {
        match t {
            TypeTag::Any => true,
            TypeTag::Nil => matches!(v, Value::Nil),
            TypeTag::Bool => matches!(v, Value::Bool(_)),
            TypeTag::Int => matches!(v, Value::Int(_) | Value::Big(_)),
            TypeTag::Float => matches!(v, Value::Float(_)),
            TypeTag::Number => v.is_number(),
            TypeTag::Str => matches!(v, Value::Str(_)),
            TypeTag::List => matches!(v, Value::List(_)),
            TypeTag::Tuple => matches!(v, Value::Tuple(_)),
            TypeTag::Map => matches!(v, Value::Map(_) | Value::Instance(_)),
            TypeTag::Set => matches!(v, Value::Set(_)),
            TypeTag::Range => matches!(v, Value::Range(_)),
            TypeTag::Func => v.is_callable(),
            TypeTag::Tensor => matches!(v, Value::Tensor(_)),
            TypeTag::Type => matches!(v, Value::Class(_) | Value::Type(_) | Value::Enum(_)),
        }
    }

    // ----- comprehensions ----------------------------------------------------------

    fn eval_comprehension(&mut self, c: &Comprehension, env: &Rc<Env>) -> R {
        let scope = Env::child(env, EnvKind::Block);
        let mut list = Vec::new();
        let mut dict = Dict::new();
        self.comp_clause(c, 0, &scope, &mut list, &mut dict)?;
        Ok(match c.kind {
            CompKind::List => Value::list(list),
            CompKind::Map => Value::Map(Rc::new(RefCell::new(dict))),
            CompKind::Set => {
                let mut d = Dict::new();
                for v in list {
                    d.insert(v, Value::Nil);
                }
                Value::Set(Rc::new(RefCell::new(d)))
            }
        })
    }

    fn comp_clause(
        &mut self,
        c: &Comprehension,
        i: usize,
        env: &Rc<Env>,
        list: &mut Vec<Value>,
        dict: &mut Dict,
    ) -> R<()> {
        let Some(clause) = c.clauses.get(i) else {
            let el = self.eval(&c.element, env)?;
            if c.kind == CompKind::Map {
                let v = match &c.value {
                    Some(v) => self.eval(v, env)?,
                    None => Value::Nil,
                };
                dict.insert(el, v);
            } else {
                list.push(el);
            }
            return Ok(());
        };
        match clause {
            CompClause::If(cond) => {
                if self.eval(cond, env)?.truthy() {
                    self.comp_clause(c, i + 1, env, list, dict)?;
                }
            }
            CompClause::For(pat, iter) => {
                let it = self.eval(iter, env)?;
                let mut items = self.iterate(&it, iter.span)?;
                while let Some(item) = items.next(self)? {
                    self.bind_pattern(pat, item, env, iter.span)?;
                    self.comp_clause(c, i + 1, env, list, dict)?;
                }
            }
        }
        Ok(())
    }

    /// Calls `f` on each frame for `super` resolution: the innermost frame
    /// that belongs to a class method.
    /// Java/C#/C++ methods see their class's members without `this.`:
    /// declared fields, static fields and other methods.
    fn implicit_member(&self, n: Sym) -> Option<Value> {
        let frame = self.frames.iter().rev().find(|f| f.class.is_some())?;
        let class = frame.class.clone()?;
        if let Some(Value::Instance(inst)) = &frame.receiver
            && inst.class.all_field_names().contains(&n)
            && let Some(v) = inst.get(n)
        {
            return Some(v);
        }
        let mut c = Some(class.clone());
        while let Some(cl) = c {
            if let Some(v) = cl.attrs.borrow().get(&n) {
                return Some(v.clone());
            }
            c = cl.superclass.clone();
        }
        if let Some(recv @ Value::Instance(_)) = &frame.receiver
            && let Some(m) = class.find_method(n)
        {
            return Some(Value::Bound(Rc::new(Bound {
                receiver: recv.clone(),
                func: m,
            })));
        }
        class.find_static(n)
    }

    /// Assignment counterpart of [`Interp::implicit_member`].
    fn assign_implicit_member(&self, n: Sym, value: &Value) -> bool {
        let Some(frame) = self.frames.iter().rev().find(|f| f.class.is_some()) else {
            return false;
        };
        if let Some(Value::Instance(inst)) = &frame.receiver
            && inst.class.all_field_names().contains(&n)
        {
            inst.set(n, value.clone());
            return true;
        }
        let mut c = frame.class.clone();
        while let Some(cl) = c {
            if cl.attrs.borrow().contains_key(&n) {
                cl.attrs.borrow_mut().insert(n, value.clone());
                return true;
            }
            c = cl.superclass.clone();
        }
        false
    }

    pub(crate) fn current_method(&self) -> Option<(Rc<Class>, Value)> {
        self.frames
            .iter()
            .rev()
            .find_map(|f| Some((f.class.clone()?, f.receiver.clone()?)))
    }

    /// Current call depth.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Runs pending top-level defers (used when a REPL session ends).
    pub fn flush_defers(&mut self) -> R<()> {
        let deferred = std::mem::take(self.defers.last_mut().unwrap_or(&mut Vec::new()));
        self.run_defers(deferred, Ok(Value::Nil)).map(|_| ())
    }
}

impl Default for Interp {
    fn default() -> Self {
        Interp::new()
    }
}
