//! Generators: functions containing `yield` run lazily.
//!
//! Calling a generator function binds its arguments and returns a
//! [`Generator`] without running any code. Each `next()` resumes the body
//! until the next `yield`. The interpreter is a recursive tree walker, so a
//! paused generator can't keep a Rust stack; instead the body is compiled
//! into a small tree of `GStmt`s and run by a stepper that keeps its
//! position in an explicit stack of `Frame`s.
//!
//! Statements that contain no `yield` run atomically on the main
//! interpreter. `yield` may appear as a statement, as an assignment or
//! declaration value (`x = yield v`, which receives `send(v)`), and inside
//! `if`, `while`/`until`, `for`, `loop`, `repeat`, `match` arms and nested
//! blocks.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use chimera_core::Span;

use super::iter::ValueIter;
use super::{Args, Env, EnvKind, Flow, Frame as CallFrame, Interp, sym};
use crate::ast::*;
use crate::value::*;

type GBlock = Rc<Vec<GStmt>>;

/// A statement of a generator body.
#[derive(Debug)]
enum GStmt {
    /// No `yield` inside: run on the main interpreter.
    Atomic(Stmt),
    /// `yield …`, possibly receiving a sent value.
    Yield {
        values: Vec<Expr>,
        delegate: bool,
        target: Option<Target>,
        span: Span,
    },
    If {
        cond: Expr,
        then: GBlock,
        otherwise: Option<GBlock>,
    },
    While {
        cond: Expr,
        body: GBlock,
        negate: bool,
        post: bool,
    },
    For {
        pattern: Pattern,
        iter: Expr,
        body: GBlock,
    },
    /// C-style `for (init; cond; step)`.
    CFor {
        init: Option<Stmt>,
        cond: Option<Expr>,
        step: Option<Expr>,
        body: GBlock,
    },
    Loop(GBlock),
    Repeat {
        count: Expr,
        body: GBlock,
    },
    Match {
        subject: Option<Expr>,
        arms: Vec<(Pattern, Option<Expr>, GBlock)>,
        span: Span,
    },
    Block(GBlock),
}

/// Where a sent value goes.
#[derive(Debug, Clone)]
enum Target {
    Assign(Expr),
    Let(Pattern),
}

/// Compiles a function body for stepping. Fails with a message when a
/// `yield` appears somewhere the stepper can't pause.
fn compile(def: &FnDef, block: &Block) -> Result<GBlock, (String, Span)> {
    let mut out = Vec::with_capacity(block.stmts.len());
    for s in &block.stmts {
        out.push(compile_stmt(def, s)?);
    }
    Ok(Rc::new(out))
}

fn compile_stmt(def: &FnDef, s: &Stmt) -> Result<GStmt, (String, Span)> {
    if !def.yields_in(s.span) {
        return Ok(GStmt::Atomic(s.clone()));
    }
    let unsupported = |what: &str| {
        Err((
            format!("`yield` inside {what} is not supported in generators yet"),
            s.span,
        ))
    };
    match &s.kind {
        StmtKind::Expr(e) => compile_expr_stmt(def, e, s.span),
        StmtKind::Let {
            pattern,
            value: Some(v),
            ..
        } => match &v.kind {
            ExprKind::Yield { values, delegate } => Ok(GStmt::Yield {
                values: values.clone(),
                delegate: *delegate,
                target: Some(Target::Let(pattern.clone())),
                span: v.span,
            }),
            _ => unsupported("this declaration"),
        },
        StmtKind::While {
            cond,
            body,
            negate,
            post,
        } => {
            if def.yields_in(cond.span) {
                return unsupported("a loop condition");
            }
            Ok(GStmt::While {
                cond: cond.clone(),
                body: compile(def, body)?,
                negate: *negate,
                post: *post,
            })
        }
        StmtKind::For {
            pattern,
            iter,
            body,
            ..
        } => {
            if def.yields_in(iter.span) {
                return unsupported("a loop header");
            }
            Ok(GStmt::For {
                pattern: pattern.clone(),
                iter: iter.clone(),
                body: compile(def, body)?,
            })
        }
        StmtKind::Loop(body) => Ok(GStmt::Loop(compile(def, body)?)),
        StmtKind::Repeat { count, body } => Ok(GStmt::Repeat {
            count: count.clone(),
            body: compile(def, body)?,
        }),
        StmtKind::Try { .. } => unsupported("`try`"),
        StmtKind::Switch { .. } => unsupported("`switch`"),
        StmtKind::CFor {
            init,
            cond,
            step,
            body,
        } => {
            if cond.as_ref().is_some_and(|c| def.yields_in(c.span))
                || step.as_ref().is_some_and(|c| def.yields_in(c.span))
            {
                return unsupported("a loop header");
            }
            Ok(GStmt::CFor {
                init: init.as_deref().cloned(),
                cond: cond.clone(),
                step: step.clone(),
                body: compile(def, body)?,
            })
        }
        _ => unsupported("this statement"),
    }
}

fn compile_expr_stmt(def: &FnDef, e: &Expr, span: Span) -> Result<GStmt, (String, Span)> {
    match &e.kind {
        ExprKind::Yield { values, delegate } => Ok(GStmt::Yield {
            values: values.clone(),
            delegate: *delegate,
            target: None,
            span: e.span,
        }),
        ExprKind::Assign { target, value } if matches!(value.kind, ExprKind::Yield { .. }) => {
            let ExprKind::Yield { values, delegate } = &value.kind else {
                unreachable!()
            };
            Ok(GStmt::Yield {
                values: values.clone(),
                delegate: *delegate,
                target: Some(Target::Assign((**target).clone())),
                span: value.span,
            })
        }
        ExprKind::If {
            cond,
            then,
            otherwise,
        } => {
            if def.yields_in(cond.span) {
                return Err((
                    "`yield` inside an `if` condition is not supported in generators yet".into(),
                    cond.span,
                ));
            }
            Ok(GStmt::If {
                cond: (**cond).clone(),
                then: compile(def, then)?,
                otherwise: otherwise.as_ref().map(|b| compile(def, b)).transpose()?,
            })
        }
        ExprKind::Block(b) => Ok(GStmt::Block(compile(def, b)?)),
        ExprKind::Match { subject, arms } => {
            if let Some(s) = subject
                && def.yields_in(s.span)
            {
                return Err((
                    "`yield` inside a `match` subject is not supported in generators yet".into(),
                    s.span,
                ));
            }
            let mut out = Vec::with_capacity(arms.len());
            for arm in arms {
                out.push((
                    arm.pattern.clone(),
                    arm.guard.clone(),
                    compile(def, &arm.body)?,
                ));
            }
            Ok(GStmt::Match {
                subject: subject.as_ref().map(|s| (**s).clone()),
                arms: out,
                span: e.span,
            })
        }
        _ => Err((
            "`yield` must be a statement or an assignment value (`x = yield v`) in generators"
                .into(),
            span,
        )),
    }
}

/// A position in the generator body.
#[derive(Debug)]
enum Frame {
    Block {
        block: GBlock,
        idx: usize,
    },
    While {
        cond: Expr,
        body: GBlock,
        negate: bool,
        skip_test: bool,
    },
    For {
        pattern: Pattern,
        iter: ValueIter,
        body: GBlock,
        span: Span,
    },
    Loop {
        body: GBlock,
    },
    CFor {
        cond: Option<Expr>,
        step: Option<Expr>,
        body: GBlock,
        started: bool,
    },
    Repeat {
        remaining: i64,
        body: GBlock,
    },
    Delegate {
        iter: ValueIter,
    },
}

impl Frame {
    fn is_loop(&self) -> bool {
        matches!(
            self,
            Frame::While { .. }
                | Frame::For { .. }
                | Frame::Loop { .. }
                | Frame::CFor { .. }
                | Frame::Repeat { .. }
        )
    }
}

#[derive(Debug)]
struct State {
    env: Rc<Env>,
    frames: Vec<Frame>,
    pending: Option<Target>,
    done: bool,
    running: bool,
    /// The value of `return v` (Python's `StopIteration.value`).
    result: Value,
}

/// A paused generator.
#[derive(Debug)]
pub struct Generator {
    name: Rc<str>,
    def: Rc<FnDef>,
    receiver: Option<Value>,
    /// Where the generator was created, for tracebacks raised while it is
    /// resumed from somewhere without a span of its own (`list(gen)`).
    origin: (FileId, Span),
    state: RefCell<State>,
}

impl Generator {
    /// Creates a generator for `def` whose parameters are bound in `env`.
    pub fn start(
        interp: &mut Interp,
        def: &Rc<FnDef>,
        name: Rc<str>,
        env: Rc<Env>,
        receiver: Option<Value>,
        span: Span,
    ) -> R {
        let body = match compile(def, &def.body) {
            Ok(b) => b,
            Err((msg, at)) => {
                let saved = interp.cur_file;
                interp.cur_file = def.file;
                let e = interp.error("SyntaxError", msg, at);
                interp.cur_file = saved;
                return Err(e);
            }
        };
        Ok(Value::Object(Rc::new(Generator {
            name,
            def: def.clone(),
            receiver,
            origin: (interp.cur_file, span),
            state: RefCell::new(State {
                env,
                frames: vec![Frame::Block {
                    block: body,
                    idx: 0,
                }],
                pending: None,
                done: false,
                running: false,
                result: Value::Nil,
            }),
        })))
    }

    /// Whether the generator has finished.
    pub fn is_done(&self) -> bool {
        self.state.borrow().done
    }

    /// Resumes until the next yielded value (`None` when finished).
    pub fn resume(&self, interp: &mut Interp, sent: Value, span: Span) -> R<Option<Value>> {
        {
            let st = self.state.borrow();
            if st.done {
                return Ok(None);
            }
            if st.running {
                drop(st);
                return interp.value_err(
                    format!("generator `{}` is already running", self.name),
                    span,
                );
            }
        }
        self.state.borrow_mut().running = true;
        let (file, span) = if span.is_empty() {
            self.origin
        } else {
            (interp.cur_file, span)
        };
        interp.frames.push(CallFrame {
            name: self.name.clone(),
            file,
            span,
            class: None,
            receiver: self.receiver.clone(),
        });
        let saved_file = interp.cur_file;
        interp.cur_file = self.def.file;
        interp.depth += 1;
        let result = self.step(interp, sent);
        interp.depth -= 1;
        interp.cur_file = saved_file;
        interp.frames.pop();
        let mut st = self.state.borrow_mut();
        st.running = false;
        if !matches!(result, Ok(Some(_))) {
            st.done = true;
            st.frames.clear();
        }
        result
    }

    fn env(&self) -> Rc<Env> {
        self.state.borrow().env.clone()
    }

    fn step(&self, interp: &mut Interp, sent: Value) -> R<Option<Value>> {
        let env = self.env();
        let pending = self.state.borrow_mut().pending.take();
        if let Some(t) = pending {
            assign_target(interp, &t, sent, &env)?;
        }
        loop {
            // Take the top frame out so the interpreter can run freely.
            let Some(mut frame) = self.state.borrow_mut().frames.pop() else {
                return Ok(None);
            };
            match &mut frame {
                Frame::Block { block, idx } => {
                    if *idx >= block.len() {
                        continue;
                    }
                    let block = block.clone();
                    let i = *idx;
                    *idx += 1;
                    self.push(frame);
                    match &block[i] {
                        GStmt::Atomic(s) => match interp.exec(s, &env, true) {
                            Ok(_) => {}
                            Err(Flow::Break) => self.unwind(true),
                            Err(Flow::Continue) => self.unwind(false),
                            Err(Flow::Return(v)) => {
                                self.state.borrow_mut().result = v;
                                return Ok(None);
                            }
                            Err(e) => return Err(e),
                        },
                        GStmt::Yield {
                            values,
                            delegate,
                            target,
                            span,
                        } => {
                            let mut vals = Vec::with_capacity(values.len());
                            for v in values {
                                vals.push(interp.eval(v, &env)?);
                            }
                            let value = match vals.len() {
                                0 => Value::Nil,
                                1 => vals.pop().unwrap_or(Value::Nil),
                                _ => Value::tuple(vals),
                            };
                            if *delegate {
                                let iter = interp.iterate(&value, *span)?;
                                self.push(Frame::Delegate { iter });
                                continue;
                            }
                            self.state.borrow_mut().pending = target.clone();
                            return Ok(Some(value));
                        }
                        GStmt::If {
                            cond,
                            then,
                            otherwise,
                        } => {
                            if interp.eval(cond, &env)?.truthy() {
                                self.push(Frame::Block {
                                    block: then.clone(),
                                    idx: 0,
                                });
                            } else if let Some(o) = otherwise {
                                self.push(Frame::Block {
                                    block: o.clone(),
                                    idx: 0,
                                });
                            }
                        }
                        GStmt::While {
                            cond,
                            body,
                            negate,
                            post,
                        } => self.push(Frame::While {
                            cond: cond.clone(),
                            body: body.clone(),
                            negate: *negate,
                            skip_test: *post,
                        }),
                        GStmt::For {
                            pattern,
                            iter,
                            body,
                        } => {
                            let mut it = interp.eval(iter, &env)?;
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
                            let iter_state = interp.iterate(&it, iter.span)?;
                            self.push(Frame::For {
                                pattern: pattern.clone(),
                                iter: iter_state,
                                body: body.clone(),
                                span: iter.span,
                            });
                        }
                        GStmt::Loop(body) => self.push(Frame::Loop { body: body.clone() }),
                        GStmt::CFor {
                            init,
                            cond,
                            step,
                            body,
                        } => {
                            if let Some(s) = init {
                                interp.exec(s, &env, true)?;
                            }
                            self.push(Frame::CFor {
                                cond: cond.clone(),
                                step: step.clone(),
                                body: body.clone(),
                                started: false,
                            });
                        }
                        GStmt::Repeat { count, body } => {
                            let n = interp.eval(count, &env)?.as_int().unwrap_or(0);
                            self.push(Frame::Repeat {
                                remaining: n,
                                body: body.clone(),
                            });
                        }
                        GStmt::Match {
                            subject,
                            arms,
                            span,
                        } => {
                            let value = match subject {
                                Some(s) => interp.eval(s, &env)?,
                                None => Value::Bool(true),
                            };
                            let mut taken = false;
                            for (pattern, guard, body) in arms {
                                if interp.match_into(pattern, &value, &env, *span)? {
                                    if let Some(g) = guard
                                        && !interp.eval(g, &env)?.truthy()
                                    {
                                        continue;
                                    }
                                    self.push(Frame::Block {
                                        block: body.clone(),
                                        idx: 0,
                                    });
                                    taken = true;
                                    break;
                                }
                            }
                            let _ = taken;
                        }
                        GStmt::Block(b) => self.push(Frame::Block {
                            block: b.clone(),
                            idx: 0,
                        }),
                    }
                }
                Frame::While {
                    cond,
                    body,
                    negate,
                    skip_test,
                } => {
                    let go = if *skip_test {
                        *skip_test = false;
                        true
                    } else {
                        interp.eval(cond, &env)?.truthy() != *negate
                    };
                    if go {
                        let body = body.clone();
                        self.push(frame);
                        self.push(Frame::Block {
                            block: body,
                            idx: 0,
                        });
                    }
                }
                Frame::For {
                    pattern,
                    iter,
                    body,
                    span,
                } => {
                    if let Some(item) = iter.next(interp)? {
                        interp.bind_pattern(pattern, item, &env, *span)?;
                        let body = body.clone();
                        self.push(frame);
                        self.push(Frame::Block {
                            block: body,
                            idx: 0,
                        });
                    }
                }
                Frame::Loop { body } => {
                    let body = body.clone();
                    self.push(frame);
                    self.push(Frame::Block {
                        block: body,
                        idx: 0,
                    });
                }
                Frame::CFor {
                    cond,
                    step,
                    body,
                    started,
                } => {
                    if *started && let Some(st) = step {
                        interp.eval(st, &env)?;
                    }
                    *started = true;
                    let go = match cond {
                        Some(c) => interp.eval(c, &env)?.truthy(),
                        None => true,
                    };
                    if go {
                        let body = body.clone();
                        self.push(frame);
                        self.push(Frame::Block {
                            block: body,
                            idx: 0,
                        });
                    }
                }
                Frame::Repeat { remaining, body } => {
                    if *remaining > 0 {
                        *remaining -= 1;
                        let body = body.clone();
                        self.push(frame);
                        self.push(Frame::Block {
                            block: body,
                            idx: 0,
                        });
                    }
                }
                Frame::Delegate { iter } => {
                    if let Some(v) = iter.next(interp)? {
                        self.push(frame);
                        return Ok(Some(v));
                    }
                }
            }
        }
    }

    fn push(&self, frame: Frame) {
        self.state.borrow_mut().frames.push(frame);
    }

    /// `break` (pop the innermost loop) or `continue` (resume it).
    fn unwind(&self, is_break: bool) {
        let mut st = self.state.borrow_mut();
        while let Some(top) = st.frames.last() {
            if top.is_loop() {
                if is_break {
                    st.frames.pop();
                }
                return;
            }
            st.frames.pop();
        }
    }
}

fn assign_target(interp: &mut Interp, t: &Target, value: Value, env: &Rc<Env>) -> R<()> {
    match t {
        Target::Assign(e) => interp.assign(e, value, env),
        Target::Let(p) => interp.bind_pattern(p, value, env, Span::default()),
    }
}

impl Object for Generator {
    fn type_name(&self) -> &str {
        "generator"
    }

    fn get_attr(&self, _interp: &mut Interp, name: &str) -> Option<R> {
        match name {
            "done" | "finished" | "gi_running" => Some(Ok(Value::Bool(self.is_done()))),
            "__name__" | "name" => Some(Ok(Value::str(&self.name))),
            _ => None,
        }
    }

    fn call_method(&self, interp: &mut Interp, name: &str, args: Args, this: &Value) -> Option<R> {
        let span = args.span;
        Some(match name {
            "__iter__" | "iter" | "each" if args.pos.is_empty() => Ok(this.clone()),
            "__next__" | "next" | "Next" | "resume" => {
                match self.resume(interp, Value::Nil, span) {
                    Ok(Some(v)) => Ok(v),
                    Ok(None) => match args.get(0) {
                        Some(default) => Ok(default.clone()),
                        None => Err(stop_iteration(
                            interp,
                            &self.state.borrow().result.clone(),
                            span,
                        )),
                    },
                    Err(e) => Err(e),
                }
            }
            "send" => {
                let v = args.get(0).cloned().unwrap_or(Value::Nil);
                match self.resume(interp, v, span) {
                    Ok(Some(v)) => Ok(v),
                    Ok(None) => Err(stop_iteration(
                        interp,
                        &self.state.borrow().result.clone(),
                        span,
                    )),
                    Err(e) => Err(e),
                }
            }
            "close" | "return" => {
                let mut st = self.state.borrow_mut();
                st.done = true;
                st.frames.clear();
                Ok(Value::Nil)
            }
            "take" | "first" => {
                let n = args.get(0).and_then(Value::as_int).unwrap_or(1).max(0) as usize;
                let mut out = Vec::with_capacity(n);
                while out.len() < n {
                    match self.resume(interp, Value::Nil, span) {
                        Ok(Some(v)) => out.push(v),
                        Ok(None) => break,
                        Err(e) => return Some(Err(e)),
                    }
                }
                if name == "first" && args.pos.is_empty() {
                    Ok(out.pop().unwrap_or(Value::Nil))
                } else {
                    Ok(Value::list(out))
                }
            }
            "each" => {
                let f = args.pos[0].clone();
                loop {
                    match self.resume(interp, Value::Nil, span) {
                        Ok(Some(v)) => {
                            if let Err(e) = interp.call_value(f.clone(), Args::new(vec![v], span)) {
                                return Some(Err(e));
                            }
                        }
                        Ok(None) => break Ok(Value::Nil),
                        Err(e) => break Err(e),
                    }
                }
            }
            _ => {
                // Everything else (map, filter, to_list, …) works on the
                // remaining items.
                let mut items = Vec::new();
                loop {
                    match self.resume(interp, Value::Nil, span) {
                        Ok(Some(v)) => items.push(v),
                        Ok(None) => break,
                        Err(e) => return Some(Err(e)),
                    }
                }
                let list = Value::list(items);
                match name {
                    "to_list" | "to_a" | "toList" | "collect" | "list" => Ok(list),
                    _ => interp.call_method(&list, sym(name), args, span),
                }
            }
        })
    }

    fn display(&self) -> String {
        format!("<generator {}>", self.name)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn stop_iteration(interp: &mut Interp, result: &Value, span: Span) -> Flow {
    let f = interp.error("StopIteration", "generator exhausted", span);
    if let Flow::Throw(t) = &f
        && let Value::Instance(i) = &t.value
    {
        i.set(sym("value"), result.clone());
    }
    f
}

impl Interp {
    /// Calls a function containing `yield`: with a trailing block (Ruby)
    /// the body runs now and `yield` calls the block; otherwise a paused
    /// generator is returned.
    pub(crate) fn call_generator(
        &mut self,
        c: &Rc<Closure>,
        mut args: Args,
        receiver: Option<Value>,
    ) -> R {
        let normal = c
            .def
            .params
            .iter()
            .filter(|p| p.kind == ParamKind::Normal)
            .count();
        let block_mode = args.pos.len() > normal && args.pos.last().is_some_and(Value::is_callable)
            || args.named.iter().any(|(k, _)| k.is("block"));
        if block_mode {
            let block = match args.named.iter().position(|(k, _)| k.is("block")) {
                Some(i) => args.named.remove(i).1,
                None => args.pos.pop().unwrap_or(Value::Nil),
            };
            return self.call_closure_with(c, args, receiver, Some(block));
        }
        let env = Env::child(&c.env, EnvKind::Function);
        if let Some(recv) = &receiver {
            env.define(sym("self"), recv.clone());
            env.define(sym("this"), recv.clone());
            if let Some(r) = c.def.receiver {
                env.define(r, recv.clone());
            }
        }
        let span = args.span;
        self.bind_params(&c.def, args, &env)?;
        Generator::start(self, &c.def, c.name(), env, receiver, span)
    }
}

/// `iter(xs)`: a stateful iterator over any iterable.
#[derive(Debug)]
pub struct IterObject {
    iter: RefCell<ValueIter>,
}

impl IterObject {
    /// Wraps an iteration in progress.
    pub fn new(iter: ValueIter) -> IterObject {
        IterObject {
            iter: RefCell::new(iter),
        }
    }

    /// The next item, if any.
    pub fn advance(&self, interp: &mut Interp) -> R<Option<Value>> {
        // Take the iterator out while it runs: the next item may come from
        // user code that touches this object again.
        let mut it = std::mem::replace(
            &mut *self.iter.borrow_mut(),
            ValueIter::Items(Vec::new().into_iter()),
        );
        let r = it.next(interp);
        *self.iter.borrow_mut() = it;
        r
    }
}

impl Object for IterObject {
    fn type_name(&self) -> &str {
        "iterator"
    }

    fn call_method(&self, interp: &mut Interp, name: &str, args: Args, this: &Value) -> Option<R> {
        let span = args.span;
        Some(match name {
            "__iter__" | "iter" => Ok(this.clone()),
            "__next__" | "next" | "Next" => match self.advance(interp) {
                Ok(Some(v)) => Ok(v),
                Ok(None) => match args.get(0) {
                    Some(d) => Ok(d.clone()),
                    None => Err(interp.error("StopIteration", "iterator exhausted", span)),
                },
                Err(e) => Err(e),
            },
            "hasNext" | "has_next" => {
                // Peek by advancing and re-queuing the item.
                match self.advance(interp) {
                    Ok(Some(v)) => {
                        let rest = std::mem::replace(
                            &mut *self.iter.borrow_mut(),
                            ValueIter::Items(Vec::new().into_iter()),
                        );
                        let mut items = vec![v];
                        let mut rest = rest;
                        while let Ok(Some(x)) = rest.next(interp) {
                            items.push(x);
                        }
                        *self.iter.borrow_mut() = ValueIter::Items(items.into_iter());
                        Ok(Value::Bool(true))
                    }
                    Ok(None) => Ok(Value::Bool(false)),
                    Err(e) => Err(e),
                }
            }
            _ => return None,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `next(it[, default])` for generators, `iter()` objects and iterator
/// instances.
pub fn next_value(interp: &mut Interp, it: &Value, default: Option<Value>, span: Span) -> R {
    let item = match it {
        Value::Object(o) => {
            let o = o.clone();
            if let Some(g) = o.as_any().downcast_ref::<Generator>() {
                g.resume(interp, Value::Nil, span)?
            } else if let Some(i) = o.as_any().downcast_ref::<IterObject>() {
                i.advance(interp)?
            } else {
                return interp.call_method(
                    it,
                    sym("__next__"),
                    Args::new(default.into_iter().collect(), span),
                    span,
                );
            }
        }
        Value::Instance(_) => {
            return match interp.try_call_method(
                it,
                sym("__next__"),
                Args::new(Vec::new(), span),
                span,
            ) {
                Ok(Some(v)) => Ok(v),
                Ok(None) => interp.call_method(it, sym("next"), Args::new(Vec::new(), span), span),
                Err(Flow::Throw(t))
                    if default.is_some()
                        && matches!(&t.value, Value::Instance(i) if i.class.has_ancestor_named("StopIteration")) =>
                {
                    Ok(default.unwrap_or(Value::Nil))
                }
                Err(e) => Err(e),
            };
        }
        other => {
            let t = other.type_name();
            return interp.type_err(format!("`{t}` is not an iterator (use iter() first)"), span);
        }
    };
    match (item, default) {
        (Some(v), _) => Ok(v),
        (None, Some(d)) => Ok(d),
        (None, None) => Err(interp.error("StopIteration", "iterator exhausted", span)),
    }
}
