//! Statements: declarations, control flow, classes, imports and the English
//! statement forms.

use std::rc::Rc;

use chimera_core::Span;

use super::pattern::{is_type_word, looks_capitalized};
use super::{Header, PResult, Parser, describe, simple_param, terms};
use crate::ast::*;
use crate::lexer::Tok;
use crate::symbol::Sym;

/// Words that may precede a declaration and are otherwise ignored.
const MODIFIERS: &[&str] = &[
    "pub",
    "public",
    "private",
    "protected",
    "internal",
    "export",
    "static",
    "final",
    "async",
    "inline",
    "virtual",
    "override",
    "abstract",
    "extern",
    "unsafe",
    "open",
    "sealed",
    "readonly",
    "synchronized",
    "transient",
    "volatile",
    "partial",
    "native",
    "lazy",
    "suspend",
    "mutating",
    "dynamic",
    "constexpr",
    "register",
    "signed",
    "unsigned",
];

/// Keywords that introduce a function definition.
pub(crate) const FN_WORDS: &[&str] = &[
    "fn",
    "func",
    "function",
    "def",
    "fun",
    "sub",
    "proc",
    "defn",
    "method",
    "procedure",
    "subroutine",
];

/// Keywords that introduce a variable declaration.
const DECL_WORDS: &[&str] = &[
    "var", "const", "val", "my", "local", "auto", "our", "dim", "Dim", "DIM", "declare", "mut",
];

/// Names that make a method a constructor.
const CTOR_NAMES: &[&str] = &[
    "__init__",
    "init",
    "constructor",
    "initialize",
    "initializer",
];

impl Parser<'_> {
    pub(crate) fn statement_inner(&mut self) -> PResult<Stmt> {
        let start = self.peek().span;
        let h = self.header();

        if matches!(self.peek().tok, Tok::IVar(_)) && self.looks_like_decorator() {
            return self.decorated(h);
        }

        if let Some(w) = self.peek().ident().map(str::to_owned) {
            let next = self.nth(1).clone();
            let next_ident = next.ident().is_some() && next.line == self.peek().line;
            match w.as_str() {
                m if MODIFIERS.contains(&m)
                    && next_ident
                    && !self.nth_is_op(2, "=")
                    && !next.is_word("in") =>
                {
                    self.bump();
                    let mut s = self.statement()?;
                    if m == "static"
                        && let StmtKind::Fn(def) = &mut s.kind
                    {
                        Rc::make_mut(def).is_static = true;
                    }
                    return Ok(s);
                }
                f if FN_WORDS.contains(&f) => {
                    let lambda_form =
                        matches!(f, "fun" | "proc" | "fn" | "function") && self.nth_is_op(2, "->");
                    if next_ident && !lambda_form {
                        return self.fn_decl(h, Vec::new(), false);
                    }
                    if f == "func"
                        && next.is_op("(")
                        && let Some(s) = self.attempt(|p| p.go_method(h))
                    {
                        return Ok(s);
                    }
                }
                "let" => return self.let_stmt(),
                d if DECL_WORDS.contains(&d)
                    && (next_ident || next.is_op("[") || next.is_op("{") || next.is_op("(")) =>
                {
                    if d == "local" && (next.is_word("function") || next.is_word("fn")) {
                        self.bump();
                        return self.fn_decl(h, Vec::new(), false);
                    }
                    if d == "const" && next.is_word("fn") {
                        self.bump();
                        return self.fn_decl(h, Vec::new(), false);
                    }
                    return self.decl_stmt();
                }
                "class" | "struct" | "record" | "interface" | "trait" | "protocol" | "object"
                | "module"
                    // `struct Point p = {1, 2};` is a C declaration.
                    if next_ident && !(w == "struct" && self.nth(2).ident().is_some() && self.nth(2).line == next.line) =>
                {
                    return self.class_decl(h);
                }
                "data" if next.is_word("class") => return self.class_decl(h),
                "case" if next.is_word("class") => return self.class_decl(h),
                "data" if next_ident && looks_capitalized(next.ident().unwrap_or("")) => {
                    return self.adt_decl(h);
                }
                "enum" if next_ident => return self.enum_decl(h),
                "type" if next_ident => return self.type_decl(h),
                "impl" | "extension" if next_ident || next.is_op("<") => return self.impl_decl(h),
                "if" | "unless" => {
                    let e = self.if_expr()?;
                    return self.expr_stmt_from(e);
                }
                "while" | "until" => return self.while_stmt(h),
                "guard" if !next.is_op("=") && !next.is_op("(") || next.space_before => return self.guard_stmt(h),
                "loop" | "forever"
                    if next.is_op("{")
                        || next.is_op(":")
                        || next.is_word("do")
                        || matches!(next.tok, Tok::Newline) =>
                {
                    return self.loop_stmt(h);
                }
                "repeat" => return self.repeat_stmt(h),
                "for" | "foreach" => return self.for_stmt(h),
                "do" => return self.do_stmt(h),
                "return" => return self.return_stmt(),
                "break" | "last" if self.ends_simple_stmt(1) => {
                    self.bump();
                    return self.simple_stmt(StmtKind::Break, start);
                }
                "continue" | "next" if self.ends_simple_stmt(1) => {
                    self.bump();
                    return self.simple_stmt(StmtKind::Continue, start);
                }
                "throw" | "raise" => return self.throw_stmt(),
                "try" => return self.try_stmt(h),
                "begin" => return self.begin_stmt(h),
                "switch" => return self.switch_stmt(h),
                "import" | "use" | "require" | "require_relative" | "include" | "using"
                | "package" | "extern"
                    if !(next.is_op("(") && !next.space_before) && !next.is_op("=") =>
                {
                    return self.import_stmt();
                }
                "from" if next_ident && self.from_import_ahead() => return self.import_stmt(),
                "module" if next.is_word("where") => {
                    self.skip_line();
                    return Ok(self.make_stmt(StmtKind::Pass, start));
                }
                "with" if !(next.is_op("(") && !next.space_before) && next.line == self.peek().line => {
                    return self.with_stmt(h);
                }
                "defer" => {
                    self.bump();
                    let s = self.statement()?;
                    let span = self.span_from(start);
                    return Ok(Stmt {
                        kind: StmtKind::Defer(Block::new(vec![s])),
                        span,
                    });
                }
                "pass" if self.ends_simple_stmt(1) => {
                    self.bump();
                    return self.simple_stmt(StmtKind::Pass, start);
                }
                "del" | "delete" | "DELETE"
                    if !next.ident().is_some_and(|w| w.eq_ignore_ascii_case("from")) && !next.is_op("(") && next_ident =>
                {
                    return self.delete_stmt();
                }
                "global" | "nonlocal" if next_ident => {
                    self.skip_line();
                    return Ok(self.make_stmt(StmtKind::Pass, start));
                }
                "model" | "network" if next_ident && self.model_ahead() => return self.model_decl(h),
                "train" => {
                    if let Some(s) = self.attempt(|p| p.no_jux(|p| p.train_stmt())) {
                        return Ok(s);
                    }
                }
                "assert" if !next.is_op("(") || next.space_before => return self.assert_stmt(),
                "add" | "subtract" | "multiply" | "divide" | "increase" | "decrease"
                | "increment" | "decrement" | "set" | "swap" | "change" => {
                    if let Some(s) = self.attempt(|p| p.english_stmt()) {
                        return Ok(s);
                    }
                }
                _ => {}
            }
        }

        if self.peek().ident().is_some() {
            // Haskell type signatures: `fib :: Int -> Int` (spaced `::`).
            if self.nth_is_op(1, "::") && self.nth(1).space_before && self.nth(2).space_before {
                self.skip_line_balanced();
                return Ok(self.make_stmt(StmtKind::Pass, start));
            }
            if let Some(s) = self.attempt(|p| p.c_decl(h)) {
                return Ok(s);
            }
            if self.nth_is_op(1, "(")
                && !self.nth(1).space_before
                && let Some(s) = self.attempt(|p| p.math_def())
            {
                return Ok(s);
            }
            if let Some(s) = self.attempt(|p| p.haskell_def(h)) {
                return Ok(s);
            }
            // R-style assignment `x <- value`.
            if self.nth_is_op(1, "<-") && self.nth(1).line == self.peek().line {
                let (name, span) = self.expect_ident("")?;
                self.bump();
                let value = self.expr()?;
                let target = Expr::new(ExprKind::Name(name), span);
                let e = Expr::new(
                    ExprKind::Assign {
                        target: Box::new(target),
                        value: Box::new(value),
                    },
                    self.span_from(start),
                );
                return self.expr_stmt_from(e);
            }
        }

        let e = self.command_or_expr()?;
        self.expr_stmt_from(e)
    }

    /// Whether the token at offset `n` ends a simple statement.
    fn ends_simple_stmt(&self, n: usize) -> bool {
        let t = self.nth(n);
        matches!(t.tok, Tok::Newline | Tok::Eof)
            || t.is_op(";")
            || t.is_op("}")
            || t.line != self.peek().line
            || t.is_word("if")
            || t.is_word("unless")
            || t.is_word("end")
    }

    fn simple_stmt(&mut self, kind: StmtKind, start: Span) -> PResult<Stmt> {
        let s = self.make_stmt(kind, start);
        let s = self.stmt_modifiers(s)?;
        self.end_stmt()?;
        Ok(s)
    }

    /// Applies `if`/`unless` modifiers to a non-expression statement.
    fn stmt_modifiers(&mut self, stmt: Stmt) -> PResult<Stmt> {
        if !self.same_line() || !(self.at_word("if") || self.at_word("unless")) {
            return Ok(stmt);
        }
        let negate = self.bump().is_word("unless");
        let cond = self.header_expr()?;
        let span = stmt.span.to(cond.span);
        let cond = if negate {
            Expr::new(ExprKind::Unary(UnOp::Not, Box::new(cond)), span)
        } else {
            cond
        };
        let e = self.mk_if(cond, Block::new(vec![stmt]), None, span);
        Ok(Stmt {
            kind: StmtKind::Expr(e),
            span,
        })
    }

    pub(crate) fn skip_line(&mut self) {
        while !self.at_newline() && !self.at_eof() && !self.at_op(";") {
            self.bump();
        }
        self.eat_op(";");
    }

    // ----- decorators and functions -----------------------------------------

    fn looks_like_decorator(&self) -> bool {
        // `@name[.attr][(args)]` alone on its line, followed by a definition.
        let mut i = self.pos + 1;
        let toks = &self.toks;
        while i + 1 < toks.len() && toks[i].is_op(".") && toks[i + 1].ident().is_some() {
            i += 2;
        }
        if i < toks.len() && toks[i].is_op("(") {
            let mut depth = 0;
            while i < toks.len() {
                if toks[i].is_op("(") {
                    depth += 1;
                } else if toks[i].is_op(")") {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                i += 1;
            }
        }
        if i >= toks.len() || !matches!(toks[i].tok, Tok::Newline) {
            return false;
        }
        while i < toks.len() && matches!(toks[i].tok, Tok::Newline) {
            i += 1;
        }
        i < toks.len()
            && (matches!(toks[i].tok, Tok::IVar(_))
                || toks[i].ident().is_some_and(|w| {
                    FN_WORDS.contains(&w) || w == "class" || MODIFIERS.contains(&w)
                }))
    }

    fn decorated(&mut self, h: Header) -> PResult<Stmt> {
        let mut decorators = Vec::new();
        while matches!(self.peek().tok, Tok::IVar(_)) {
            let t = self.bump();
            let Tok::IVar(name) = &t.tok else {
                unreachable!()
            };
            let mut e = Expr::new(ExprKind::Name(Sym::new(name)), t.span);
            while self.at_op(".") {
                self.bump();
                let (attr, _) = self.expect_ident("after `.` in a decorator")?;
                e = Expr::new(
                    ExprKind::Member {
                        object: Box::new(e),
                        name: attr,
                        optional: false,
                    },
                    self.span_from(t.span),
                );
            }
            if self.at_op("(") {
                let args = self.call_args()?;
                e = Expr::new(
                    ExprKind::Call {
                        callee: Box::new(e),
                        args,
                    },
                    self.span_from(t.span),
                );
            }
            decorators.push(e);
            self.skip_newlines();
        }
        while self.at_any_word(MODIFIERS) {
            self.bump();
        }
        if self.at_word("class") {
            // Class decorators are accepted and applied as plain calls.
            let mut s = self.class_decl(h)?;
            if let StmtKind::Class(def) = &s.kind {
                let name = def.name;
                let span = s.span;
                let mut value = Expr::new(ExprKind::Name(name), span);
                for d in decorators.into_iter().rev() {
                    value = Expr::new(
                        ExprKind::Call {
                            callee: Box::new(d),
                            args: vec![Arg::Pos(value)],
                        },
                        span,
                    );
                }
                let assign = Expr::new(
                    ExprKind::Assign {
                        target: Box::new(Expr::new(ExprKind::Name(name), span)),
                        value: Box::new(value),
                    },
                    span,
                );
                let class_stmt = s.clone();
                s = Stmt {
                    kind: StmtKind::Expr(Expr::new(
                        ExprKind::Block(Block {
                            stmts: vec![
                                class_stmt,
                                Stmt {
                                    kind: StmtKind::Expr(assign),
                                    span,
                                },
                            ],
                            scoped: false,
                        }),
                        span,
                    )),
                    span,
                };
            }
            return Ok(s);
        }
        let is_static = decorators.iter().any(
            |d| matches!(&d.kind, ExprKind::Name(n) if n.is("staticmethod") || n.is("classmethod")),
        );
        self.fn_decl(h, decorators, is_static)
    }

    /// `fn name(params) -> T { … }` and all its cousins.
    pub(crate) fn fn_decl(
        &mut self,
        h: Header,
        decorators: Vec<Expr>,
        is_static: bool,
    ) -> PResult<Stmt> {
        let saved = (self.rust_ranges, self.kotlin_fun);
        if self.at_word("fn") {
            self.rust_ranges = true;
        }
        if self.at_word("fun") {
            self.kotlin_fun = true;
        } else {
            self.kotlin_fun = false;
        }
        let r = self.fn_decl_inner(h, decorators, is_static);
        (self.rust_ranges, self.kotlin_fun) = saved;
        r
    }

    fn fn_decl_inner(
        &mut self,
        h: Header,
        decorators: Vec<Expr>,
        mut is_static: bool,
    ) -> PResult<Stmt> {
        let kw = self.bump();
        let start = kw.span;
        let keyword = kw.ident().unwrap_or("fn").to_owned();

        // Ruby `def self.name` / `def ClassName.name` / Lua `function M.name`
        // / Lua `function obj:method` / Kotlin `fun String.shout()`.
        let mut path: Vec<(Sym, Span)> = Vec::new();
        let mut lua_method = false;
        if self.at_any_word(&["self", "this"]) && self.nth_is_op(1, ".") {
            self.bump();
            self.bump();
            is_static = true;
        }
        let (mut name, mut name_span) = self.fn_name()?;
        while (self.at_op(".") || self.at_op(":") || self.at_op("::"))
            && self.nth(1).ident().is_some()
            && !self.nth(1).space_before
        {
            lua_method = self.at_op(":");
            self.bump();
            path.push((name, name_span));
            (name, name_span) = self.fn_name()?;
        }
        let _ = name_span;
        self.skip_generics();

        let bare = !self.at_op("(")
            && !self.peek().ident().is_some_and(|_| {
                self.same_line()
                    && !self
                        .at_any_word(&["do", "then", "begin", "end", "returns", "throws", "where"])
            });
        let (params, receiver) = if self.at_op("(") {
            self.params()?
        } else if self.peek().ident().is_some()
            && self.same_line()
            && !self.at_any_word(&["do", "then", "begin", "end", "returns", "throws", "where"])
        {
            (self.bare_params()?, None)
        } else {
            (Vec::new(), None)
        };
        let ret = self.ret_type()?;
        if self.eat_word("throws") || self.eat_word("raises") {
            while self.peek().ident().is_some() || self.at_op(",") || self.at_op(".") {
                self.bump();
            }
        }
        if self.at_word("where") && !self.in_class {
            // Rust where-clauses: skip to the body.
            while !self.at_op("{") && !self.at_eof() {
                self.bump();
            }
        }
        if self.at_word("const") || self.at_word("override") || self.at_word("noexcept") {
            self.bump();
        }
        let body = if self.at_op(";") && (self.in_class || keyword == "fn") {
            // Declaration without body (interfaces, prototypes).
            self.bump();
            Block::default()
        } else {
            let b = self.body(h, true, terms::BASE)?;
            self.maybe_end(h);
            b
        };
        let span = self.span_from(start);
        let mut def = self.new_fn(Some(name), params, body, span);
        def.ret = ret;
        def.receiver = receiver;
        def.is_static = is_static;
        def.decorators = decorators;
        def.bare = bare && keyword == "def";

        if let Some((last, _)) = path.last() {
            let mut target = Expr::new(ExprKind::Name(path[0].0), path[0].1);
            for (seg, seg_span) in &path[1..] {
                target = Expr::new(
                    ExprKind::Member {
                        object: Box::new(target),
                        name: *seg,
                        optional: false,
                    },
                    *seg_span,
                );
            }
            let _ = last;
            let instance = lua_method || keyword == "fun";
            let def = Rc::new(def);
            let (methods, statics) = if instance {
                (vec![def], Vec::new())
            } else {
                (Vec::new(), vec![def])
            };
            return Ok(Stmt {
                kind: StmtKind::Impl {
                    target,
                    methods,
                    statics,
                },
                span,
            });
        }
        Ok(Stmt {
            kind: StmtKind::Fn(Rc::new(def)),
            span,
        })
    }

    /// A function name, including operator method names (`def +(other)`,
    /// C++ `operator+`).
    fn fn_name(&mut self) -> PResult<(Sym, Span)> {
        if let Tok::Op(op) = self.peek().tok
            && matches!(
                op,
                "+" | "-"
                    | "*"
                    | "/"
                    | "%"
                    | "**"
                    | "=="
                    | "<"
                    | ">"
                    | "<="
                    | ">="
                    | "<=>"
                    | "["
                    | "<<"
            )
        {
            let t = self.bump();
            if op == "[" {
                self.expect_op("]", "in `[]` method name")?;
            }
            return Ok((Sym::new(op), t.span));
        }
        if self.at_word("operator")
            && let Tok::Op(op) = self.nth(1).tok
        {
            let start = self.bump().span;
            self.bump();
            if op == "(" {
                self.expect_op(")", "in `operator()`")?;
                return Ok((Sym::new("__call__"), self.span_from(start)));
            }
            return Ok((Sym::new(op), self.span_from(start)));
        }
        if self.at_op("~") {
            // C++ destructor.
            let start = self.bump().span;
            let (_, _) = self.expect_ident("after `~`")?;
            return Ok((Sym::new("__del__"), self.span_from(start)));
        }
        self.expect_ident("for the function")
    }

    pub(crate) fn skip_generics(&mut self) {
        if !self.at_op("<") {
            return;
        }
        let mut depth = 0i32;
        loop {
            if self.at_op("<") {
                depth += 1;
            } else if self.at_op(">") {
                depth -= 1;
            } else if self.at_op(">>") {
                depth -= 2;
            } else if self.at_eof() || self.at_newline() {
                return;
            }
            self.bump();
            if depth <= 0 {
                return;
            }
        }
    }

    /// Go method receivers: `func (p *Point) Scale(f float64) { … }`.
    fn go_method(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        self.expect_op("(", "")?;
        let (recv, _) = self.expect_ident("for the receiver")?;
        self.eat_op("*");
        self.eat_op("&");
        let (ty, ty_span) = self.expect_ident("for the receiver type")?;
        self.expect_op(")", "after the receiver")?;
        let (name, _) = self.expect_ident("for the method")?;
        let (params, _) = self.params()?;
        let ret = self.ret_type()?;
        let body = self.body(h, true, terms::BASE)?;
        let span = self.span_from(start);
        let mut def = self.new_fn(Some(name), params, body, span);
        def.receiver = Some(recv);
        def.ret = ret;
        Ok(Stmt {
            kind: StmtKind::Impl {
                target: Expr::new(ExprKind::Name(ty), ty_span),
                methods: vec![Rc::new(def)],
                statics: Vec::new(),
            },
            span,
        })
    }

    /// `( params )`, returning the explicit receiver (`self`, `&self`) if any.
    pub(crate) fn params(&mut self) -> PResult<(Vec<Param>, Option<Sym>)> {
        self.expect_op("(", "to start the parameter list")?;
        self.nested(|p| {
            let mut params = Vec::new();
            let mut receiver = None;
            loop {
                if p.eat_op(")") {
                    break;
                }
                if p.at_eof() {
                    let span = p.peek().span;
                    return p.err(span, "unclosed parameter list");
                }
                // Receivers: self, this, &self, &mut self, mut self.
                if params.is_empty() && receiver.is_none() {
                    let snap = p.snapshot();
                    p.eat_op("&");
                    p.eat_word("mut");
                    if p.at_any_word(&["self", "this"])
                        && (p.nth_is_op(1, ",") || p.nth_is_op(1, ")") || p.nth_is_op(1, ":"))
                    {
                        let (r, _) = p.expect_ident("")?;
                        if p.eat_op(":") {
                            p.type_expr()?;
                        }
                        receiver = Some(r);
                        if !p.eat_op(",") {
                            p.expect_op(")", "after the parameters")?;
                            break;
                        }
                        continue;
                    }
                    p.restore(snap);
                }
                params.push(p.param()?);
                if !p.eat_op(",") {
                    p.expect_op(")", "after the parameters")?;
                    break;
                }
            }
            Ok((params, receiver))
        })
    }

    /// One parameter in any of the supported spellings.
    pub(crate) fn param(&mut self) -> PResult<Param> {
        let mut kind = ParamKind::Normal;
        if self.eat_op("**") {
            kind = ParamKind::KwRest;
        } else if self.eat_op("*") || self.eat_op("...") {
            kind = ParamKind::Rest;
        } else if self.at_op("&") {
            self.bump();
        }
        // Kotlin/Scala constructor parameters, C++ const.
        while self.at_any_word(&[
            "val", "var", "let", "const", "mut", "final", "inout", "in", "out", "ref",
        ]) && self.nth(1).ident().is_some()
        {
            self.bump();
        }
        if self.at_op("[") || self.at_op("{") || self.at_op("(") {
            let pattern = self.binding_pattern()?;
            let default = self.param_default()?;
            return Ok(Param {
                pattern,
                default,
                ty: None,
                kind,
                label: None,
            });
        }

        let first = self.peek().clone();
        let Some(first_word) = first.ident().map(str::to_owned) else {
            let found = describe(&first);
            return self.err(first.span, format!("expected a parameter, found {found}"));
        };

        // Swift external labels: `_ name: T`, `from hometown: T`.
        let mut label = None;
        if self.nth(1).ident().is_some() && self.nth_is_op(2, ":") {
            label = self.bump().ident().filter(|w| *w != "_").map(Sym::new);
        }
        let mut ty = None;
        let name;
        if self.nth_is_op(1, ":") {
            name = self.expect_ident("for the parameter")?.0;
            self.bump();
            ty = Some(self.type_expr()?);
        } else if (is_type_word(&first_word)
            || looks_capitalized(&first_word)
            || matches!(first_word.as_str(), "unsigned" | "signed" | "struct")
            || self.nth_is_op(1, "::"))
            && let Some((t, n)) = self.attempt(|p| {
                let mut t = p.type_expr()?;
                // `unsigned long long n`
                while p.at_any_word(&["long", "short", "int", "char", "double"])
                    && p.nth(1).ident().is_some()
                {
                    p.bump();
                }
                if p.eat_op("...") {
                    // Java varargs `String... args`.
                }
                let (n, _) = p.expect_ident("")?;
                // C arrays: `int arr[]`, `char buf[64]`.
                if p.at_op("[") {
                    p.bump();
                    if !p.at_op("]") {
                        p.bump();
                    }
                    p.expect_op("]", "")?;
                    t = TypeExpr::List(Box::new(t));
                }
                Ok((t, n))
            })
        {
            // C/Java style: `int x`, `String[] args`.
            if self.prev().is_op("...") {
                kind = ParamKind::Rest;
            }
            ty = Some(t);
            name = n;
        } else {
            name = self.expect_ident("for the parameter")?.0;
            if self.eat_op("...") {
                kind = ParamKind::Rest;
            }
            // Go style: `x int`, `xs ...int`.
            if self.peek().ident().is_some() && !self.at_any_word(&["in", "as"]) {
                if self.eat_op("...") {
                    kind = ParamKind::Rest;
                }
                ty = Some(self.type_expr()?);
            } else if self.at_op("...") {
                self.bump();
                kind = ParamKind::Rest;
                if self.peek().ident().is_some() {
                    ty = Some(self.type_expr()?);
                }
            }
        }
        let default = self.param_default()?;
        Ok(Param {
            pattern: Pattern::Bind(name),
            default,
            ty,
            kind,
            label,
        })
    }

    fn param_default(&mut self) -> PResult<Option<Expr>> {
        if self.at_op("=") {
            self.bump();
            return Ok(Some(self.expr_no_assign()?));
        }
        Ok(None)
    }

    /// Parameters without parentheses: `def greet name`, `let f x y = …`.
    fn bare_params(&mut self) -> PResult<Vec<Param>> {
        let mut params = Vec::new();
        while self.same_line() {
            if self.at_op("=")
                || self.at_op(":")
                || self.at_op("{")
                || self.at_op("=>")
                || self.at_op("->")
            {
                break;
            }
            if self.at_any_word(&["do", "then", "begin", "where", "returns"]) {
                break;
            }
            if self.at_op(",") {
                self.bump();
                continue;
            }
            let pattern = self.simple_pattern()?;
            params.push(Param {
                pattern,
                default: None,
                ty: None,
                kind: ParamKind::Normal,
                label: None,
            });
        }
        Ok(params)
    }

    /// Optional return type: `-> T`, `: T` (when followed by a body), or a
    /// Go-style trailing type.
    fn ret_type(&mut self) -> PResult<Option<TypeExpr>> {
        if self.at_op("->") && !self.nth_is_op(1, "{") {
            self.bump();
            return Ok(Some(self.type_expr()?));
        }
        if self.at_op(":")
            && self
                .nth(1)
                .ident()
                .is_some_and(|w| is_type_word(w) || looks_capitalized(w))
            && self.same_line()
            && let Some(t) = self.attempt(|p| {
                p.bump();
                let t = p.type_expr()?;
                if p.at_op("{") || p.at_op("=") || p.at_op("=>") || p.at_word("where") {
                    Ok(t)
                } else {
                    Err(())
                }
            })
        {
            return Ok(Some(t));
        }
        if self
            .peek()
            .ident()
            .is_some_and(|w| is_type_word(w) || looks_capitalized(w))
            && let Some(t) = self.attempt(|p| {
                let t = p.type_expr()?;
                if p.at_op("{") { Ok(t) } else { Err(()) }
            })
        {
            return Ok(Some(t));
        }
        if self.at_op("(")
            && let Some(t) = self.attempt(|p| {
                // Go multiple results: `(int, error) {`.
                let t = p.type_expr()?;
                if p.at_op("{") { Ok(t) } else { Err(()) }
            })
        {
            return Ok(Some(t));
        }
        Ok(None)
    }

    // ----- declarations --------------------------------------------------------

    /// `let` in its JS, Rust, Swift, OCaml and English flavours.
    fn let_stmt(&mut self) -> PResult<Stmt> {
        let h = self.header();
        let start = self.bump().span;
        self.eat_word("rec");
        self.eat_word("mut");
        // OCaml/F# function: `let f x y = body`.
        if self.peek().ident().is_some()
            && !self.nth_is_op(1, "=")
            && !self.nth_is_op(1, ":")
            && !self.nth_is_op(1, ",")
            && !self.nth_is_word(1, "be")
            && self.nth(1).line == self.peek().line
            && (self.nth(1).ident().is_some() || self.nth_is_op(1, "(") && self.nth(1).space_before)
            && let Some(s) = self.attempt(|p| p.ocaml_fn(h, start))
        {
            return Ok(s);
        }
        let mut stmts = vec![self.declarator(start)?];
        while self.at_op(",") && self.nth(1).ident().is_some() && self.nth_is_op(2, "=") {
            self.bump();
            let s = self.peek().span;
            stmts.push(self.declarator(s)?);
        }
        // OCaml `let x = e in body`.
        if stmts.len() == 1 && (self.at_word("in") || self.peek_past_newlines().is_word("in")) {
            self.skip_newlines();
            self.bump();
            self.skip_newlines();
            let body = self.expr()?;
            let span = self.span_from(start);
            let let_stmt = stmts.pop().ok_or(())?;
            let block = Block::new(vec![
                let_stmt,
                Stmt {
                    kind: StmtKind::Expr(body),
                    span,
                },
            ]);
            return self.expr_stmt_from(Expr::new(ExprKind::Block(block), span));
        }
        self.finish_decls(stmts, start)
    }

    fn finish_decls(&mut self, mut stmts: Vec<Stmt>, start: Span) -> PResult<Stmt> {
        self.end_stmt()?;
        if stmts.len() == 1 {
            return Ok(stmts.pop().ok_or(())?);
        }
        let span = self.span_from(start);
        Ok(Stmt {
            kind: StmtKind::Expr(Expr::new(
                ExprKind::Block(Block {
                    stmts,
                    scoped: false,
                }),
                span,
            )),
            span,
        })
    }

    fn ocaml_fn(&mut self, h: Header, start: Span) -> PResult<Stmt> {
        let (name, _) = self.expect_ident("for the function")?;
        let params = self.bare_params()?;
        if params.is_empty() {
            return Err(());
        }
        let ret = if self.eat_op(":") {
            Some(self.type_expr()?)
        } else {
            None
        };
        self.expect_op("=", "")?;
        self.skip_newlines();
        let body = if self.at_newline() {
            self.body(h, false, terms::BASE)?
        } else {
            let e = self.expr()?;
            Block::of_expr(e)
        };
        let span = self.span_from(start);
        let mut def = self.new_fn(Some(name), params, body, span);
        def.ret = ret;
        let def_stmt = Stmt {
            kind: StmtKind::Fn(Rc::new(def)),
            span,
        };
        if self.at_word("in") || self.peek_past_newlines().is_word("in") {
            self.skip_newlines();
            self.bump();
            self.skip_newlines();
            let rest = self.expr()?;
            let span = self.span_from(start);
            let block = Block {
                stmts: vec![
                    def_stmt,
                    Stmt {
                        kind: StmtKind::Expr(rest),
                        span,
                    },
                ],
                scoped: true,
            };
            return self.expr_stmt_from(Expr::new(ExprKind::Block(block), span));
        }
        self.end_stmt()?;
        Ok(def_stmt)
    }

    /// One `pattern [: T] [= value]` declarator.
    fn declarator(&mut self, start: Span) -> PResult<Stmt> {
        let mut pattern = self.binding_pattern()?;
        // Lua/Python `local a, b = 1, 2`.
        if self.at_op(",") && self.nth(1).ident().is_some() && !self.nth_is_op(2, "=") {
            let mut items = vec![pattern];
            while self.eat_op(",") {
                items.push(self.binding_pattern()?);
            }
            pattern = Pattern::Tuple(items);
        }
        let mut ty = None;
        if self.at_op(":") && !self.nth_is_op(1, "=") {
            self.bump();
            ty = Some(self.type_expr()?);
        } else if self.at_word("as") || self.at_word("As") {
            self.bump();
            self.eat_word("New");
            self.eat_word("new");
            ty = Some(self.type_expr()?);
        } else if self.peek().ident().is_some()
            && self.same_line()
            && !self.at_any_word(&["be", "in", "to", "if", "unless"])
        {
            // Go: `var x int = 5`.
            ty = Some(self.type_expr()?);
        }
        let value =
            if self.at_op("=") || self.at_op(":=") || self.at_word("be") || self.at_word("to") {
                self.bump();
                self.skip_newlines();
                let v = self.expr()?;
                Some(self.value_tuple(v)?)
            } else {
                None
            };
        Ok(self.make_stmt(StmtKind::Let { pattern, ty, value }, start))
    }

    /// `var`/`const`/`val`/`my`/`local`/`auto`/`dim` declarations.
    fn decl_stmt(&mut self) -> PResult<Stmt> {
        let start = self.bump().span;
        self.eat_word("mut");
        let mut stmts = vec![self.declarator(start)?];
        while self.at_op(",") && self.nth(1).ident().is_some() {
            self.bump();
            let s = self.peek().span;
            stmts.push(self.declarator(s)?);
        }
        self.finish_decls(stmts, start)
    }

    /// C/Java/C# declarations: `int x = 5;`, `String[] xs = {…};`,
    /// `List<int> xs;`, and C-style functions `int add(int a, int b) { … }`.
    fn c_decl(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.peek().span;
        while self.at_any_word(MODIFIERS) || self.at_word("const") || self.at_word("struct") {
            self.bump();
        }
        let first = self.peek().ident().ok_or(())?.to_owned();
        let typey = is_type_word(&first)
            || matches!(first.as_str(), "unsigned" | "signed")
            || (self.nth_is_op(1, "::") && self.nth(2).ident().is_some())
            || (looks_capitalized(&first)
                && (self.nth(1).ident().is_some()
                    || self.nth_is_op(1, "<")
                    || self.nth_is_op(1, "[")));
        if !typey || first == "print" {
            return Err(());
        }
        let ty = self.type_expr()?;
        // `unsigned long long x`, `long double y`.
        while self.at_any_word(&["long", "short", "int", "char", "double"])
            && self.nth(1).ident().is_some()
        {
            self.bump();
        }
        while self.eat_op("*") || self.eat_op("&") {}
        let t = self.peek().clone();
        let Some(name) = t.ident().map(Sym::new) else {
            return Err(());
        };
        if !t.space_before || t.line != self.prev().line {
            return Err(());
        }
        if super::expr::is_operator_word(t.ident().unwrap_or("")) {
            return Err(());
        }
        self.bump();
        if self.at_op("(") {
            // C-style function definition.
            let (params, receiver) = self.params()?;
            if self.eat_word("const") || self.eat_word("override") || self.eat_word("noexcept") {}
            if self.eat_word("throws") {
                while self.peek().ident().is_some() || self.at_op(",") || self.at_op(".") {
                    self.bump();
                }
            }
            if !(self.at_op("{") || self.at_op("=>") || self.at_newline() || self.at_op(":")) {
                return Err(());
            }
            let body = self.body(h, true, terms::BASE)?;
            let span = self.span_from(start);
            let mut def = self.new_fn(Some(name), params, body, span);
            def.ret = Some(ty);
            def.receiver = receiver;
            return Ok(Stmt {
                kind: StmtKind::Fn(Rc::new(def)),
                span,
            });
        }
        if !(self.at_op("=")
            || self.at_op(";")
            || self.at_op(",")
            || self.at_op("[")
            || self.at_newline()
            || self.at_eof()
            || self.at_op("}"))
        {
            return Err(());
        }
        let mut stmts = Vec::new();
        let mut decl_name = name;
        loop {
            let mut decl_ty = ty.clone();
            let mut array_len = None;
            if self.at_op("[") {
                // C arrays: `int a[5]`, `int a[] = {…}`.
                self.bump();
                if !self.at_op("]") {
                    array_len = Some(self.expr()?);
                }
                self.expect_op("]", "after the array size")?;
                decl_ty = TypeExpr::List(Box::new(decl_ty));
            }
            let value = if self.eat_op("=") {
                let v = self.expr()?;
                Some(array_init(v, &decl_ty))
            } else if let Some(len) = array_len {
                let span = len.span;
                Some(self.call_expr("__array_of", vec![len], span))
            } else {
                Some(default_for_type(&decl_ty, self.prev().span))
            };
            stmts.push(self.make_stmt(
                StmtKind::Let {
                    pattern: Pattern::Bind(decl_name),
                    ty: Some(decl_ty),
                    value,
                },
                start,
            ));
            if self.at_op(",") && self.nth(1).ident().is_some() {
                self.bump();
                decl_name = self.expect_ident("")?.0;
                continue;
            }
            break;
        }
        self.finish_decls(stmts, start)
    }

    /// Haskell-style definitions: `square x = x * x`, `fib 0 = 0`,
    /// `abs n | n < 0 = -n | otherwise = n`.
    fn haskell_def(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.peek().span;
        let w = self.peek().ident().ok_or(())?;
        if super::expr::is_reserved(w) || is_type_word(w) || looks_capitalized(w) {
            return Err(());
        }
        let name = Sym::new(w);
        let line = self.peek().line;
        self.bump();
        let mut params = Vec::new();
        while self.peek().line == line && !self.at_newline() && !self.at_op("=") && !self.at_op("|")
        {
            if !self.peek().space_before {
                return Err(());
            }
            let t = self.peek();
            let ok = matches!(
                t.tok,
                Tok::Ident(_) | Tok::Int(_) | Tok::Float(_) | Tok::Str(_) | Tok::BigInt(_)
            ) || t.is_op("(")
                || t.is_op("[");
            if !ok {
                return Err(());
            }
            if let Some(w) = t.ident()
                && super::expr::is_operator_word(w)
            {
                return Err(());
            }
            params.push(Param {
                pattern: self.simple_pattern()?,
                default: None,
                ty: None,
                kind: ParamKind::Normal,
                label: None,
            });
        }
        if params.is_empty() {
            // `main = do …`: a Haskell action is a definition, run when called.
            if self.at_op("=") && self.nth_is_word(1, "do") {
                self.bump();
                let e = self.expr()?;
                self.end_stmt()?;
                let span = self.span_from(start);
                let def = self.new_fn(Some(name), Vec::new(), Block::of_expr(e), span);
                return Ok(Stmt {
                    kind: StmtKind::Fn(Rc::new(def)),
                    span,
                });
            }
            return Err(());
        }
        // Guards may start on the next line.
        if self.at_newline()
            && self.peek_past_newlines().is_op("|")
            && self.peek_past_newlines().col > h.indent
        {
            self.skip_newlines();
        }
        let body = if self.at_op("|") {
            // Guards.
            let mut arms: Vec<(Expr, Expr)> = Vec::new();
            while self.at_op("|") || (self.at_newline() && self.peek_past_newlines().is_op("|")) {
                self.skip_newlines();
                self.bump();
                let cond = if self.at_word("otherwise") {
                    let t = self.bump();
                    Expr::new(ExprKind::Bool(true), t.span)
                } else {
                    self.no_bar(|p| p.expr_no_assign())?
                };
                self.expect_op("=", "after a guard")?;
                let value = self.no_bar(|p| p.expr_no_assign())?;
                arms.push((cond, value));
            }
            let mut result: Option<Expr> = None;
            for (cond, value) in arms.into_iter().rev() {
                let span = cond.span.to(value.span);
                result = Some(self.mk_if(
                    cond,
                    Block::of_expr(value),
                    result.map(Block::of_expr),
                    span,
                ));
            }
            Block::of_expr(result.ok_or(())?)
        } else {
            self.expect_op("=", "")?;
            if self.at_op("=") {
                return Err(());
            }
            self.skip_newlines();
            let e = self.expr()?;
            Block::of_expr(e)
        };
        let _ = h;
        self.end_stmt()?;
        let span = self.span_from(start);
        let def = self.new_fn(Some(name), params, body, span);
        Ok(Stmt {
            kind: StmtKind::Fn(Rc::new(def)),
            span,
        })
    }

    // ----- classes, enums, impls -----------------------------------------------

    fn class_decl(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.peek().span;
        let mut kind = self.bump().ident().unwrap_or("class").to_owned();
        if kind == "data" || kind == "case" {
            self.expect_word("class", "")?;
            kind = "data".to_owned();
        }
        let (name, _) = self.expect_ident("for the class")?;
        self.skip_generics();
        let mut auto_init = matches!(kind.as_str(), "struct" | "record" | "data");
        let namespace = matches!(kind.as_str(), "module" | "object");
        let mut fields: Vec<FieldDef> = Vec::new();
        let mut superclass = None;

        if self.at_op("(") {
            let ctor_params =
                matches!(kind.as_str(), "record" | "data") || self.paren_params_look_typed();
            if ctor_params {
                let (params, _) = self.params()?;
                auto_init = true;
                for p in params {
                    if let Some(n) = p.name() {
                        fields.push(FieldDef {
                            name: n,
                            default: p.default,
                            ty: p.ty,
                        });
                    }
                }
            } else {
                // Python bases: `class Dog(Animal):`.
                self.bump();
                let bases = self.nested(|p| {
                    let mut bases = Vec::new();
                    while !p.at_op(")") && !p.at_eof() {
                        if p.peek().ident().is_some() && p.nth_is_op(1, "=") {
                            // metaclass=… keyword arguments are ignored.
                            p.bump();
                            p.bump();
                            p.expr()?;
                        } else {
                            bases.push(p.expr()?);
                        }
                        if !p.eat_op(",") {
                            break;
                        }
                    }
                    p.expect_op(")", "after the base classes")?;
                    Ok(bases)
                })?;
                superclass = bases
                    .into_iter()
                    .find(|b| !matches!(&b.kind, ExprKind::Name(n) if n.is("object")));
            }
        }
        loop {
            if self.at_word("extends") || self.at_word("inherits") || self.at_op("<") {
                self.bump();
                superclass = Some(self.class_ref()?);
            } else if self.at_op(":")
                && self.same_line()
                && self.nth(1).ident().is_some_and(|w| {
                    looks_capitalized(w) || matches!(w, "public" | "private" | "protected")
                })
                && self.nth(1).line == self.peek().line
            {
                self.bump();
                while self.at_any_word(&["public", "private", "protected", "virtual"]) {
                    self.bump();
                }
                superclass = Some(self.class_ref()?);
                if self.at_op("(") {
                    self.call_args()?;
                }
                while self.eat_op(",") {
                    self.class_ref()?;
                }
            } else if self.at_word("implements") || self.at_word("with") || self.at_word("conforms")
            {
                self.bump();
                self.class_ref()?;
                while self.eat_op(",") {
                    self.class_ref()?;
                }
            } else {
                break;
            }
        }

        let saved = self.in_class;
        self.in_class = true;
        let members =
            if self.at_op(";") || self.at_newline() && !self.class_body_follows(h) || self.at_eof()
            {
                self.eat_op(";");
                Block::default()
            } else {
                let b = self.body_with(h, false, terms::END, Self::class_member);
                self.in_class = saved;
                let b = b?;
                self.maybe_end(h);
                b
            };
        self.in_class = saved;

        let mut def = ClassDef {
            name,
            superclass,
            methods: Vec::new(),
            statics: Vec::new(),
            fields,
            class_attrs: Vec::new(),
            auto_init,
            span: start,
        };
        self.collect_members(&mut def, members.stmts, &name, namespace)?;
        def.span = self.span_from(start);
        let span = def.span;
        Ok(Stmt {
            kind: StmtKind::Class(Rc::new(def)),
            span,
        })
    }

    fn class_ref(&mut self) -> PResult<Expr> {
        let (name, span) = self.expect_ident("for the type")?;
        let mut e = Expr::new(ExprKind::Name(name), span);
        while (self.at_op(".") || self.at_op("::")) && self.nth(1).ident().is_some() {
            self.bump();
            let (attr, s) = self.expect_ident("")?;
            e = Expr::new(
                ExprKind::Member {
                    object: Box::new(e),
                    name: attr,
                    optional: false,
                },
                span.to(s),
            );
        }
        self.skip_generics();
        Ok(e)
    }

    /// Whether `( … )` after a class name declares constructor parameters
    /// (`class P(val x: Int)`) rather than base classes (`class D(Base)`).
    fn paren_params_look_typed(&self) -> bool {
        let mut i = self.pos + 1;
        let mut depth = 0;
        while i < self.toks.len() {
            let t = &self.toks[i];
            if t.is_op("(") || t.is_op("[") || t.is_op("<") {
                depth += 1;
            } else if t.is_op(")") || t.is_op("]") || t.is_op(">") {
                if depth == 0 {
                    return false;
                }
                depth -= 1;
            } else if depth == 0 && (t.is_op(":") || t.is_word("val") || t.is_word("var")) {
                return true;
            } else if depth == 0
                && t.ident().is_some()
                && self.toks.get(i + 1).is_some_and(|n| n.ident().is_some())
            {
                // `int x` (Java record components).
                return true;
            }
            i += 1;
        }
        false
    }

    fn class_body_follows(&self, h: Header) -> bool {
        let t = self.peek_past_newlines();
        t.col > h.indent && !matches!(t.tok, Tok::Eof) || t.is_op("{") || t.is_word("end")
    }

    /// One member of a class body.
    fn class_member(&mut self) -> PResult<Stmt> {
        let start = self.peek().span;
        let h = self.header();
        let mut is_static = false;
        if matches!(self.peek().tok, Tok::IVar(_)) && self.looks_like_decorator() {
            return self.decorated(h);
        }
        loop {
            if self.at_word("static") && !self.nth_is_op(1, "(") && !self.nth_is_op(1, "=") {
                self.bump();
                is_static = true;
            } else if self.at_any_word(MODIFIERS)
                && self.nth(1).ident().is_some()
                && self.nth(1).line == self.peek().line
            {
                self.bump();
            } else {
                break;
            }
        }
        // Ruby accessors and similar metadata.
        if self.at_any_word(&[
            "attr_accessor",
            "attr_reader",
            "attr_writer",
            "include",
            "extend",
            "pass",
        ]) {
            self.skip_line();
            return Ok(self.make_stmt(StmtKind::Pass, start));
        }
        // JS getters/setters: `get area() { … }`.
        if self.at_any_word(&["get", "set"])
            && self.nth(1).ident().is_some()
            && self.nth_is_op(2, "(")
        {
            let is_get = self.bump().is_word("get");
            let mut s = self.js_method(h, is_static)?;
            if is_get && let StmtKind::Fn(def) = &mut s.kind {
                let span = def.span;
                Rc::make_mut(def)
                    .decorators
                    .push(Expr::new(ExprKind::Name(Sym::new("property")), span));
            }
            return Ok(s);
        }
        if let Some(w) = self.peek().ident()
            && FN_WORDS.contains(&w)
            && (self.nth(1).ident().is_some() || matches!(self.nth(1).tok, Tok::Op(_)))
        {
            return self.fn_decl(h, Vec::new(), is_static);
        }
        // Swift `init(…) { }`, JS `constructor(…) { }`, shorthand methods
        // `area() { … }`, Java constructors `Point(int x) { … }`.
        if self.peek().ident().is_some()
            && self.nth_is_op(1, "(")
            && let Some(s) = self.attempt(|p| p.js_method(h, is_static))
        {
            return Ok(s);
        }
        // Field declarations: `x: Int = 0`, `x;`, `int x = 0;`, `var x = 0`.
        if self.peek().ident().is_some() {
            if self.nth_is_op(1, ":") && self.nth(2).ident().is_some() {
                let (name, _) = self.expect_ident("")?;
                self.bump();
                let ty = Some(self.type_expr()?);
                let value = if self.eat_op("=") {
                    Some(self.expr()?)
                } else {
                    None
                };
                if !self.eat_op(",") {
                    self.end_stmt()?;
                }
                return Ok(self.make_stmt(
                    StmtKind::Let {
                        pattern: Pattern::Bind(name),
                        ty,
                        value,
                    },
                    start,
                ));
            }
            if (self.nth_is_op(1, ";") || self.nth_is_op(1, ","))
                && !super::expr::is_reserved(self.peek().ident().unwrap_or(""))
            {
                let (name, _) = self.expect_ident("")?;
                self.bump();
                return Ok(self.make_stmt(
                    StmtKind::Let {
                        pattern: Pattern::Bind(name),
                        ty: None,
                        value: None,
                    },
                    start,
                ));
            }
        }
        if let Some(s) = self.attempt(|p| p.go_fields()) {
            return Ok(s);
        }
        let mut s = self.statement()?;
        if is_static {
            match &mut s.kind {
                StmtKind::Fn(def) => Rc::make_mut(def).is_static = true,
                StmtKind::Let { pattern, value, .. } => {
                    // `static int count = 0;` → class attribute.
                    if let Pattern::Bind(n) = pattern {
                        let v = value.take().unwrap_or(Expr::new(ExprKind::Nil, s.span));
                        let target = Expr::new(ExprKind::Name(*n), s.span);
                        s.kind = StmtKind::Expr(Expr::new(
                            ExprKind::Assign {
                                target: Box::new(target),
                                value: Box::new(v),
                            },
                            s.span,
                        ));
                    }
                }
                _ => {}
            }
        }
        Ok(s)
    }

    /// Go struct fields: `Name string`, `Width, Height float64 `json:"w"``.
    fn go_fields(&mut self) -> PResult<Stmt> {
        let start = self.peek().span;
        let mut names = vec![self.expect_ident("")?.0];
        while self.eat_op(",") {
            names.push(self.expect_ident("")?.0);
        }
        let t = self.peek();
        let typey = t.line == self.prev().line
            && (t.is_op("[")
                || t.is_op("*")
                || t.ident()
                    .is_some_and(|w| is_type_word(w) || looks_capitalized(w) || w == "map"));
        if !typey {
            return Err(());
        }
        let ty = self.type_expr()?;
        if matches!(self.peek().tok, Tok::Str(_)) {
            self.bump(); // struct tag
        }
        if !(self.at_newline() || self.at_op(";") || self.at_op("}") || self.at_eof()) {
            return Err(());
        }
        self.eat_op(";");
        let stmts = names
            .into_iter()
            .map(|name| {
                self.make_stmt(
                    StmtKind::Let {
                        pattern: Pattern::Bind(name),
                        ty: Some(ty.clone()),
                        value: None,
                    },
                    start,
                )
            })
            .collect();
        self.finish_decls(stmts, start)
    }

    /// A method written without a keyword: `name(params) { body }`.
    fn js_method(&mut self, h: Header, is_static: bool) -> PResult<Stmt> {
        let start = self.peek().span;
        let (name, _) = self.expect_ident("for the method")?;
        let (params, receiver) = self.params()?;
        let ret = self.ret_type()?;
        if !(self.at_op("{")
            || self.at_op(":")
            || self.at_op("=>")
            || self.at_newline() && self.in_class)
        {
            return Err(());
        }
        if self.at_newline() {
            // Ruby-ish methods need a keyword; a bare call on its own line is
            // not a method definition.
            let next = self.peek_past_newlines();
            if !(next.col > h.indent && !next.is_word("end")) {
                return Err(());
            }
        }
        let body = self.body(h, true, terms::BASE)?;
        self.maybe_end(h);
        let span = self.span_from(start);
        let mut def = self.new_fn(Some(name), params, body, span);
        def.ret = ret;
        def.receiver = receiver;
        def.is_static = is_static;
        Ok(Stmt {
            kind: StmtKind::Fn(Rc::new(def)),
            span,
        })
    }

    fn collect_members(
        &mut self,
        def: &mut ClassDef,
        members: Vec<Stmt>,
        class_name: &Sym,
        namespace: bool,
    ) -> PResult<()> {
        // Multi-declarations (`int x, y;`, Go `X, Y float64`) arrive as blocks.
        let members: Vec<Stmt> = members
            .into_iter()
            .flat_map(|m| match m.kind {
                StmtKind::Expr(Expr {
                    kind: ExprKind::Block(b),
                    ..
                }) if b
                    .stmts
                    .iter()
                    .all(|s| matches!(s.kind, StmtKind::Let { .. })) =>
                {
                    b.stmts
                }
                kind => vec![Stmt { kind, span: m.span }],
            })
            .collect();
        for m in members {
            match m.kind {
                StmtKind::Fn(f) => {
                    let mut f = Rc::unwrap_or_clone(f);
                    let fname = f.name.map(|n| n.as_str());
                    if let Some(n) = fname.as_deref()
                        && (CTOR_NAMES.contains(&n) || n == &*class_name.as_str())
                    {
                        f.name = Some(Sym::new("init"));
                    }
                    if f.is_static || namespace {
                        def.statics.push(Rc::new(f));
                    } else {
                        def.methods.push(Rc::new(f));
                    }
                }
                StmtKind::Let {
                    pattern: Pattern::Bind(name),
                    ty,
                    value,
                } => def.fields.push(FieldDef {
                    name,
                    default: value,
                    ty,
                }),
                StmtKind::Expr(Expr {
                    kind: ExprKind::Assign { target, value },
                    ..
                }) if matches!(target.kind, ExprKind::Name(_)) => {
                    if let ExprKind::Name(n) = target.kind {
                        def.class_attrs.push((n, *value));
                    }
                }
                StmtKind::Class(inner) => {
                    let span = inner.span;
                    let n = inner.name;
                    def.class_attrs.push((
                        n,
                        Expr::new(
                            ExprKind::Block(Block {
                                stmts: vec![
                                    Stmt {
                                        kind: StmtKind::Class(inner),
                                        span,
                                    },
                                    Stmt {
                                        kind: StmtKind::Expr(Expr::new(ExprKind::Name(n), span)),
                                        span,
                                    },
                                ],
                                scoped: true,
                            }),
                            span,
                        ),
                    ));
                }
                StmtKind::Enum(inner) => {
                    let span = inner.span;
                    let n = inner.name;
                    def.class_attrs.push((
                        n,
                        Expr::new(
                            ExprKind::Block(Block {
                                stmts: vec![
                                    Stmt {
                                        kind: StmtKind::Enum(inner),
                                        span,
                                    },
                                    Stmt {
                                        kind: StmtKind::Expr(Expr::new(ExprKind::Name(n), span)),
                                        span,
                                    },
                                ],
                                scoped: true,
                            }),
                            span,
                        ),
                    ));
                }
                StmtKind::Pass => {}
                StmtKind::Expr(Expr {
                    kind: ExprKind::Str(_) | ExprKind::Interp(_),
                    ..
                }) => {} // docstring
                _ => {
                    return self.err(
                        m.span,
                        "only methods, fields and nested types may appear in a class body",
                    );
                }
            }
        }
        Ok(())
    }

    /// `enum Color { Red, Green, Blue }`, Rust/Swift payload enums, TS valued enums.
    fn enum_decl(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        self.eat_word("class");
        let (name, _) = self.expect_ident("for the enum")?;
        self.skip_generics();
        if self.at_op(":") && self.nth(1).ident().is_some() && self.nth(1).line == self.peek().line
        {
            self.bump();
            self.type_expr()?;
        }
        let saved = self.in_class;
        self.in_class = true;
        let items = self.body_with(h, false, terms::END, Self::enum_item);
        self.in_class = saved;
        let items = items?;
        self.maybe_end(h);
        let mut variants = Vec::new();
        let mut methods = Vec::new();
        for item in items.stmts {
            match item.kind {
                StmtKind::Enum(e) => variants.extend(e.variants.iter().cloned()),
                StmtKind::Fn(f) => methods.push(f),
                StmtKind::Pass => {}
                _ => return self.err(item.span, "unexpected item in an enum body"),
            }
        }
        let span = self.span_from(start);
        Ok(Stmt {
            kind: StmtKind::Enum(Rc::new(EnumDef {
                name,
                variants,
                methods,
                span,
            })),
            span,
        })
    }

    /// One enum body item: variants (wrapped in a temporary `Enum` statement)
    /// or a method.
    fn enum_item(&mut self) -> PResult<Stmt> {
        let start = self.peek().span;
        let h = self.header();
        if let Some(w) = self.peek().ident()
            && FN_WORDS.contains(&w)
        {
            return self.fn_decl(h, Vec::new(), false);
        }
        self.eat_word("case");
        let mut variants = Vec::new();
        loop {
            if self.at_op(";") {
                self.bump();
                break;
            }
            let (vname, _) = self.expect_ident("for the variant")?;
            let mut arity = 0;
            let mut field_names = Vec::new();
            if self.at_op("(") {
                let (params, _) = self.nested(|p| {
                    p.bump();
                    let mut params = Vec::new();
                    while !p.at_op(")") && !p.at_eof() {
                        if p.peek().ident().is_some() && p.nth_is_op(1, ":") {
                            let (n, _) = p.expect_ident("")?;
                            p.bump();
                            p.type_expr()?;
                            params.push(Some(n));
                        } else {
                            p.type_expr()?;
                            params.push(None);
                        }
                        if !p.eat_op(",") {
                            break;
                        }
                    }
                    p.expect_op(")", "after the variant fields")?;
                    Ok((params, ()))
                })?;
                arity = params.len();
                if params.iter().all(Option::is_some) {
                    field_names = params.into_iter().flatten().collect();
                }
            } else if self.at_op("{") {
                self.bump();
                self.nested(|p| {
                    while !p.at_op("}") && !p.at_eof() {
                        p.skip_newlines();
                        let (n, _) = p.expect_ident("for the field")?;
                        if p.eat_op(":") {
                            p.type_expr()?;
                        }
                        field_names.push(n);
                        p.skip_newlines();
                        if !p.eat_op(",") {
                            p.skip_newlines();
                            break;
                        }
                    }
                    p.expect_op("}", "after the variant fields")?;
                    Ok(())
                })?;
                arity = field_names.len();
            }
            let value = if self.eat_op("=") {
                Some(self.expr()?)
            } else {
                None
            };
            variants.push(VariantDef {
                name: vname,
                arity,
                field_names,
                value,
            });
            if self.eat_op(",") {
                self.skip_newlines();
                if self.at_op("}") || self.at_op(";") || self.at_terminator(terms::END) {
                    break;
                }
                continue;
            }
            break;
        }
        self.eat_op(";");
        let span = self.span_from(start);
        Ok(Stmt {
            kind: StmtKind::Enum(Rc::new(EnumDef {
                name: Sym::new("_"),
                variants,
                methods: Vec::new(),
                span,
            })),
            span,
        })
    }

    /// `type …`: ML/F# algebraic data types, Go structs, or ignored aliases.
    fn type_decl(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.peek().span;
        if self.nth_is_word(2, "struct") {
            // Go: `type Point struct { X int; Y int }`.
            self.bump();
            let (name, _) = self.expect_ident("")?;
            self.bump();
            let saved = self.in_class;
            self.in_class = true;
            let members = self.brace_block_with(Self::go_field);
            self.in_class = saved;
            let mut def = ClassDef {
                name,
                superclass: None,
                methods: Vec::new(),
                statics: Vec::new(),
                fields: Vec::new(),
                class_attrs: Vec::new(),
                auto_init: true,
                span: start,
            };
            self.collect_members(&mut def, members?.stmts, &name, false)?;
            def.span = self.span_from(start);
            let span = def.span;
            return Ok(Stmt {
                kind: StmtKind::Class(Rc::new(def)),
                span,
            });
        }
        if self.nth_is_word(2, "interface") {
            self.bump();
            self.bump();
            self.bump();
            if self.at_op("{") {
                self.skip_balanced();
            }
            return Ok(self.make_stmt(StmtKind::Pass, start));
        }
        if let Some(s) = self.attempt(|p| p.adt_decl(h)) {
            return Ok(s);
        }
        // A type alias: parsed loosely and ignored at runtime.
        self.skip_line_balanced();
        Ok(self.make_stmt(StmtKind::Pass, start))
    }

    fn go_field(&mut self) -> PResult<Stmt> {
        if let Some(s) = self.attempt(|p| p.go_fields()) {
            return Ok(s);
        }
        let start = self.peek().span;
        let (name, _) = self.expect_ident("for the field")?;
        let ty = if self.same_line() && !self.at_op(";") {
            Some(self.type_expr()?)
        } else {
            None
        };
        // Struct tags: `json:"name"`.
        while matches!(self.peek().tok, Tok::Str(_)) {
            self.bump();
        }
        self.end_stmt()?;
        Ok(self.make_stmt(
            StmtKind::Let {
                pattern: Pattern::Bind(name),
                ty,
                value: None,
            },
            start,
        ))
    }

    /// ML/Haskell ADTs: `type shape = Circle of float | Square of float`,
    /// `data Shape = Circle Float | Rect Float Float deriving Show`.
    fn adt_decl(&mut self, _h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        let (name, _) = self.expect_ident("for the type")?;
        // Type parameters (`type 'a tree`, `data Tree a`).
        while self.peek().ident().is_some() && !self.at_op("=") {
            self.bump();
        }
        self.skip_generics();
        self.expect_op("=", "")?;
        self.skip_newlines();
        self.eat_op("|");
        let mut variants = Vec::new();
        loop {
            self.skip_newlines();
            let t = self.peek().clone();
            let Some(vname) = t.ident().filter(|w| looks_capitalized(w)).map(Sym::new) else {
                return Err(());
            };
            self.bump();
            let mut arity = 0;
            let mut field_names = Vec::new();
            if self.eat_word("of") {
                // OCaml: `of float * float`.
                loop {
                    self.type_atom()?;
                    arity += 1;
                    if !self.eat_op("*") {
                        break;
                    }
                }
            } else if self.at_op("{") {
                // Haskell record syntax: `Circle { radius :: Float }`.
                self.bump();
                while !self.at_op("}") && !self.at_eof() {
                    let (n, _) = self.expect_ident("")?;
                    field_names.push(n);
                    if self.eat_op("::") || self.eat_op(":") {
                        self.type_expr()?;
                    }
                    if !self.eat_op(",") {
                        break;
                    }
                }
                self.expect_op("}", "")?;
                arity = field_names.len();
            } else if self.at_op("(") {
                let (params, _) = self.params()?;
                arity = params.len();
            } else {
                // Haskell: juxtaposed field types.
                while self.same_line() && !self.at_op("|") && !self.at_word("deriving") {
                    self.type_atom()?;
                    arity += 1;
                }
            }
            variants.push(VariantDef {
                name: vname,
                arity,
                field_names,
                value: None,
            });
            if self.at_op("|") || (self.at_newline() && self.peek_past_newlines().is_op("|")) {
                self.skip_newlines();
                self.bump();
                continue;
            }
            break;
        }
        if self.eat_word("deriving") {
            self.skip_line_balanced();
        }
        self.end_stmt()?;
        let span = self.span_from(start);
        Ok(Stmt {
            kind: StmtKind::Enum(Rc::new(EnumDef {
                name,
                variants,
                methods: Vec::new(),
                span,
            })),
            span,
        })
    }

    /// `impl Point { … }`, `impl Display for Point { … }`, `extension String { … }`.
    fn impl_decl(&mut self, h: Header) -> PResult<Stmt> {
        let kw = self.bump();
        let start = kw.span;
        let swift = kw.is_word("extension");
        self.skip_generics();
        let mut target = self.class_ref()?;
        if self.eat_word("for") {
            target = self.class_ref()?;
        }
        if self.at_op(":") && self.nth(1).ident().is_some() {
            // Swift protocol conformances.
            self.bump();
            self.class_ref()?;
            while self.eat_op(",") {
                self.class_ref()?;
            }
        }
        if self.at_word("where") {
            while !self.at_op("{") && !self.at_eof() {
                self.bump();
            }
        }
        let saved = self.in_class;
        self.in_class = true;
        let body = self.body_with(h, false, terms::END, Self::class_member);
        self.in_class = saved;
        let body = body?;
        self.maybe_end(h);
        let mut methods = Vec::new();
        let mut statics = Vec::new();
        for s in body.stmts {
            match s.kind {
                StmtKind::Fn(f) => {
                    if f.receiver.is_some() || (swift && !f.is_static) {
                        methods.push(f);
                    } else {
                        statics.push(f);
                    }
                }
                StmtKind::Pass => {}
                _ => {
                    // Associated types/consts are ignored.
                }
            }
        }
        let span = self.span_from(start);
        Ok(Stmt {
            kind: StmtKind::Impl {
                target,
                methods,
                statics,
            },
            span,
        })
    }

    pub(crate) fn skip_balanced(&mut self) {
        let mut depth = 0;
        loop {
            if self.at_eof() {
                return;
            }
            let t = self.bump();
            if t.is_op("(") || t.is_op("[") || t.is_op("{") {
                depth += 1;
            } else if t.is_op(")") || t.is_op("]") || t.is_op("}") {
                depth -= 1;
            }
            if depth <= 0 {
                return;
            }
        }
    }

    fn skip_line_balanced(&mut self) {
        while !self.at_eof() && !self.at_newline() && !self.at_op(";") {
            if self.at_op("(") || self.at_op("[") || self.at_op("{") {
                self.skip_balanced();
            } else {
                self.bump();
            }
        }
        self.eat_op(";");
    }

    // ----- loops ---------------------------------------------------------------

    fn while_stmt(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.peek().span;
        let negate = self.bump().is_word("until");
        let cond = self.header_expr()?;
        let body = self.body(h, false, terms::BASE)?;
        self.maybe_end(h);
        let span = self.span_from(start);
        Ok(Stmt {
            kind: StmtKind::While {
                cond,
                body,
                negate,
                post: false,
            },
            span,
        })
    }

    fn loop_stmt(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        let body = self.body(h, false, terms::BASE)?;
        self.maybe_end(h);
        Ok(self.make_stmt(StmtKind::Loop(body), start))
    }

    fn repeat_stmt(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        let block_next =
            self.at_op("{") || self.at_op(":") || self.at_newline() || self.at_word("do");
        if block_next {
            // Lua `repeat … until cond`, or `repeat { … } while cond`.
            let body = self.body(h, false, terms::BASE | terms::UNTIL)?;
            self.skip_newlines();
            if self.at_word("until") || self.at_word("while") {
                let negate = self.bump().is_word("until");
                let cond = self.header_expr()?;
                self.end_stmt()?;
                return Ok(self.make_stmt(
                    StmtKind::While {
                        cond,
                        body,
                        negate,
                        post: true,
                    },
                    start,
                ));
            }
            self.maybe_end(h);
            return Ok(self.make_stmt(StmtKind::Loop(body), start));
        }
        let count = self.in_header(|p| p.expr_bp(super::expr::bp::RANGE + 1))?;
        self.eat_word("times");
        let body = self.body(h, false, terms::BASE)?;
        self.maybe_end(h);
        Ok(self.make_stmt(StmtKind::Repeat { count, body }, start))
    }

    /// Mathematical definitions: `f(x) = x^2 + 1`, `loss(w, b) = (w - b)^2`.
    fn math_def(&mut self) -> PResult<Stmt> {
        let start = self.peek().span;
        let (name, _) = self.expect_ident("")?;
        if super::expr::is_reserved(&name.as_str()) {
            return Err(());
        }
        self.expect_op("(", "")?;
        let mut params = Vec::new();
        while !self.at_op(")") {
            let (p, _) = self.expect_ident("")?;
            params.push(simple_param(p));
            if !self.eat_op(",") {
                break;
            }
        }
        self.expect_op(")", "")?;
        if !self.at_op("=") || self.nth_is_op(1, "=") {
            return Err(());
        }
        self.bump();
        let body = self.expr()?;
        self.end_stmt()?;
        let span = self.span_from(start);
        let def = self.new_fn(Some(name), params, Block::of_expr(body), span);
        Ok(Stmt {
            kind: StmtKind::Fn(Rc::new(def)),
            span,
        })
    }

    /// Swift: `guard cond else { … }` runs the block when `cond` fails;
    /// `guard let x = opt else { return }` binds `x` for the rest of the scope.
    fn guard_stmt(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        let mut conds = vec![self.in_header(|p| p.expr())?];
        while self.eat_op(",") {
            conds.push(self.in_header(|p| p.expr())?);
        }
        self.expect_word("else", "after the `guard` condition")?;
        let body = self.body(h, false, terms::BASE)?;
        let mut cond = conds.remove(0);
        for c in conds {
            let span = cond.span.to(c.span);
            cond = Expr::new(ExprKind::And(Box::new(cond), Box::new(c)), span);
        }
        let span = self.span_from(start);
        let negated = Expr::new(ExprKind::Unary(UnOp::Not, Box::new(cond)), span);
        let e = self.mk_if(negated, body, None, span);
        Ok(Stmt {
            kind: StmtKind::Expr(e),
            span,
        })
    }

    /// Whether the current line ends with `{`.
    fn line_ends_with_brace(&self) -> bool {
        let line = self.peek().line;
        let mut i = 0;
        let mut last = None;
        while self.nth(i).line == line && !matches!(self.nth(i).tok, Tok::Newline | Tok::Eof) {
            last = Some(i);
            i += 1;
        }
        last.is_some_and(|i| self.nth(i).is_op("{"))
    }

    fn for_stmt(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        let _each = self.eat_word("each") || self.eat_word("every");
        let parens = self.at_op("(");
        if self.c_for_ahead() {
            return self.c_for(h, start);
        }
        if parens {
            self.bump();
        }
        // Rust's `for i in 0..n {` excludes `n`.
        let saved = self.rust_ranges;
        if !parens && self.line_ends_with_brace() {
            self.rust_ranges = true;
        }
        let result = self.in_header(|p| {
            if parens {
                p.no_brace = false;
            }
            p.for_header(parens)
        });
        self.rust_ranges = saved;
        let (pattern, iter) = result?;
        if parens {
            self.expect_op(")", "to close the `for` header")?;
        }
        let body = self.body(h, false, terms::BASE)?;
        self.maybe_end(h);
        // BASIC: `next i`.
        if self.at_word("next") && self.nth(1).ident().is_some() && self.ends_simple_stmt(2) {
            self.bump();
            self.bump();
        }
        let fresh_scope = block_has_closure(&body);
        Ok(self.make_stmt(
            StmtKind::For {
                pattern,
                iter,
                body,
                fresh_scope,
            },
            start,
        ))
    }

    fn for_header(&mut self, parens: bool) -> PResult<(Pattern, Expr)> {
        let start = self.peek().span;
        // PHP: `foreach ($xs as $x)` / `foreach ($m as $k => $v)`.
        if parens
            && let Some(r) = self.attempt(|p| {
                let iter = p.expr()?;
                p.expect_word("as", "")?;
                let first = p.binding_pattern()?;
                let pattern = if p.eat_op("=>") {
                    Pattern::Tuple(vec![first, p.binding_pattern()?])
                } else {
                    first
                };
                let iter = if matches!(pattern, Pattern::Tuple(_)) {
                    let span = iter.span;
                    p.call_expr("__items", vec![iter], span)
                } else {
                    iter
                };
                Ok((pattern, iter))
            })
        {
            return Ok(r);
        }
        // Declaration keywords and C-style types before the loop variable.
        while self.at_any_word(&[
            "let", "var", "const", "val", "my", "local", "auto", "mut", "final",
        ]) || (self.peek().ident().is_some_and(is_type_word) && self.nth(1).ident().is_some())
        {
            self.bump();
        }
        if self.peek().ident().is_some_and(looks_capitalized) && self.nth(1).ident().is_some() {
            self.bump(); // Java/C#: `for (String s : xs)`
        }
        let first = self.binding_pattern()?;
        let pattern = if self.at_op(",") {
            let mut items = vec![first];
            while self.eat_op(",") {
                items.push(self.binding_pattern()?);
            }
            Pattern::Tuple(items)
        } else {
            first
        };
        if self.at_op("=") || self.at_op(":=") {
            self.bump();
            if self.eat_word("range") {
                // Go: `for i := range xs` / `for i, v := range xs`.
                let iter = self.expr()?;
                let span = iter.span;
                let f = if matches!(pattern, Pattern::Tuple(_)) {
                    "__go_range_pairs"
                } else {
                    "__go_range_keys"
                };
                return Ok((pattern, self.call_expr(f, vec![iter], span)));
            }
            let from = self.expr()?;
            if self.eat_op(",") {
                // Lua numeric for: `for i = 1, 10[, step]` (inclusive).
                let to = self.expr()?;
                let step = if self.eat_op(",") {
                    Some(Box::new(self.expr()?))
                } else {
                    None
                };
                let span = self.span_from(start);
                return Ok((
                    pattern,
                    Expr::new(
                        ExprKind::Range {
                            start: Some(Box::new(from)),
                            end: Some(Box::new(to)),
                            inclusive: true,
                            step,
                        },
                        span,
                    ),
                ));
            }
            // BASIC `for i = 1 to 10 step 2` parses as a range expression.
            return Ok((pattern, from));
        }
        if self.eat_word("from") {
            let lo = self.expr_bp(super::expr::bp::RANGE + 1)?;
            let (inclusive, down) =
                if self.eat_word("to") || self.eat_word("through") || self.eat_word("thru") {
                    (true, false)
                } else if self.eat_word("until") {
                    (false, false)
                } else if self.eat_word("downto") || self.eat_word("down") {
                    self.eat_word("to");
                    (true, true)
                } else {
                    let span = self.peek().span;
                    return self.err(span, "expected `to` in `for … from … to …`");
                };
            let hi = self.expr_bp(super::expr::bp::RANGE + 1)?;
            let mut step = if self.eat_word("by") || self.eat_word("step") {
                Some(Box::new(self.expr_bp(super::expr::bp::RANGE + 1)?))
            } else {
                None
            };
            if down && step.is_none() {
                let span = hi.span;
                step = Some(Box::new(Expr::new(ExprKind::Int(-1), span)));
            }
            let span = self.span_from(start);
            return Ok((
                pattern,
                Expr::new(
                    ExprKind::Range {
                        start: Some(Box::new(lo)),
                        end: Some(Box::new(hi)),
                        inclusive,
                        step,
                    },
                    span,
                ),
            ));
        }
        if self.at_word("in")
            || self.at_word("of")
            || self.at_op(":")
            || self.at_op("<-")
            || self.at_op("∈")
        {
            self.bump();
            let iter = self.expr()?;
            return Ok((pattern, iter));
        }
        let span = self.peek().span;
        let found = describe(self.peek());
        self.err(
            span,
            format!("expected `in` in the `for` loop, found {found}"),
        )
    }

    /// Whether a C-style `init; cond; step` header follows.
    fn c_for_ahead(&self) -> bool {
        let mut i = self.pos;
        let mut depth = 0i32;
        let parens = self.toks[i].is_op("(");
        while i < self.toks.len() {
            let t = &self.toks[i];
            match &t.tok {
                Tok::Op("(" | "[") => depth += 1,
                Tok::Op(")" | "]") => {
                    depth -= 1;
                    if depth <= 0 && parens {
                        return false;
                    }
                }
                Tok::Op("{") if depth == 0 => return false,
                Tok::Op(";") if depth == i32::from(parens) => return true,
                Tok::Newline | Tok::Eof => return false,
                _ => {}
            }
            i += 1;
        }
        false
    }

    fn c_for(&mut self, h: Header, start: Span) -> PResult<Stmt> {
        let parens = self.eat_op("(");
        let (init, cond, step) = self.nested(|p| {
            let init = if p.eat_op(";") {
                None
            } else {
                Some(Box::new(p.statement()?))
            };
            let cond = if p.at_op(";") { None } else { Some(p.expr()?) };
            p.expect_op(";", "after the loop condition")?;
            let step = if p.at_op(")") || p.at_op("{") {
                None
            } else {
                let first = p.expr()?;
                if p.at_op(",") {
                    let mut items = vec![first];
                    while p.eat_op(",") {
                        items.push(p.expr()?);
                    }
                    let span = items[0].span;
                    Some(Expr::new(ExprKind::Tuple(items), span))
                } else {
                    Some(first)
                }
            };
            Ok((init, cond, step))
        })?;
        if parens {
            self.expect_op(")", "to close the `for` header")?;
        }
        let body = self.body(h, false, terms::BASE)?;
        self.maybe_end(h);
        Ok(self.make_stmt(
            StmtKind::CFor {
                init,
                cond,
                step,
                body,
            },
            start,
        ))
    }

    fn do_stmt(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        let body = if self.at_op("{") {
            self.brace_block()?
        } else if self.at_newline() {
            self.body(h, false, terms::BASE)?
        } else {
            self.body(h, false, terms::BASE)?
        };
        let next = self.peek_past_newlines();
        if next.is_word("while") || next.is_word("until") {
            self.skip_newlines();
            let negate = self.bump().is_word("until");
            let cond = self.header_expr()?;
            self.end_stmt()?;
            return Ok(self.make_stmt(
                StmtKind::While {
                    cond,
                    body,
                    negate,
                    post: true,
                },
                start,
            ));
        }
        self.maybe_end(h);
        let span = self.span_from(start);
        self.expr_stmt_from(Expr::new(ExprKind::Block(body), span))
    }

    // ----- jumps and exceptions --------------------------------------------------

    fn return_stmt(&mut self) -> PResult<Stmt> {
        let start = self.bump().span;
        let value = if self.ends_simple_stmt(0) || self.at_terminator(terms::BASE) {
            None
        } else {
            let v = self.expr()?;
            Some(self.statement_level_tuple(v)?)
        };
        self.simple_stmt(StmtKind::Return(value), start)
    }

    fn throw_stmt(&mut self) -> PResult<Stmt> {
        let start = self.bump().span;
        if self.ends_simple_stmt(0) {
            let span = self.prev().span;
            return self.simple_stmt(StmtKind::Throw(Expr::new(ExprKind::Nil, span)), start);
        }
        let mut value = self.expr()?;
        if self.at_op(",") {
            // Ruby: `raise ArgumentError, "message"`.
            self.bump();
            let msg = self.expr()?;
            let span = value.span.to(msg.span);
            value = Expr::new(
                ExprKind::Call {
                    callee: Box::new(value),
                    args: vec![Arg::Pos(msg)],
                },
                span,
            );
        }
        if self.eat_word("from") {
            self.expr()?; // Python exception chaining is accepted and ignored.
        }
        self.simple_stmt(StmtKind::Throw(value), start)
    }

    pub(crate) fn try_stmt_pub(&mut self, h: Header) -> PResult<Stmt> {
        self.try_stmt(h)
    }

    fn try_stmt(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        let body = self.body(h, false, terms::CATCH | terms::END)?;
        self.handlers(h, start, body)
    }

    fn begin_stmt(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        let body = if self.at_op("{") {
            self.brace_block()?
        } else if self.at_newline() {
            self.skip_separators();
            self.terminated_block_with(terms::CATCH | terms::END | terms::ELSE, Self::statement)?
        } else {
            self.inline_block_with(terms::CATCH | terms::END, Self::statement)?
        };
        if self
            .peek_past_newlines()
            .ident()
            .is_some_and(|w| super::CATCH_WORDS.contains(&w))
        {
            return self.handlers(h, start, body);
        }
        self.maybe_end(h);
        let span = self.span_from(start);
        self.expr_stmt_from(Expr::new(ExprKind::Block(body), span))
    }

    fn handlers(&mut self, h: Header, start: Span, body: Block) -> PResult<Stmt> {
        let mut catches = Vec::new();
        let mut else_block = None;
        let mut finally = None;
        loop {
            let next = self.peek_past_newlines().clone();
            let attach = next.line == self.prev().line || next.col >= h.indent;
            if !attach {
                break;
            }
            let Some(w) = next.ident() else { break };
            match w {
                "catch" | "except" | "rescue" => {
                    self.skip_newlines();
                    let kw = self.bump();
                    let ch = Header {
                        indent: h.indent,
                        line: kw.line,
                    };
                    let (classes, binding) = self.catch_head(kw.ident().unwrap_or("catch"))?;
                    let body = self.body(ch, false, terms::BASE)?;
                    catches.push(Catch {
                        classes,
                        binding,
                        body,
                    });
                }
                "else" if !catches.is_empty() && finally.is_none() => {
                    self.skip_newlines();
                    let kw = self.bump();
                    let ch = Header {
                        indent: h.indent,
                        line: kw.line,
                    };
                    else_block = Some(self.body(ch, false, terms::BASE)?);
                }
                "finally" | "ensure" => {
                    self.skip_newlines();
                    let kw = self.bump();
                    let ch = Header {
                        indent: h.indent,
                        line: kw.line,
                    };
                    finally = Some(self.body(ch, false, terms::BASE)?);
                }
                _ => break,
            }
        }
        self.maybe_end(h);
        if catches.is_empty() && finally.is_none() {
            let span = self.span_from(start);
            return self.err(
                span,
                "`try` needs at least one `catch`/`except`/`rescue` or `finally`",
            );
        }
        Ok(self.make_stmt(
            StmtKind::Try {
                body,
                catches,
                else_block,
                finally,
            },
            start,
        ))
    }

    /// The head of a handler: exception classes and the bound name.
    fn catch_head(&mut self, kw: &str) -> PResult<(Vec<Expr>, Option<Sym>)> {
        let mut classes = Vec::new();
        let mut binding = None;
        let parens = self.at_op("(") && kw == "catch";
        if parens {
            self.bump();
        }
        let stop = |p: &Parser<'_>| {
            p.at_op(")")
                || p.at_op("{")
                || p.at_op(":") && !p.nth_is_op(1, ":")
                || p.at_newline()
                || p.at_eof()
                || p.at_word("then")
                || p.at_word("do")
        };
        if !stop(self) {
            if kw == "except" {
                // `except ValueError as e`, `except (A, B) as e`.
                if self.at_op("(") {
                    let e = self.expr()?;
                    match e.kind {
                        ExprKind::Tuple(items) => classes.extend(items),
                        _ => classes.push(e),
                    }
                } else {
                    classes.push(self.class_ref()?);
                    while self.eat_op(",") {
                        classes.push(self.class_ref()?);
                    }
                }
                if self.eat_word("as") {
                    binding = Some(self.expect_ident("after `as`")?.0);
                }
            } else if kw == "rescue" {
                // `rescue ZeroDivisionError, TypeError => e`, `rescue => e`.
                while self.peek().ident().is_some() {
                    classes.push(self.class_ref()?);
                    if !self.eat_op(",") {
                        break;
                    }
                }
                if self.eat_op("=>") {
                    binding = Some(self.expect_ident("after `=>`")?.0);
                }
            } else {
                // catch (e) / catch (Exception e) / catch (e: Exception) /
                // catch (IOException | X e) / catch let e / catch e.
                self.eat_word("let");
                self.eat_word("var");
                let mut names = Vec::new();
                loop {
                    let e = self.class_ref()?;
                    names.push(e);
                    if self.eat_op("|") {
                        continue;
                    }
                    break;
                }
                if self.eat_op(":") {
                    // `e: Exception`
                    if let Some(Expr {
                        kind: ExprKind::Name(n),
                        ..
                    }) = names.pop()
                    {
                        binding = Some(n);
                    }
                    classes.push(self.class_ref()?);
                } else if self.peek().ident().is_some() {
                    binding = Some(self.expect_ident("")?.0);
                    classes = names;
                } else if names.len() == 1
                    && matches!(&names[0].kind, ExprKind::Name(n) if !looks_capitalized(&n.as_str()))
                {
                    if let ExprKind::Name(n) = names[0].kind {
                        binding = Some(n);
                    }
                } else {
                    classes = names;
                }
            }
        }
        if parens {
            self.expect_op(")", "after the handler")?;
        }
        Ok((classes, binding))
    }

    /// C/Java/Go/Swift/JS `switch`.
    fn switch_stmt(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        let subject = self.header_expr()?;
        let braces = self.at_op("{");
        if braces {
            self.bump();
        } else {
            self.eat_op(":");
            self.eat_word("do");
        }
        let mut cases: Vec<SwitchCase> = Vec::new();
        let mut default = None;
        let mut pending: Vec<Expr> = Vec::new();
        loop {
            self.skip_separators();
            if braces && self.eat_op("}") {
                break;
            }
            if !braces && self.at_end_word() {
                self.maybe_end(h);
                break;
            }
            if self.at_eof() {
                return self.err(start, "unclosed `switch`");
            }
            let arrow;
            if self.eat_word("case") || self.eat_word("when") {
                loop {
                    let v = self.nested(|p| {
                        p.no_arrow_lambda = true;
                        p.switch_value()
                    })?;
                    pending.push(v);
                    if !self.eat_op(",") {
                        break;
                    }
                }
                arrow = self.at_op("->") || self.at_op("=>");
                if !(self.eat_op(":")
                    || self.eat_op("->")
                    || self.eat_op("=>")
                    || self.eat_word("then"))
                {
                    let span = self.peek().span;
                    return self.err(span, "expected `:` after the `case` value");
                }
            } else if self.eat_word("default") || self.eat_word("else") {
                arrow = self.at_op("->") || self.at_op("=>");
                if !(self.eat_op(":") || self.eat_op("->") || self.eat_op("=>")) {
                    self.eat_word("then");
                }
                let body = self.case_body(arrow)?;
                default = Some(body);
                continue;
            } else {
                let span = self.peek().span;
                let found = describe(self.peek());
                return self.err(span, format!("expected `case` or `default`, found {found}"));
            }
            let body = self.case_body(arrow)?;
            if body.stmts.is_empty() && !arrow {
                continue; // stacked labels share the next body
            }
            cases.push(SwitchCase {
                values: std::mem::take(&mut pending),
                body,
            });
        }
        Ok(self.make_stmt(
            StmtKind::Switch {
                subject,
                cases,
                default,
            },
            start,
        ))
    }

    fn switch_value(&mut self) -> PResult<Expr> {
        if self.at_op(".") && self.nth(1).ident().is_some() {
            // Swift `case .north:` — matched by variant name.
            let start = self.bump().span;
            let (name, _) = self.expect_ident("")?;
            let span = self.span_from(start);
            return Ok(self.call_expr(
                "__variant_named",
                vec![Expr::new(ExprKind::Str(name.as_str()), span)],
                span,
            ));
        }
        let lo = self.expr_bp(super::expr::bp::RANGE + 1)?;
        if self.at_op("...") || self.at_op("..") || self.at_op("..=") {
            let inclusive = !self.bump().is_op("..");
            let hi = self.expr_bp(super::expr::bp::RANGE + 1)?;
            let span = lo.span.to(hi.span);
            return Ok(Expr::new(
                ExprKind::Range {
                    start: Some(Box::new(lo)),
                    end: Some(Box::new(hi)),
                    inclusive,
                    step: None,
                },
                span,
            ));
        }
        Ok(lo)
    }

    fn case_body(&mut self, arrow: bool) -> PResult<Block> {
        if arrow && !self.at_newline() {
            if self.at_op("{") {
                return self.brace_block();
            }
            let s = self.statement()?;
            return Ok(Block::new(vec![s]));
        }
        let mut stmts = Vec::new();
        loop {
            self.skip_separators();
            if self.at_terminator(terms::END | terms::CASE | terms::WHEN)
                || self.at_word("else") && self.nth_is_op(1, ":")
            {
                break;
            }
            match self.statement() {
                Ok(s) => stmts.push(s),
                Err(()) => return Err(()),
            }
        }
        Ok(Block::new(stmts))
    }

    // ----- imports and misc --------------------------------------------------------

    fn from_import_ahead(&self) -> bool {
        let mut i = self.pos + 1;
        while i < self.toks.len() {
            let t = &self.toks[i];
            if t.is_word("import") {
                return true;
            }
            if !(t.ident().is_some() || t.is_op(".")) || matches!(t.tok, Tok::Newline) {
                return false;
            }
            i += 1;
        }
        false
    }

    fn import_stmt(&mut self) -> PResult<Stmt> {
        let kw = self.bump();
        let start = kw.span;
        let word = kw.ident().unwrap_or("import").to_owned();
        let pass = |p: &mut Self| -> PResult<Stmt> {
            p.skip_line_balanced();
            Ok(p.make_stmt(StmtKind::Pass, start))
        };
        match word.as_str() {
            "package" | "extern" => return pass(self),
            "using" => {
                // `using namespace std;` / `using System;` / C# `using (x) { }`.
                if self.at_op("(") {
                    return pass(self);
                }
                return pass(self);
            }
            "use" if self.at_word("strict") || self.at_word("warnings") => return pass(self),
            _ => {}
        }
        let mut imp = Import {
            path: Vec::new(),
            file: None,
            alias: None,
            names: None,
            wildcard: false,
        };
        if word == "from" {
            imp.path = self.module_path()?;
            self.expect_word("import", "")?;
            if self.eat_op("*") {
                imp.wildcard = true;
            } else {
                let parens = self.eat_op("(");
                imp.names = Some(self.import_names()?);
                if parens {
                    self.expect_op(")", "")?;
                }
            }
            self.end_stmt()?;
            return Ok(self.make_stmt(StmtKind::Import(imp), start));
        }
        // Go grouped imports: `import ( "fmt"; str "strings" )`.
        if self.at_op("(") && word == "import" {
            self.bump();
            let mut stmts = Vec::new();
            loop {
                self.skip_separators();
                if self.eat_op(")") || self.at_eof() {
                    break;
                }
                let s = self.peek().span;
                let alias = if self.peek().ident().is_some() || self.at_op(".") {
                    self.bump().ident().map(Sym::new)
                } else {
                    None
                };
                let mut one = Import {
                    path: Vec::new(),
                    file: None,
                    alias: None,
                    names: None,
                    wildcard: false,
                };
                self.module_source(&mut one)?;
                match alias {
                    Some(a) if a.is("_") => continue,
                    Some(a) => one.alias = Some(a),
                    None => {}
                }
                stmts.push(self.make_stmt(StmtKind::Import(one), s));
            }
            return self.finish_decls(stmts, start);
        }
        // ES modules: `import { a, b as c } from "m"`, `import x from "m"`,
        // `import * as m from "m"`.
        if word == "import" && (self.at_op("{") || self.at_op("*") || self.nth_is_word(1, "from")) {
            let mut names = None;
            let mut alias = None;
            if self.eat_op("{") {
                names = Some(self.nested(|p| {
                    let n = p.import_names()?;
                    p.expect_op("}", "")?;
                    Ok(n)
                })?);
            } else if self.eat_op("*") {
                self.expect_word("as", "")?;
                alias = Some(self.expect_ident("")?.0);
            } else {
                alias = Some(self.expect_ident("")?.0);
            }
            self.expect_word("from", "")?;
            self.module_source(&mut imp)?;
            imp.names = names;
            imp.alias = alias;
            self.end_stmt()?;
            return Ok(self.make_stmt(StmtKind::Import(imp), start));
        }
        if matches!(self.peek().tok, Tok::Str(_)) || self.at_op("<") {
            self.module_source(&mut imp)?;
        } else {
            self.eat_word("qualified");
            imp.path = self.module_path()?;
            if self.at_op("::") && self.nth_is_op(1, "{") {
                // Rust `use a::{b, c};`
                self.bump();
                self.bump();
                imp.names = Some(self.nested(|p| {
                    let n = p.import_names()?;
                    p.expect_op("}", "")?;
                    Ok(n)
                })?);
            } else if (self.at_op(".") || self.at_op("::")) && self.nth_is_op(1, "*") {
                self.bump();
                self.bump();
                imp.wildcard = true;
            }
        }
        if self.eat_word("as") {
            imp.alias = Some(self.expect_ident("after `as`")?.0);
        }
        // Python: `import a, b`.
        if self.at_op(",") && word == "import" {
            let mut stmts = vec![self.make_stmt(StmtKind::Import(imp), start)];
            while self.eat_op(",") {
                let s = self.peek().span;
                let path = self.module_path()?;
                let alias = if self.eat_word("as") {
                    Some(self.expect_ident("")?.0)
                } else {
                    None
                };
                stmts.push(self.make_stmt(
                    StmtKind::Import(Import {
                        path,
                        file: None,
                        alias,
                        names: None,
                        wildcard: false,
                    }),
                    s,
                ));
            }
            return self.finish_decls(stmts, start);
        }
        self.end_stmt()?;
        Ok(self.make_stmt(StmtKind::Import(imp), start))
    }

    fn module_source(&mut self, imp: &mut Import) -> PResult<()> {
        if self.at_op("<") {
            // `include <stdio.h>`
            while !self.at_op(">") && !self.at_eof() && !self.at_newline() {
                self.bump();
            }
            self.eat_op(">");
            imp.path = vec![Sym::new("__foreign")];
            return Ok(());
        }
        let t = self.bump();
        let Tok::Str(s) = &t.tok else {
            return self.err(t.span, "expected a module name or path");
        };
        let path = s.as_plain().unwrap_or_default();
        if path.ends_with(".chi") || path.starts_with('.') || path.starts_with('/') {
            imp.file = Some(path);
        } else {
            imp.path = path
                .split(['.', '/', ':'])
                .filter(|p| !p.is_empty())
                .map(Sym::new)
                .collect();
        }
        Ok(())
    }

    fn module_path(&mut self) -> PResult<Vec<Sym>> {
        let mut path = vec![self.expect_ident("for the module")?.0];
        while (self.at_op(".") || self.at_op("::") || self.at_op("/"))
            && self.nth(1).ident().is_some()
            && !self.nth(1).space_before
        {
            self.bump();
            path.push(self.expect_ident("")?.0);
        }
        Ok(path)
    }

    fn import_names(&mut self) -> PResult<Vec<(Sym, Option<Sym>)>> {
        let mut names = Vec::new();
        loop {
            self.skip_newlines();
            if self.at_op(")") || self.at_op("}") {
                break;
            }
            if self.at_word("self") {
                self.bump();
            } else {
                let (n, _) = self.expect_ident("to import")?;
                let alias = if self.eat_word("as") {
                    Some(self.expect_ident("after `as`")?.0)
                } else {
                    None
                };
                names.push((n, alias));
            }
            self.skip_newlines();
            if !self.eat_op(",") {
                break;
            }
        }
        Ok(names)
    }

    /// Python `with open(f) as fh:` → bind, run, and close on the way out.
    fn with_stmt(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        let mut stmts = Vec::new();
        let mut names = Vec::new();
        loop {
            let saved = self.no_as_cast;
            self.no_as_cast = true;
            let value = self.header_expr();
            self.no_as_cast = saved;
            let value = value?;
            let name = if self.eat_word("as") {
                self.expect_ident("after `as`")?.0
            } else {
                Sym::new(&format!("__with{}", names.len()))
            };
            let span = value.span;
            stmts.push(Stmt {
                kind: StmtKind::Let {
                    pattern: Pattern::Bind(name),
                    ty: None,
                    value: Some(value),
                },
                span,
            });
            names.push((name, span));
            if !self.eat_op(",") {
                break;
            }
        }
        let body = self.body(h, false, terms::BASE)?;
        self.maybe_end(h);
        let closers = names
            .iter()
            .rev()
            .map(|(n, span)| Stmt {
                kind: StmtKind::Expr(self.call_expr(
                    "__close",
                    vec![Expr::new(ExprKind::Name(*n), *span)],
                    *span,
                )),
                span: *span,
            })
            .collect();
        let span = self.span_from(start);
        stmts.push(Stmt {
            kind: StmtKind::Try {
                body,
                catches: Vec::new(),
                else_block: None,
                finally: Some(Block::new(closers)),
            },
            span,
        });
        Ok(Stmt {
            kind: StmtKind::Expr(Expr::new(
                ExprKind::Block(Block {
                    stmts,
                    scoped: false,
                }),
                span,
            )),
            span,
        })
    }

    fn delete_stmt(&mut self) -> PResult<Stmt> {
        let start = self.bump().span;
        let mut stmts = Vec::new();
        loop {
            let target = self.postfix_expr()?;
            let span = target.span;
            stmts.push(Stmt {
                kind: StmtKind::Delete(target),
                span,
            });
            if !self.eat_op(",") {
                break;
            }
        }
        self.finish_decls(stmts, start)
    }

    fn assert_stmt(&mut self) -> PResult<Stmt> {
        let start = self.bump().span;
        let cond = self.expr()?;
        let mut args = vec![Arg::Pos(cond)];
        if self.eat_op(",") || self.eat_op(":") {
            args.push(Arg::Pos(self.expr()?));
        }
        let span = self.span_from(start);
        let e = Expr::new(
            ExprKind::Call {
                callee: Box::new(self.sym_expr("assert", start)),
                args,
            },
            span,
        );
        self.expr_stmt_from(e)
    }

    /// English statements: `add 5 to total`, `set x to 10`, `swap a and b`, …
    fn english_stmt(&mut self) -> PResult<Stmt> {
        let start = self.peek().span;
        let verb = self.bump().ident().unwrap_or("").to_owned();
        let below_range = super::expr::bp::RANGE + 1;
        let e = match verb.as_str() {
            "add" => {
                let value = self.expr_bp(below_range)?;
                self.expect_word("to", "")?;
                let target = self.postfix_expr()?;
                if !target.is_assignable() {
                    return Err(());
                }
                let span = self.span_from(start);
                let added = self.call_expr("__english_add", vec![target.clone(), value], span);
                ExprKind::Assign {
                    target: Box::new(target),
                    value: Box::new(added),
                }
            }
            "subtract" => {
                let value = self.expr_bp(below_range)?;
                self.expect_word("from", "")?;
                let target = self.postfix_expr()?;
                compound(BinOp::Sub, target, value)?
            }
            "multiply" | "divide" => {
                let target = self.postfix_expr()?;
                self.expect_word("by", "")?;
                let value = self.expr()?;
                let op = if verb == "multiply" {
                    BinOp::Mul
                } else {
                    BinOp::Div
                };
                compound(op, target, value)?
            }
            "increase" | "increment" | "decrease" | "decrement" => {
                let target = self.postfix_expr()?;
                let value = if self.eat_word("by") {
                    self.expr()?
                } else {
                    Expr::new(ExprKind::Int(1), target.span)
                };
                let op = if verb.starts_with("inc") {
                    BinOp::Add
                } else {
                    BinOp::Sub
                };
                compound(op, target, value)?
            }
            "set" | "change" => {
                let target = self.postfix_expr()?;
                if !(self.eat_word("to") || self.eat_op("=")) {
                    return Err(());
                }
                if !target.is_assignable() {
                    return Err(());
                }
                let value = self.expr()?;
                ExprKind::Assign {
                    target: Box::new(target),
                    value: Box::new(value),
                }
            }
            "swap" => {
                let a = self.postfix_expr()?;
                self.expect_word("and", "")?;
                let b = self.postfix_expr()?;
                if !a.is_assignable() || !b.is_assignable() {
                    return Err(());
                }
                let span = a.span.to(b.span);
                ExprKind::Assign {
                    target: Box::new(Expr::new(ExprKind::Tuple(vec![a.clone(), b.clone()]), span)),
                    value: Box::new(Expr::new(ExprKind::Tuple(vec![b, a]), span)),
                }
            }
            _ => return Err(()),
        };
        let span = self.span_from(start);
        self.expr_stmt_from(Expr::new(e, span))
    }

    /// `a, b = b, a` and other bare tuples at statement level.
    /// The value of a declaration or assignment: a bare tuple unless the
    /// comma starts another declarator (`let a = 1, b = 2`).
    pub(crate) fn value_tuple(&mut self, first: Expr) -> PResult<Expr> {
        if self.at_op(",") && self.nth(1).ident().is_some() && self.nth_is_op(2, "=") {
            return Ok(first);
        }
        self.statement_level_tuple(first)
    }

    pub(crate) fn statement_level_tuple(&mut self, first: Expr) -> PResult<Expr> {
        if !self.at_op(",") || !self.same_line() {
            return Ok(first);
        }
        let mut items = vec![first];
        while self.eat_op(",") {
            if self.at_newline() || self.at_eof() || self.at_op("=") {
                break;
            }
            items.push(self.expr_no_assign()?);
        }
        let span = items[0].span.to(items[items.len() - 1].span);
        Ok(Expr::new(ExprKind::Tuple(items), span))
    }
}

fn compound(op: BinOp, target: Expr, value: Expr) -> PResult<ExprKind> {
    if !target.is_assignable() {
        return Err(());
    }
    Ok(ExprKind::CompoundAssign {
        op,
        target: Box::new(target),
        value: Box::new(value),
    })
}

/// C/Java array initialisers `{1, 2, 3}` arrive as set/map literals.
fn array_init(v: Expr, ty: &TypeExpr) -> Expr {
    let is_array = matches!(ty, TypeExpr::List(_))
        || matches!(ty, TypeExpr::Named(n, _) if ["vector", "Vec", "List", "ArrayList", "array", "Array"].contains(&&*n.as_str()));
    match v.kind {
        ExprKind::Set(items) if is_array => Expr::new(ExprKind::List(items), v.span),
        // `struct Point p = {3, 4};` constructs the struct positionally.
        ExprKind::Set(items) if matches!(ty, TypeExpr::Named(n, _) if super::pattern::looks_capitalized(&n.as_str())) =>
        {
            let TypeExpr::Named(n, _) = ty else {
                unreachable!()
            };
            Expr::new(
                ExprKind::Call {
                    callee: Box::new(Expr::new(ExprKind::Name(*n), v.span)),
                    args: items.into_iter().map(Arg::Pos).collect(),
                },
                v.span,
            )
        }
        ExprKind::Map(ref items) if is_array && items.is_empty() => {
            Expr::new(ExprKind::List(Vec::new()), v.span)
        }
        _ => v,
    }
}

/// Default value for an uninitialised typed declaration (`int x;` → 0).
pub(crate) fn default_for_type(ty: &TypeExpr, span: Span) -> Expr {
    let kind = match ty {
        TypeExpr::Named(n, _) => match crate::value::TypeTag::from_name(&n.as_str()) {
            Some(crate::value::TypeTag::Int) => ExprKind::Int(0),
            Some(crate::value::TypeTag::Float) => ExprKind::Float(0.0),
            Some(crate::value::TypeTag::Bool) => ExprKind::Bool(false),
            Some(crate::value::TypeTag::Str) if !n.is("char") && !n.is("rune") => {
                ExprKind::Str(Rc::from(""))
            }
            Some(crate::value::TypeTag::List) => ExprKind::List(Vec::new()),
            Some(crate::value::TypeTag::Map) if !n.is("object") && !n.is("Object") => {
                ExprKind::Map(Vec::new())
            }
            Some(crate::value::TypeTag::Set) => ExprKind::Call {
                callee: Box::new(Expr::new(ExprKind::Name(Sym::new("set")), span)),
                args: Vec::new(),
            },
            _ => ExprKind::Nil,
        },
        TypeExpr::List(_) => ExprKind::List(Vec::new()),
        _ => ExprKind::Nil,
    };
    Expr::new(kind, span)
}

/// Whether a block (transitively) creates closures, so a loop over it needs a
/// fresh scope per iteration.
pub(crate) fn block_has_closure(b: &Block) -> bool {
    fn expr(e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Lambda(_) => true,
            ExprKind::Unary(_, a) | ExprKind::Spread(a) | ExprKind::Gradient(a) => expr(a),
            ExprKind::Derivative { func, .. } => expr(func),
            ExprKind::Binary(_, a, b)
            | ExprKind::And(a, b)
            | ExprKind::Or(a, b)
            | ExprKind::Coalesce(a, b) => expr(a) || expr(b),
            ExprKind::Assign { target, value } | ExprKind::CompoundAssign { target, value, .. } => {
                expr(target) || expr(value)
            }
            ExprKind::Compare(a, rest) => expr(a) || rest.iter().any(|(_, e)| expr(e)),
            ExprKind::Call { callee, args } => {
                expr(callee)
                    || args.iter().any(|a| match a {
                        Arg::Pos(e) | Arg::Named(_, e) | Arg::Spread(e) | Arg::KwSpread(e) => {
                            expr(e)
                        }
                    })
            }
            ExprKind::Member { object, .. } => expr(object),
            ExprKind::Index { object, index } => expr(object) || expr(index),
            ExprKind::List(items) | ExprKind::Tuple(items) | ExprKind::Set(items) => {
                items.iter().any(expr)
            }
            ExprKind::Map(items) => items.iter().any(|(k, v)| expr(k) || expr(v)),
            ExprKind::If {
                cond,
                then,
                otherwise,
            } => expr(cond) || block(then) || otherwise.as_ref().is_some_and(block),
            ExprKind::Block(b) => block(b),
            ExprKind::Match { subject, arms } => {
                subject.as_ref().is_some_and(|s| expr(s)) || arms.iter().any(|a| block(&a.body))
            }
            ExprKind::Comprehension(c) => expr(&c.element),
            ExprKind::StructLit { fields, .. } => fields.iter().any(|(_, e)| expr(e)),
            _ => false,
        }
    }
    fn stmt(s: &Stmt) -> bool {
        match &s.kind {
            StmtKind::Expr(e) | StmtKind::Throw(e) => expr(e),
            StmtKind::Let { value, .. } => value.as_ref().is_some_and(expr),
            StmtKind::Fn(_) | StmtKind::Class(_) => true,
            StmtKind::While { cond, body, .. } => expr(cond) || block(body),
            StmtKind::Loop(b) | StmtKind::Defer(b) => block(b),
            StmtKind::Repeat { body, .. }
            | StmtKind::For { body, .. }
            | StmtKind::CFor { body, .. } => block(body),
            StmtKind::Return(e) => e.as_ref().is_some_and(expr),
            StmtKind::Try {
                body,
                catches,
                else_block,
                finally,
            } => {
                block(body)
                    || catches.iter().any(|c| block(&c.body))
                    || else_block.as_ref().is_some_and(block)
                    || finally.as_ref().is_some_and(block)
            }
            StmtKind::Switch { cases, default, .. } => {
                cases.iter().any(|c| block(&c.body)) || default.as_ref().is_some_and(block)
            }
            _ => false,
        }
    }
    fn block(b: &Block) -> bool {
        b.stmts.iter().any(stmt)
    }
    block(b)
}
