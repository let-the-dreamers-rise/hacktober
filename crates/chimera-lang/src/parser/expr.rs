//! Expressions: a Pratt parser with English operators, every common lambda
//! syntax, comprehensions, `match`/`case`/`when`, struct literals and
//! interpolated strings.

use std::rc::Rc;

use chimera_core::Span;

use super::pattern::looks_capitalized;
use super::{Header, PResult, Parser, describe, simple_param, terms};
use crate::ast::*;
use crate::bigint::BigInt;
use crate::lexer::{StrLit, StrPart, Tok};
use crate::symbol::Sym;

/// Binding powers, lowest to highest.
pub(crate) mod bp {
    pub const PIPE: u8 = 2;
    pub const TERNARY: u8 = 3;
    pub const COALESCE: u8 = 4;
    pub const OR: u8 = 5;
    pub const AND: u8 = 6;
    pub const NOT: u8 = 7;
    pub const COMPARE: u8 = 8;
    pub const BITOR: u8 = 9;
    pub const BITXOR: u8 = 10;
    pub const BITAND: u8 = 11;
    pub const SHIFT: u8 = 12;
    pub const RANGE: u8 = 13;
    pub const ADD: u8 = 14;
    pub const MUL: u8 = 15;
    pub const UNARY: u8 = 16;
    pub const POW: u8 = 17;
}

/// Words that never start a command call and are not plain names at
/// statement start.
pub(crate) fn is_reserved(w: &str) -> bool {
    matches!(
        w,
        "if" | "unless"
            | "else"
            | "elif"
            | "elsif"
            | "elseif"
            | "while"
            | "until"
            | "for"
            | "foreach"
            | "do"
            | "end"
            | "then"
            | "return"
            | "break"
            | "continue"
            | "def"
            | "fn"
            | "func"
            | "function"
            | "fun"
            | "class"
            | "struct"
            | "enum"
            | "let"
            | "var"
            | "const"
            | "val"
            | "in"
            | "and"
            | "or"
            | "not"
            | "is"
            | "true"
            | "false"
            | "nil"
            | "null"
            | "None"
            | "True"
            | "False"
            | "self"
            | "this"
            | "super"
            | "new"
            | "typeof"
            | "import"
            | "from"
            | "match"
            | "case"
            | "when"
            | "switch"
            | "try"
            | "catch"
            | "except"
            | "finally"
            | "raise"
            | "throw"
            | "lambda"
            | "yield"
            | "await"
            | "select"
            | "SELECT"
            | "with"
            | "as"
            | "of"
            | "to"
            | "loop"
            | "repeat"
            | "where"
            | "otherwise"
            | "undefined"
            | "begin"
            | "rescue"
            | "ensure"
            | "please"
            | "the"
            | "fi"
            | "done"
            | "proc"
            | "sub"
            | "impl"
            | "trait"
            | "type"
            | "data"
            | "model"
            | "train"
            | "defer"
            | "pass"
            | "del"
            | "global"
            | "nonlocal"
            | "assert"
    )
}

/// Words that act as infix operators.
pub(crate) fn is_operator_word(w: &str) -> bool {
    // Haskell backtick infix names (`mod`) are operators.
    w.starts_with('`')
        || matches!(
            w.to_ascii_lowercase().as_str(),
            "and"
                | "or"
                | "in"
                | "is"
                | "if"
                | "unless"
                | "while"
                | "until"
                | "then"
                | "do"
                | "else"
                | "end"
                | "as"
                | "of"
                | "to"
                | "by"
                | "mod"
                | "div"
                | "xor"
                | "plus"
                | "minus"
                | "times"
                | "divided"
                | "multiplied"
                | "modulo"
                | "equals"
                | "squared"
                | "cubed"
                | "with"
                | "step"
                | "downto"
                | "through"
                | "like"
                | "between"
                | "not"
                | "contains"
                | "matches"
                | "over"
                | "where"
                | "when"
                | "otherwise"
                | "elif"
                | "for"
                | "on"
        )
}

/// Functions printed by command calls such as `puts [1, 2]`.
fn print_like(w: &str) -> bool {
    matches!(
        w,
        "print"
            | "puts"
            | "p"
            | "pp"
            | "say"
            | "echo"
            | "show"
            | "display"
            | "println"
            | "printf"
            | "putStrLn"
            | "log"
            | "warn"
            | "write"
            | "writeln"
            | "disp"
            | "cat"
    )
}

impl<'s> Parser<'s> {
    // ----- entry points ------------------------------------------------------

    /// A full expression, including assignment.
    pub(crate) fn expr(&mut self) -> PResult<Expr> {
        self.enter()?;
        let r = self.expr_inner();
        self.leave();
        r
    }

    fn expr_inner(&mut self) -> PResult<Expr> {
        let lhs = self.expr_bp(bp::PIPE)?;
        // `x ||= v`, `x &&= v`, `x ??= v` → `x = x || v` (etc.).
        for (op, make) in [
            ("||=", ExprKind::Or as fn(Box<Expr>, Box<Expr>) -> ExprKind),
            ("&&=", ExprKind::And),
            ("??=", ExprKind::Coalesce),
        ] {
            if self.at_op(op) && lhs.is_assignable() {
                self.bump();
                self.skip_newlines();
                let rhs = self.expr()?;
                let span = lhs.span.to(rhs.span);
                let value = Expr::new(make(Box::new(lhs.clone()), Box::new(rhs)), span);
                return Ok(Expr::new(
                    ExprKind::Assign {
                        target: Box::new(lhs),
                        value: Box::new(value),
                    },
                    span,
                ));
            }
        }
        let Some(op) = self.assign_op() else {
            return Ok(lhs);
        };
        if !lhs.is_assignable() {
            let span = self.peek().span;
            return self.err(span, "cannot assign to this expression");
        }
        self.bump();
        self.skip_newlines();
        let rhs = self.expr()?;
        let span = lhs.span.to(rhs.span);
        let kind = match op {
            None => ExprKind::Assign {
                target: Box::new(lhs),
                value: Box::new(rhs),
            },
            Some(op) => ExprKind::CompoundAssign {
                op,
                target: Box::new(lhs),
                value: Box::new(rhs),
            },
        };
        Ok(Expr::new(kind, span))
    }

    /// `Some(None)` = plain assignment, `Some(Some(op))` = compound.
    fn assign_op(&self) -> Option<Option<BinOp>> {
        let Tok::Op(op) = self.peek().tok else {
            return None;
        };
        Some(match op {
            "=" if !self.eq_is_cmp => None,
            ":=" => None,
            "+=" => Some(BinOp::Add),
            "-=" => Some(BinOp::Sub),
            "*=" => Some(BinOp::Mul),
            "/=" => Some(BinOp::Div),
            "%=" => Some(BinOp::Mod),
            "//=" => Some(BinOp::FloorDiv),
            "**=" | "^=" => Some(BinOp::Pow),
            "&=" => Some(BinOp::BitAnd),
            "|=" => Some(BinOp::BitOr),
            "<<=" => Some(BinOp::Shl),
            ">>=" => Some(BinOp::Shr),
            _ => return None,
        })
    }

    /// An expression without assignment (call arguments, defaults, …).
    pub(crate) fn expr_no_assign(&mut self) -> PResult<Expr> {
        self.expr_bp(bp::PIPE)
    }

    /// A primary expression with its postfix operators (call, index, member).
    pub(crate) fn postfix_expr(&mut self) -> PResult<Expr> {
        let e = self.primary()?;
        self.postfix_ops(e)
    }

    // ----- Pratt core ----------------------------------------------------------

    pub(crate) fn expr_bp(&mut self, min: u8) -> PResult<Expr> {
        self.enter()?;
        let r = self.expr_bp_inner(min);
        self.leave();
        r
    }

    fn expr_bp_inner(&mut self, min: u8) -> PResult<Expr> {
        let mut lhs = self.prefix()?;
        loop {
            // Fluent pipelines continue on the next line: `xs\n  |> map(f)`.
            if self.at_newline() && self.peek_past_newlines().is_op("|>") && min <= bp::PIPE {
                self.skip_newlines();
            }
            let t = self.peek().clone();
            let word = t.ident().map(str::to_ascii_lowercase);
            let word = word.as_deref();
            let start = lhs.span;

            // Haskell `f $ x`: lowest-precedence, right-associative application.
            if t.is_op("$") && min <= bp::PIPE {
                self.bump();
                self.skip_newlines();
                let arg = self.expr_bp(bp::PIPE)?;
                let span = start.to(arg.span);
                lhs = match lhs.kind {
                    // A partial application `filter even $ xs` supplies the
                    // last argument: `filter(even, xs)`.
                    ExprKind::Call { callee, mut args }
                        if matches!(callee.kind, ExprKind::Name(_) | ExprKind::Member { .. }) =>
                    {
                        args.push(Arg::Pos(arg));
                        Expr::new(ExprKind::Call { callee, args }, span)
                    }
                    kind => Expr::new(
                        ExprKind::Call {
                            callee: Box::new(Expr::new(kind, lhs.span)),
                            args: vec![Arg::Pos(arg)],
                        },
                        span,
                    ),
                };
                continue;
            }
            // `|>`
            if t.is_op("|>") && min <= bp::PIPE {
                self.bump();
                self.skip_newlines();
                let rhs = self.expr_bp(bp::PIPE + 1)?;
                let span = start.to(rhs.span);
                lhs = Expr::new(
                    ExprKind::Binary(BinOp::Pipe, Box::new(lhs), Box::new(rhs)),
                    span,
                );
                continue;
            }
            // Ternaries: `c ? a : b` and Python's `a if c else b`.
            if t.is_op("?") && min <= bp::TERNARY && t.space_before {
                self.bump();
                self.skip_newlines();
                let then = self.nested(|p| p.expr())?;
                self.skip_newlines();
                self.expect_op(":", "in the `? :` expression")?;
                self.skip_newlines();
                let otherwise = self.expr_bp(bp::TERNARY)?;
                let span = start.to(otherwise.span);
                lhs = self.mk_if(
                    lhs,
                    Block::of_expr(then),
                    Some(Block::of_expr(otherwise)),
                    span,
                );
                continue;
            }
            if word == Some("if") && min <= bp::TERNARY && self.same_line() && !self.eq_is_cmp {
                let snap = self.snapshot();
                self.bump();
                let parsed = (|| -> PResult<(Expr, Expr)> {
                    let cond = self.expr_bp(bp::TERNARY + 1)?;
                    if !self.eat_word("else") {
                        return Err(());
                    }
                    let otherwise = self.expr_bp(bp::TERNARY)?;
                    Ok((cond, otherwise))
                })();
                match parsed {
                    Ok((cond, otherwise)) => {
                        let span = start.to(otherwise.span);
                        lhs = self.mk_if(
                            cond,
                            Block::of_expr(lhs),
                            Some(Block::of_expr(otherwise)),
                            span,
                        );
                        continue;
                    }
                    Err(()) => {
                        self.restore(snap);
                        break;
                    }
                }
            }
            if (t.is_op("??") || t.is_op("?:")) && min <= bp::COALESCE {
                self.bump();
                self.skip_newlines();
                let rhs = self.expr_bp(bp::COALESCE + 1)?;
                let span = start.to(rhs.span);
                lhs = Expr::new(ExprKind::Coalesce(Box::new(lhs), Box::new(rhs)), span);
                continue;
            }
            if (t.is_op("||") || word == Some("or")) && min <= bp::OR {
                self.bump();
                self.skip_newlines();
                let rhs = self.expr_bp(bp::OR + 1)?;
                let span = start.to(rhs.span);
                lhs = Expr::new(ExprKind::Or(Box::new(lhs), Box::new(rhs)), span);
                continue;
            }
            if (t.is_op("&&") || word == Some("and")) && min <= bp::AND {
                self.bump();
                self.skip_newlines();
                let rhs = self.expr_bp(bp::AND + 1)?;
                let span = start.to(rhs.span);
                lhs = Expr::new(ExprKind::And(Box::new(lhs), Box::new(rhs)), span);
                continue;
            }
            if min <= bp::COMPARE
                && let Some(e) = self.comparison(lhs.clone())?
            {
                lhs = e;
                continue;
            }
            // Haskell backtick infix: `a `div` b` is `div(a, b)`.
            if let Some(w) = t.ident()
                && w.starts_with('`')
                && w.ends_with('`')
                && w.len() > 2
                && min <= bp::MUL
            {
                let f = Sym::new(&w[1..w.len() - 1]);
                self.bump();
                let rhs = self.expr_bp(bp::MUL + 1)?;
                let span = start.to(rhs.span);
                let callee = Expr::new(ExprKind::Name(f), t.span);
                lhs = Expr::new(
                    ExprKind::Call {
                        callee: Box::new(callee),
                        args: vec![Arg::Pos(lhs), Arg::Pos(rhs)],
                    },
                    span,
                );
                continue;
            }
            if let Some((op, lbp, rbp)) = self.binary_op() {
                if lbp < min {
                    break;
                }
                let tok = self.bump();
                if tok.is_word("divided") || tok.is_word("multiplied") {
                    self.eat_word("by");
                }
                self.skip_newlines();
                let rhs = self.expr_bp(rbp)?;
                let span = start.to(rhs.span);
                lhs = Expr::new(ExprKind::Binary(op, Box::new(lhs), Box::new(rhs)), span);
                continue;
            }
            // Lua concatenation is written with spaces: `name .. "!"`.
            if min <= bp::RANGE
                && t.is_op("..")
                && t.space_before
                && self.nth(1).space_before
                && !self.nth_is_op(1, "<")
            {
                self.bump();
                let rhs = self.expr_bp(bp::RANGE)?;
                let span = start.to(rhs.span);
                lhs = Expr::new(
                    ExprKind::Binary(BinOp::Concat, Box::new(lhs), Box::new(rhs)),
                    span,
                );
                continue;
            }
            if min <= bp::RANGE && self.at_range_op() {
                lhs = self.range_rest(lhs)?;
                continue;
            }
            break;
        }
        Ok(lhs)
    }

    /// Whether `word` is an infix word operator followed by an operand.
    fn word_op_ahead(&self) -> bool {
        self.same_line() && starts_expr(self.nth(1))
    }

    fn binary_op(&self) -> Option<(BinOp, u8, u8)> {
        let t = self.peek();
        if let Tok::Op(op) = t.tok {
            let r = match op {
                "|" if !self.no_bitor => (BinOp::BitOr, bp::BITOR, bp::BITOR + 1),
                "&" => (BinOp::BitAnd, bp::BITAND, bp::BITAND + 1),
                "<<" => (BinOp::Shl, bp::SHIFT, bp::SHIFT + 1),
                ">>" => (BinOp::Shr, bp::SHIFT, bp::SHIFT + 1),
                "+" => (BinOp::Add, bp::ADD, bp::ADD + 1),
                "-" => (BinOp::Sub, bp::ADD, bp::ADD + 1),
                "++" if t.space_before => (BinOp::Concat, bp::ADD, bp::ADD + 1),
                "*" => (BinOp::Mul, bp::MUL, bp::MUL + 1),
                "/" => (BinOp::Div, bp::MUL, bp::MUL + 1),
                "%" => (BinOp::Mod, bp::MUL, bp::MUL + 1),
                "//" => (BinOp::FloorDiv, bp::MUL, bp::MUL + 1),
                "@" => (BinOp::MatMul, bp::MUL, bp::MUL + 1),
                "**" | "^" => (BinOp::Pow, bp::POW, bp::UNARY),
                _ => return None,
            };
            return Some(r);
        }
        let w = t.ident()?.to_ascii_lowercase();
        if !self.word_op_ahead() && !matches!(w.as_str(), "divided" | "multiplied") {
            return None;
        }
        Some(match w.as_str() {
            "xor" => (BinOp::BitXor, bp::BITXOR, bp::BITXOR + 1),
            "plus" => (BinOp::Add, bp::ADD, bp::ADD + 1),
            "minus" => (BinOp::Sub, bp::ADD, bp::ADD + 1),
            "times" => (BinOp::Mul, bp::MUL, bp::MUL + 1),
            "multiplied" if self.nth_is_word(1, "by") => (BinOp::Mul, bp::MUL, bp::MUL + 1),
            "divided" if self.nth_is_word(1, "by") => (BinOp::Div, bp::MUL, bp::MUL + 1),
            "over" => (BinOp::Div, bp::MUL, bp::MUL + 1),
            "mod" | "modulo" => (BinOp::Mod, bp::MUL, bp::MUL + 1),
            "div" => (BinOp::FloorDiv, bp::MUL, bp::MUL + 1),
            _ => return None,
        })
    }

    fn at_range_op(&self) -> bool {
        let t = self.peek();
        if t.is_op("..") || t.is_op("..=") || t.is_op("...") {
            return true;
        }
        matches!(t.ident(), Some("to" | "until" | "downto" | "through")) && self.word_op_ahead()
    }

    fn range_rest(&mut self, lhs: Expr) -> PResult<Expr> {
        let t = self.bump();
        let mut inclusive = match &t.tok {
            Tok::Op("..") => !self.rust_ranges,
            Tok::Op("..=") | Tok::Op("...") => true,
            Tok::Ident(w) => w != "until",
            _ => false,
        };
        let down = t.is_word("downto");
        if t.is_op("..") && self.at_op("<") {
            // Swift `..<`.
            self.bump();
            inclusive = false;
        }
        // Open-ended range in a slice/index: `xs[2..]`.
        let end = if starts_expr(self.peek()) && !self.at_op("]") {
            Some(Box::new(self.expr_bp(bp::RANGE + 1)?))
        } else {
            None
        };
        let mut step = if self.at_word("by") || self.at_word("step") {
            self.bump();
            Some(Box::new(self.expr_bp(bp::RANGE + 1)?))
        } else {
            None
        };
        if down && step.is_none() {
            step = Some(Box::new(Expr::new(ExprKind::Int(-1), t.span)));
        }
        let span = self.span_from(lhs.span);
        Ok(Expr::new(
            ExprKind::Range {
                start: Some(Box::new(lhs)),
                end,
                inclusive,
                step,
            },
            span,
        ))
    }

    /// Parses a comparison chain starting at `lhs`, if one follows.
    fn comparison(&mut self, lhs: Expr) -> PResult<Option<Expr>> {
        let mut rest: Vec<(CmpOp, Expr)> = Vec::new();
        let start = lhs.span;
        let mut lhs_opt = Some(lhs);
        while let Some(op) = self.cmp_op()? {
            let rhs = match op {
                CmpOpOrSpecial::Op(op) => {
                    self.skip_newlines();
                    let rhs = self.expr_bp(bp::COMPARE + 1)?;
                    rest.push((op, rhs));
                    continue;
                }
                CmpOpOrSpecial::Predicate(name) => {
                    let subject = self.chain_value(&mut lhs_opt, &rest);
                    let span = self.span_from(start);
                    Expr::new(
                        ExprKind::Call {
                            callee: Box::new(self.sym_expr(&name, span)),
                            args: vec![Arg::Pos(subject)],
                        },
                        span,
                    )
                }
                CmpOpOrSpecial::Between => {
                    let lo = self.expr_bp(bp::COMPARE + 1)?;
                    if !(self.eat_word("and") || self.eat_word("AND") || self.eat_op("&&")) {
                        let span = self.peek().span;
                        return self.err(span, "expected `and` in `between … and …`");
                    }
                    let hi = self.expr_bp(bp::COMPARE + 1)?;
                    let subject = self.chain_value(&mut lhs_opt, &rest);
                    let span = self.span_from(start);
                    Expr::new(
                        ExprKind::Compare(
                            Box::new(lo),
                            vec![(CmpOp::Le, subject), (CmpOp::Le, hi)],
                        ),
                        span,
                    )
                }
                CmpOpOrSpecial::DivisibleBy => {
                    let d = self.expr_bp(bp::COMPARE + 1)?;
                    let subject = self.chain_value(&mut lhs_opt, &rest);
                    let span = self.span_from(start);
                    let m = Expr::new(
                        ExprKind::Binary(BinOp::Mod, Box::new(subject), Box::new(d)),
                        span,
                    );
                    Expr::new(
                        ExprKind::Compare(
                            Box::new(m),
                            vec![(CmpOp::Eq, Expr::new(ExprKind::Int(0), span))],
                        ),
                        span,
                    )
                }
                CmpOpOrSpecial::Like(negate) => {
                    let pat = self.expr_bp(bp::COMPARE + 1)?;
                    let subject = self.chain_value(&mut lhs_opt, &rest);
                    let span = self.span_from(start);
                    let call = self.call_expr("__like", vec![subject, pat], span);
                    if negate {
                        Expr::new(ExprKind::Unary(UnOp::Not, Box::new(call)), span)
                    } else {
                        call
                    }
                }
                CmpOpOrSpecial::Contains => {
                    let needle = self.expr_bp(bp::COMPARE + 1)?;
                    let subject = self.chain_value(&mut lhs_opt, &rest);
                    let span = self.span_from(start);
                    Expr::new(
                        ExprKind::Compare(Box::new(needle), vec![(CmpOp::In, subject)]),
                        span,
                    )
                }
            };
            return Ok(Some(rhs));
        }
        if rest.is_empty() {
            return Ok(None);
        }
        let lhs = lhs_opt.ok_or(())?;
        let span = self.span_from(start);
        Ok(Some(Expr::new(
            ExprKind::Compare(Box::new(lhs), rest),
            span,
        )))
    }

    /// Collapses a partially parsed chain into one value (for predicates
    /// that follow a comparison, which is rare).
    fn chain_value(&self, lhs: &mut Option<Expr>, rest: &[(CmpOp, Expr)]) -> Expr {
        let base = lhs
            .take()
            .unwrap_or(Expr::new(ExprKind::Nil, Span::default()));
        if rest.is_empty() {
            return base;
        }
        let span = base.span.to(rest[rest.len() - 1].1.span);
        Expr::new(ExprKind::Compare(Box::new(base), rest.to_vec()), span)
    }

    fn cmp_op(&mut self) -> PResult<Option<CmpOpOrSpecial>> {
        let t = self.peek().clone();
        if let Tok::Op(op) = t.tok {
            let op = match op {
                "==" => CmpOp::Eq,
                "=" if self.eq_is_cmp => CmpOp::Eq,
                "!=" => CmpOp::Ne,
                "===" => CmpOp::StrictEq,
                "!==" => CmpOp::StrictNe,
                "<=" => CmpOp::Le,
                ">=" => CmpOp::Ge,
                ">" => CmpOp::Gt,
                "<=>" => CmpOp::Spaceship,
                "≈" => CmpOp::Approx,
                "∉" => CmpOp::NotIn,
                "<" => {
                    if self.nth_is_op(1, ">") && !self.nth(1).space_before {
                        // SQL `<>`.
                        self.bump();
                        CmpOp::Ne
                    } else {
                        CmpOp::Lt
                    }
                }
                "<-" => {
                    // `a <- b` inside an expression is `a < -b`; the R
                    // assignment form only exists at statement start.
                    self.split_arrow_token();
                    CmpOp::Lt
                }
                _ => return Ok(None),
            };
            self.bump();
            return Ok(Some(CmpOpOrSpecial::Op(op)));
        }
        let Some(w) = t.ident().map(str::to_ascii_lowercase) else {
            return Ok(None);
        };
        if !self.same_line() {
            return Ok(None);
        }
        match w.as_str() {
            "in" if !self.no_in_context() => {
                self.bump();
                Ok(Some(CmpOpOrSpecial::Op(CmpOp::In)))
            }
            "not"
                if self
                    .nth(1)
                    .ident()
                    .is_some_and(|n| n.eq_ignore_ascii_case("in")) =>
            {
                self.bump();
                self.bump();
                Ok(Some(CmpOpOrSpecial::Op(CmpOp::NotIn)))
            }
            "not"
                if self
                    .nth(1)
                    .ident()
                    .is_some_and(|n| n.eq_ignore_ascii_case("like")) =>
            {
                self.bump();
                self.bump();
                Ok(Some(CmpOpOrSpecial::Like(true)))
            }
            "not"
                if self
                    .nth(1)
                    .ident()
                    .is_some_and(|n| n.eq_ignore_ascii_case("between")) =>
            {
                // Rare; treat as negated between via predicate fallback.
                Ok(None)
            }
            "equals" if self.word_op_ahead() => {
                self.bump();
                Ok(Some(CmpOpOrSpecial::Op(CmpOp::Eq)))
            }
            "like" if self.word_op_ahead() => {
                self.bump();
                Ok(Some(CmpOpOrSpecial::Like(false)))
            }
            "between" if self.word_op_ahead() => {
                self.bump();
                Ok(Some(CmpOpOrSpecial::Between))
            }
            "contains" if self.word_op_ahead() => {
                self.bump();
                Ok(Some(CmpOpOrSpecial::Contains))
            }
            "does" if self.nth_is_word(1, "not") && self.nth_is_word(2, "equal") => {
                self.bump();
                self.bump();
                self.bump();
                Ok(Some(CmpOpOrSpecial::Op(CmpOp::Ne)))
            }
            "is" => self.is_phrase(),
            _ => Ok(None),
        }
    }

    fn no_in_context(&self) -> bool {
        false
    }

    /// Everything that can follow `is`.
    fn is_phrase(&mut self) -> PResult<Option<CmpOpOrSpecial>> {
        let w = |p: &Self, n: usize| p.nth(n).ident().map(str::to_ascii_lowercase);
        let n1 = w(self, 1);
        let n2 = w(self, 2);
        let n3 = w(self, 3);
        let take = |p: &mut Self, count: usize| {
            for _ in 0..count {
                p.bump();
            }
        };
        let op = |o| Ok(Some(CmpOpOrSpecial::Op(o)));
        match (n1.as_deref(), n2.as_deref(), n3.as_deref()) {
            (Some("not"), Some("in"), _) => {
                take(self, 3);
                op(CmpOp::NotIn)
            }
            (Some("not"), Some("equal"), _) => {
                take(self, 3);
                self.eat_word("to");
                op(CmpOp::Ne)
            }
            (Some("not"), _, _) => {
                take(self, 2);
                op(CmpOp::IsNot)
            }
            (Some("greater" | "more" | "bigger" | "larger"), Some("than"), Some("or")) => {
                take(self, 4);
                self.eat_word("equal");
                self.eat_word("to");
                op(CmpOp::Ge)
            }
            (Some("greater" | "more" | "bigger" | "larger"), Some("than"), _) => {
                take(self, 3);
                op(CmpOp::Gt)
            }
            (Some("less" | "smaller" | "fewer"), Some("than"), Some("or")) => {
                take(self, 4);
                self.eat_word("equal");
                self.eat_word("to");
                op(CmpOp::Le)
            }
            (Some("less" | "smaller" | "fewer"), Some("than"), _) => {
                take(self, 3);
                op(CmpOp::Lt)
            }
            (Some("at"), Some("least"), _) => {
                take(self, 3);
                op(CmpOp::Ge)
            }
            (Some("at"), Some("most"), _) => {
                take(self, 3);
                op(CmpOp::Le)
            }
            (Some("equal"), Some("to"), _) => {
                take(self, 3);
                op(CmpOp::Eq)
            }
            (Some("in"), _, _) => {
                take(self, 2);
                op(CmpOp::In)
            }
            (Some("a" | "an"), _, _) => {
                take(self, 2);
                op(CmpOp::Is)
            }
            (Some("divisible"), Some("by"), _) => {
                take(self, 3);
                Ok(Some(CmpOpOrSpecial::DivisibleBy))
            }
            (Some("between"), _, _) => {
                take(self, 2);
                Ok(Some(CmpOpOrSpecial::Between))
            }
            (
                Some(
                    p @ ("even" | "odd" | "empty" | "positive" | "negative" | "zero" | "prime"
                    | "blank"),
                ),
                _,
                _,
            ) if !starts_expr(self.nth(2)) || self.nth(2).line != self.peek().line => {
                let name = format!("is_{p}");
                take(self, 2);
                Ok(Some(CmpOpOrSpecial::Predicate(name)))
            }
            _ => {
                self.bump();
                op(CmpOp::Is)
            }
        }
    }

    // ----- prefix and primary -------------------------------------------------

    fn prefix(&mut self) -> PResult<Expr> {
        let t = self.peek().clone();
        let start = t.span;
        let unary = |p: &mut Self, op: UnOp, min: u8| -> PResult<Expr> {
            p.bump();
            let operand = p.expr_bp(min)?;
            let span = start.to(operand.span);
            Ok(Expr::new(ExprKind::Unary(op, Box::new(operand)), span))
        };
        match &t.tok {
            Tok::Op("-") => return unary(self, UnOp::Neg, bp::UNARY),
            Tok::Op("+") => return unary(self, UnOp::Pos, bp::UNARY),
            Tok::Op("!") => return unary(self, UnOp::Not, bp::UNARY),
            Tok::Op("~") => return unary(self, UnOp::BitNot, bp::UNARY),
            Tok::Op("√") => return unary(self, UnOp::Sqrt, bp::UNARY),
            Tok::Op("&") => {
                self.bump();
                self.eat_word("mut");
                return self.expr_bp(bp::UNARY);
            }
            Tok::Op("++") | Tok::Op("--") => {
                let delta = if t.is_op("++") { 1 } else { -1 };
                self.bump();
                let target = self.expr_bp(bp::UNARY)?;
                let span = start.to(target.span);
                return Ok(Expr::new(
                    ExprKind::IncDec {
                        target: Box::new(target),
                        delta,
                        prefix: true,
                    },
                    span,
                ));
            }
            Tok::Op("∇") => {
                // `∇f(x, y)` is `(∇f)(x, y)`: the operator binds to the
                // function, then the call applies.
                self.bump();
                let f = self.primary()?;
                let span = start.to(f.span);
                let g = Expr::new(ExprKind::Gradient(Box::new(f)), span);
                return self.postfix_ops(g);
            }
            Tok::Ident(w)
                if w.eq_ignore_ascii_case("not") && !self.nth_is_op(1, "(") || w == "not" =>
            {
                return unary(self, UnOp::Not, bp::NOT);
            }
            Tok::Op("#") => {
                // Lua length: `#t` → `len(t)`.
                let start = self.bump().span;
                let operand = self.expr_bp(bp::UNARY)?;
                let span = start.to(operand.span);
                return Ok(self.call_expr("len", vec![operand], span));
            }
            Tok::Ident(w) if w == "typeof" && starts_expr(self.nth(1)) => {
                return unary(self, UnOp::TypeOf, bp::UNARY);
            }
            Tok::Ident(w) if w == "await" && starts_expr(self.nth(1)) => {
                self.bump();
                return self.expr_bp(bp::UNARY);
            }
            Tok::Ident(w) if w == "yield" && !self.nth_is_op(1, "=") && !self.nth_is_op(1, ".") => {
                let start = self.bump().span;
                self.yields.borrow_mut().push(start.start);
                let mut delegate = false;
                if self.at_word("from") && starts_expr(self.nth(1)) || self.at_op("*") {
                    self.bump();
                    delegate = true;
                }
                let mut values = Vec::new();
                let parens = self.at_op("(") && !self.peek().space_before;
                if starts_expr(self.peek()) && self.same_line() {
                    if parens {
                        // Ruby `yield(a, b)`.
                        self.bump();
                        while !self.at_op(")") && !self.at_eof() {
                            values.push(self.expr()?);
                            if !self.eat_op(",") {
                                break;
                            }
                        }
                        self.expect_op(")", "to close `yield(`")?;
                    } else {
                        values.push(self.expr()?);
                        while self.at_op(",") && self.same_line() && starts_expr(self.nth(1)) {
                            self.bump();
                            values.push(self.expr()?);
                        }
                    }
                }
                return Ok(Expr::new(
                    ExprKind::Yield { values, delegate },
                    self.span_from(start),
                ));
            }
            _ => {}
        }
        let e = self.primary()?;
        let numeric = matches!(
            e.kind,
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::BigInt(_)
        );
        let mut e = self.postfix_ops(e)?;
        // Implicit multiplication: `2x`, `3(a + b)`, `2π`.
        if numeric {
            let n = self.peek();
            let implicit = !n.space_before
                && (n.is_op("(")
                    || n.is_op("√")
                    || n.ident().is_some_and(|w| !is_operator_word(w)));
            if implicit {
                let rhs = self.expr_bp(bp::UNARY)?;
                let span = e.span.to(rhs.span);
                e = Expr::new(
                    ExprKind::Binary(BinOp::Mul, Box::new(e), Box::new(rhs)),
                    span,
                );
            }
        }
        Ok(e)
    }

    fn primary(&mut self) -> PResult<Expr> {
        self.enter()?;
        let r = self.primary_inner();
        self.leave();
        r
    }

    fn primary_inner(&mut self) -> PResult<Expr> {
        let t = self.peek().clone();
        let start = t.span;
        match &t.tok {
            Tok::Int(v) => {
                self.bump();
                Ok(Expr::new(ExprKind::Int(*v), start))
            }
            Tok::BigInt(s) => {
                self.bump();
                let big = BigInt::parse(s).unwrap_or_default();
                Ok(Expr::new(ExprKind::BigInt(Rc::new(big)), start))
            }
            Tok::Float(v) => {
                self.bump();
                Ok(Expr::new(ExprKind::Float(*v), start))
            }
            Tok::Str(s) => {
                self.bump();
                self.string_literal(s, start)
            }
            Tok::IVar(name) => {
                self.bump();
                let span = start;
                Ok(Expr::new(
                    ExprKind::Member {
                        object: Box::new(Expr::new(ExprKind::SelfRef, span)),
                        name: Sym::new(name),
                        optional: false,
                    },
                    span,
                ))
            }
            Tok::Ident(w) => {
                let w = w.clone();
                self.word_primary(&w, start)
            }
            Tok::Op("(") => self.paren_expr(),
            Tok::Op("[") if self.go_slice_type_ahead() => self.go_composite(),
            Tok::Op("[") => self.list_literal(),
            Tok::Op("{") => self.brace_expr(),
            Tok::Op("\\") => self.haskell_lambda(),
            Tok::Op("|") | Tok::Op("||") => self.closure_lambda(),
            Tok::Op("->") => {
                // Ruby stabby lambda: `->(x) { … }`, `-> { … }`.
                let h = self.header();
                self.bump();
                let params = if self.at_op("(") {
                    self.params()?.0
                } else {
                    Vec::new()
                };
                let body = self.body(h, true, terms::BASE)?;
                self.maybe_end(h);
                Ok(self.lambda(params, body, self.span_from(start)))
            }
            Tok::Op(":")
                if matches!(self.nth(1).tok, Tok::Op(op) if ["+", "-", "*", "/", "%", "**", "&", "|", "^", "<<", ">>"].contains(&op))
                    && !self.nth(1).space_before
                    && (self.nth_is_op(2, ")") || self.nth_is_op(2, ",")) =>
            {
                // Ruby operator symbol `:+`.
                self.bump();
                let op = self.bump();
                let Tok::Op(o) = op.tok else { return Err(()) };
                Ok(Expr::new(ExprKind::Str(o.into()), self.span_from(start)))
            }
            Tok::Op(":") if self.nth(1).ident().is_some() && !self.nth(1).space_before => {
                // Ruby symbol `:name`.
                self.bump();
                let (name, _) = self.expect_ident("")?;
                Ok(Expr::new(
                    ExprKind::Str(name.as_str()),
                    self.span_from(start),
                ))
            }
            Tok::Op(".") if self.nth(1).ident().is_some() && !self.nth(1).space_before => {
                // Swift implicit member `.north`.
                self.bump();
                let (name, span) = self.expect_ident("")?;
                Ok(self.call_expr(
                    "__variant_named",
                    vec![Expr::new(ExprKind::Str(name.as_str()), span)],
                    self.span_from(start),
                ))
            }
            Tok::Op("...") | Tok::Op("*") => {
                self.bump();
                let inner = self.expr_bp(bp::UNARY)?;
                let span = start.to(inner.span);
                Ok(Expr::new(ExprKind::Spread(Box::new(inner)), span))
            }
            Tok::Op("**") => {
                // `**kwargs` in a call is handled by call_args; elsewhere it is
                // a dictionary spread.
                self.bump();
                let inner = self.expr_bp(bp::UNARY)?;
                let span = start.to(inner.span);
                Ok(Expr::new(ExprKind::Spread(Box::new(inner)), span))
            }
            _ => {
                let found = describe(&t);
                self.err(start, format!("expected an expression, found {found}"))
            }
        }
    }

    fn word_primary(&mut self, w: &str, start: Span) -> PResult<Expr> {
        let lower = w.to_ascii_lowercase();
        // Single-parameter arrow lambdas: `x => x * 2`, `x -> x * 2`.
        if (self.nth_is_op(1, "=>") || (self.nth_is_op(1, "->") && !self.no_arrow_lambda))
            && !is_reserved(w)
        {
            let (name, _) = self.expect_ident("")?;
            self.bump();
            self.skip_newlines();
            let body = if self.at_op("{") {
                self.brace_block()?
            } else {
                Block::of_expr(self.nested(|p| p.expr())?)
            };
            return Ok(self.lambda(vec![simple_param(name)], body, self.span_from(start)));
        }
        let lit = |p: &mut Self, kind: ExprKind| -> PResult<Expr> {
            p.bump();
            Ok(Expr::new(kind, start))
        };
        match w {
            "true" | "True" | "TRUE" => return lit(self, ExprKind::Bool(true)),
            "false" | "False" | "FALSE" => return lit(self, ExprKind::Bool(false)),
            "nil" | "null" | "None" | "NULL" | "undefined" | "Nothing" | "nullptr" | "NIL" => {
                return lit(self, ExprKind::Nil);
            }
            "self" | "this" | "me" | "Self" if !self.nth_is_op(1, "=>") => {
                if w == "Self" {
                    return lit(self, ExprKind::Name(Sym::new("Self")));
                }
                return lit(self, ExprKind::SelfRef);
            }
            "super" => return lit(self, ExprKind::Super),
            "fn" | "function" | "func" | "fun" | "lambda" | "proc" | "def"
                if !self.nth_is_op(1, "=") =>
            {
                return self.keyword_lambda();
            }
            "if" | "unless" => return self.if_expr(),
            "match" | "when"
                if !self.nth_is_op(1, "=")
                    && !(self.nth_is_op(1, "(") && !self.nth(1).space_before && w == "match")
                    && !self.nth_is_op(1, ".") =>
            {
                return self.match_expr();
            }
            "case" if !self.nth_is_op(1, "=") => return self.match_expr(),
            "do" | "begin" if !self.nth_is_op(1, "=") => {
                let h = self.header();
                self.bump();
                let body = if self.at_op("{") {
                    self.brace_block()?
                } else {
                    self.body(h, false, terms::BASE)?
                };
                self.maybe_end(h);
                return Ok(Expr::new(ExprKind::Block(body), self.span_from(start)));
            }
            "new" if self.nth(1).ident().is_some() => return self.new_expr(),
            "map"
                if self.nth_is_op(1, "[")
                    && !self.nth(1).space_before
                    && self.go_map_type_ahead() =>
            {
                return self.go_composite();
            }
            "make"
                if self.nth_is_op(1, "(")
                    && (self.nth_is_op(2, "[")
                        || self.nth_is_word(2, "map")
                        || self.nth_is_word(2, "chan")) =>
            {
                // Go: `make([]int, n)`, `make(map[string]int)`.
                self.bump();
                self.bump();
                let ty = self.nested(|p| p.type_expr())?;
                let mut size = None;
                if self.eat_op(",") {
                    size = Some(self.nested(|p| p.expr())?);
                    if self.eat_op(",") {
                        self.nested(|p| p.expr())?; // capacity
                    }
                }
                self.expect_op(")", "to close `make(`")?;
                let span = self.span_from(start);
                return Ok(match ty {
                    TypeExpr::List(elem) => {
                        let zero = super::stmt::default_for_type(&elem, span);
                        match size {
                            Some(n) => self.call_expr("__array_of", vec![n, zero], span),
                            None => Expr::new(ExprKind::List(Vec::new()), span),
                        }
                    }
                    _ => Expr::new(ExprKind::Map(Vec::new()), span),
                });
            }
            "please" => return self.please_expr(),
            "try" if self.nth_is_op(1, "{") || self.nth_is_op(1, ":") => {
                // Kotlin/Scala: `val x = try { parse(s) } catch (e: Exception) { -1 }`.
                let h = self.header();
                let stmt = self.try_stmt_pub(h)?;
                let span = stmt.span;
                return Ok(Expr::new(ExprKind::Block(Block::new(vec![stmt])), span));
            }
            "train" if self.nth(1).ident().is_some() && self.nth_is_word(2, "on") => {
                // `history = train net on X, y for 100 epochs`
                let stmt = self.no_jux(|p| p.train_stmt())?;
                let span = stmt.span;
                return Ok(Expr::new(ExprKind::Block(Block::new(vec![stmt])), span));
            }
            "vec!" if self.nth_is_op(1, "[") => {
                // Rust: `vec![1, 2, 3]`, `vec![0; n]`.
                self.bump();
                if let Some(e) = self.attempt(|p| {
                    p.bump();
                    let item = p.nested(|p| p.expr())?;
                    p.expect_op(";", "")?;
                    let count = p.nested(|p| p.expr())?;
                    p.expect_op("]", "")?;
                    let span = p.span_from(start);
                    Ok(p.call_expr("__array_of", vec![count, item], span))
                }) {
                    return Ok(e);
                }
                return self.list_literal();
            }
            "let" | "var"
                if self.nth(1).ident().is_some()
                    || self.nth_is_op(1, "(")
                    || self.nth_is_op(1, "[") =>
            {
                // `if let Some(x) = f()`, `while let [a, b] = …`, Swift `if let x = opt`.
                self.bump();
                let pattern = self.nested(|p| {
                    p.no_arrow_lambda = true;
                    p.match_pattern()
                })?;
                self.expect_op("=", "in `let` condition")?;
                let value = self.expr_bp(bp::AND + 1)?;
                let span = self.span_from(start);
                return Ok(Expr::new(
                    ExprKind::Let {
                        pattern: Box::new(pattern),
                        value: Box::new(value),
                    },
                    span,
                ));
            }
            "raise" | "throw" | "return" | "break" | "continue" => {
                self.bump();
                let ends = self.at_newline()
                    || self.at_eof()
                    || [",", "}", ")", "]", ";", ":"].iter().any(|o| self.at_op(o))
                    || self.at_end_word();
                let value = if ends { None } else { Some(self.expr()?) };
                let span = self.span_from(start);
                let kind = match lower.as_str() {
                    "raise" | "throw" => {
                        StmtKind::Throw(value.unwrap_or_else(|| Expr::new(ExprKind::Nil, span)))
                    }
                    "return" => StmtKind::Return(value),
                    "break" => StmtKind::Break,
                    _ => StmtKind::Continue,
                };
                return Ok(Expr::new(
                    ExprKind::Jump(Box::new(Stmt { kind, span })),
                    span,
                ));
            }
            "the" if self.the_phrase_ahead() => return self.the_phrase(),
            _ => {}
        }
        match lower.as_str() {
            "select" if self.sql_select_ahead() => return self.select_query(),
            "from" if self.nth(1).ident().is_some() && self.nth_is_word(2, "in") => {
                return self.linq_query();
            }
            "insert"
                if self
                    .nth(1)
                    .ident()
                    .is_some_and(|n| n.eq_ignore_ascii_case("into")) =>
            {
                return self.insert_query();
            }
            "update"
                if self.nth(1).ident().is_some()
                    && self
                        .nth(2)
                        .ident()
                        .is_some_and(|n| n.eq_ignore_ascii_case("set")) =>
            {
                return self.update_query();
            }
            "delete"
                if self
                    .nth(1)
                    .ident()
                    .is_some_and(|n| n.eq_ignore_ascii_case("from")) =>
            {
                return self.delete_query();
            }
            "tensor" | "matrix" | "vector"
                if self.nth_is_op(1, "[") && self.nth(1).space_before =>
            {
                self.bump();
                let data = self.list_literal()?;
                let span = self.span_from(start);
                return Ok(self.call_expr("tensor", vec![data], span));
            }
            "ai" | "ask" | "claude"
                if matches!(self.nth(1).tok, Tok::Str(_)) && self.nth(1).space_before =>
            {
                self.bump();
                let prompt = self.expr_bp(bp::ADD)?;
                let span = self.span_from(start);
                return Ok(self.call_expr("ai", vec![prompt], span));
            }
            _ => {}
        }
        self.bump();
        if w == "print" && self.kotlin_fun {
            return Ok(Expr::new(ExprKind::Name(Sym::new("print_string")), start));
        }
        Ok(Expr::new(ExprKind::Name(Sym::new(w)), start))
    }

    fn new_expr(&mut self) -> PResult<Expr> {
        let start = self.bump().span;
        let ty = self.type_atom_expr()?;
        // `new int[5]` / `new String[] {"a"}`.
        if self.at_op("[") {
            self.bump();
            let len = if self.at_op("]") {
                None
            } else {
                Some(self.expr()?)
            };
            self.expect_op("]", "")?;
            while self.eat_op("[") {
                self.expect_op("]", "")?;
            }
            if self.at_op("{") {
                let init = self.brace_expr()?;
                return Ok(match init.kind {
                    ExprKind::Set(items) => Expr::new(ExprKind::List(items), init.span),
                    ExprKind::Map(items) if items.is_empty() => {
                        Expr::new(ExprKind::List(Vec::new()), init.span)
                    }
                    _ => init,
                });
            }
            let span = self.span_from(start);
            return Ok(self.call_expr("__array_of", len.into_iter().collect(), span));
        }
        let args = if self.at_op("(") {
            self.call_args()?
        } else {
            Vec::new()
        };
        // C# collection initialisers: `new List<int> { 1, 2 }`,
        // `new Dictionary<string, int> { {"a", 1} }`.
        let struct_like =
            self.nth(1).ident().is_some() && (self.nth_is_op(2, "=") || self.nth_is_op(2, ":"));
        if self.at_op("{") && !self.no_brace && !struct_like && !self.nth_is_op(1, "}") {
            let init = self.brace_expr()?;
            let items = match init.kind {
                ExprKind::Set(items) => items,
                ExprKind::Map(pairs) => pairs
                    .into_iter()
                    .map(|(k, v)| {
                        let span = k.span.to(v.span);
                        Expr::new(ExprKind::Tuple(vec![k, v]), span)
                    })
                    .collect(),
                _ => vec![init],
            };
            let span = self.span_from(start);
            let list = Expr::new(ExprKind::List(items), span);
            return Ok(Expr::new(
                ExprKind::Call {
                    callee: Box::new(ty),
                    args: vec![Arg::Pos(list)],
                },
                span,
            ));
        }
        let mut e = Expr::new(
            ExprKind::Call {
                callee: Box::new(ty),
                args,
            },
            self.span_from(start),
        );
        // C# object initialisers: `new Point { X = 1 }`.
        if self.at_op("{") && !self.no_brace {
            let fields = self.struct_fields()?;
            e = Expr::new(
                ExprKind::StructLit {
                    ty: Box::new(e),
                    fields,
                },
                self.span_from(start),
            );
        }
        Ok(e)
    }

    /// A type name used as a value (`new java.util.ArrayList<>()`).
    fn type_atom_expr(&mut self) -> PResult<Expr> {
        let (mut name, span) = self.expect_ident("after `new`")?;
        let mut e = Expr::new(ExprKind::Name(name), span);
        while (self.at_op(".") || self.at_op("::")) && self.nth(1).ident().is_some() {
            self.bump();
            let (n, s) = self.expect_ident("")?;
            name = n;
            e = Expr::new(
                ExprKind::Member {
                    object: Box::new(e),
                    name,
                    optional: false,
                },
                span.to(s),
            );
        }
        self.skip_generics();
        Ok(e)
    }

    fn please_expr(&mut self) -> PResult<Expr> {
        let start = self.bump().span;
        let from = self.peek().span.start;
        let mut to = from;
        let mut depth = 0i32;
        while !self.at_eof()
            && (depth > 0 || !self.at_newline() && !self.at_op(";") && !self.at_op("}"))
        {
            if self.at_op("(") || self.at_op("[") || self.at_op("{") {
                depth += 1;
            } else if self.at_op(")") || self.at_op("]") || self.at_op("}") {
                depth -= 1;
            }
            to = self.bump().span.end;
        }
        let text = self.src[from..to].trim().to_owned();
        Ok(Expr::new(
            ExprKind::Please(text.into()),
            self.span_from(start),
        ))
    }

    fn the_phrase_ahead(&self) -> bool {
        let mut i = 1;
        while self.nth(i).ident().is_some() && self.nth(i).line == self.peek().line && i < 6 {
            if self.nth_is_word(i, "of") {
                return i > 1;
            }
            i += 1;
        }
        false
    }

    /// `the square root of 16`, `the length of name`, `the sum of xs`.
    fn the_phrase(&mut self) -> PResult<Expr> {
        let start = self.bump().span;
        let mut words = Vec::new();
        while !self.at_word("of") {
            words.push(self.bump().ident().unwrap_or("").to_owned());
        }
        self.bump();
        let arg = self.expr_bp(bp::ADD)?;
        let name = words.join("_");
        let name = match name.as_str() {
            "square_root" => "sqrt",
            "cube_root" => "cbrt",
            "length" | "size" | "number_of_items" | "count" => "len",
            "average" | "mean" => "mean",
            "maximum" | "largest" | "biggest" => "max",
            "minimum" | "smallest" => "min",
            "absolute_value" => "abs",
            "total" => "sum",
            "type" => "type",
            other => other,
        }
        .to_owned();
        let span = self.span_from(start);
        Ok(self.call_expr(&name, vec![arg], span))
    }

    // ----- strings -------------------------------------------------------------

    pub(crate) fn string_literal(&mut self, s: &StrLit, span: Span) -> PResult<Expr> {
        if let Some(text) = s.as_plain() {
            return Ok(Expr::new(ExprKind::Str(text.into()), span));
        }
        let mut parts = Vec::new();
        for part in &s.parts {
            match part {
                StrPart::Text(t) => parts.push(InterpPart::Lit(t.as_str().into())),
                StrPart::Code { offset, len, spec } => {
                    let (expr, errs) =
                        super::parse_expr_range(self.src, self.file, *offset, offset + len);
                    self.errors.extend(errs);
                    let Some(expr) = expr else {
                        return Err(());
                    };
                    parts.push(InterpPart::Expr(expr, spec.as_deref().map(Rc::from)));
                }
            }
        }
        Ok(Expr::new(ExprKind::Interp(parts), span))
    }

    // ----- brackets ------------------------------------------------------------

    fn paren_expr(&mut self) -> PResult<Expr> {
        let start = self.peek().span;
        // Arrow lambdas: `(a, b) => …`, `(x: Int) -> …`.
        if let Some(e) = self.attempt(|p| p.arrow_lambda()) {
            return Ok(e);
        }
        self.bump();
        // Haskell operator sections: `(+)`, `(* 3)`, `(2 ^)`, `(++ "!")`.
        if let Tok::Op(op) = self.peek().tok
            && is_section_op(op)
        {
            if self.nth_is_op(1, ")") {
                self.bump();
                self.bump();
                return Ok(Expr::new(ExprKind::Str(op.into()), self.span_from(start)));
            }
            if op != "-" && op != "!" && op != "~" && op != "*"
                || op == "*" && self.nth(1).space_before
            {
                let rhs = self.nested(|p| {
                    p.bump();
                    let e = p.expr()?;
                    p.expect_op(")", "to close the operator section")?;
                    Ok(e)
                })?;
                let span = self.span_from(start);
                let opx = Expr::new(ExprKind::Str(op.into()), span);
                return Ok(self.call_expr("__section_right", vec![opx, rhs], span));
            }
        }
        if let Some(e) = self.attempt(|p| {
            let lhs = p.nested(|p| p.postfix_expr())?;
            let Tok::Op(op) = p.peek().tok else {
                return Err(());
            };
            if !is_section_op(op) || !p.nth_is_op(1, ")") {
                return Err(());
            }
            p.bump();
            p.bump();
            let span = p.span_from(start);
            let opx = Expr::new(ExprKind::Str(op.into()), span);
            Ok(p.call_expr("__section_left", vec![opx, lhs], span))
        }) {
            return Ok(e);
        }
        self.nested(|p| {
            p.skip_newlines();
            if p.eat_op(")") {
                return Ok(Expr::new(ExprKind::Tuple(Vec::new()), p.span_from(start)));
            }
            let first = p.expr()?;
            p.skip_newlines();
            if p.at_word("for") {
                // Generator expression → list comprehension.
                let comp = p.comprehension_tail(CompKind::List, first, None)?;
                p.expect_op(")", "to close the generator expression")?;
                return Ok(Expr::new(
                    ExprKind::Comprehension(Box::new(comp)),
                    p.span_from(start),
                ));
            }
            if p.eat_op(")") {
                return Ok(first);
            }
            let mut items = vec![first];
            while p.eat_op(",") {
                p.skip_newlines();
                if p.at_op(")") {
                    break;
                }
                items.push(p.expr()?);
                p.skip_newlines();
            }
            p.expect_op(")", "to close the parentheses")?;
            Ok(Expr::new(ExprKind::Tuple(items), p.span_from(start)))
        })
    }

    fn arrow_lambda(&mut self) -> PResult<Expr> {
        let start = self.peek().span;
        let (params, _) = self.params()?;
        if self.at_op(":") && self.nth(1).ident().is_some() {
            self.bump();
            self.type_expr()?;
        }
        if !(self.at_op("=>") || (self.at_op("->") && !self.no_arrow_lambda)) {
            return Err(());
        }
        self.bump();
        self.skip_newlines();
        let body = if self.at_op("{") {
            self.brace_block()?
        } else {
            Block::of_expr(self.nested(|p| p.expr())?)
        };
        Ok(self.lambda(params, body, self.span_from(start)))
    }

    fn list_literal(&mut self) -> PResult<Expr> {
        let start = self.bump().span;
        self.nested(|p| {
            p.skip_newlines();
            if p.eat_op("]") {
                return Ok(Expr::new(ExprKind::List(Vec::new()), p.span_from(start)));
            }
            let haskell = p.haskell_comp_ahead();
            let first = if haskell {
                p.expr_bp(bp::BITOR + 1)?
            } else {
                p.expr()?
            };
            p.skip_newlines();
            if p.at_word("for") {
                let comp = p.comprehension_tail(CompKind::List, first, None)?;
                p.expect_op("]", "to close the list comprehension")?;
                return Ok(Expr::new(
                    ExprKind::Comprehension(Box::new(comp)),
                    p.span_from(start),
                ));
            }
            if haskell && p.at_op("|") {
                p.bump();
                let comp = p.haskell_comprehension(first)?;
                p.expect_op("]", "to close the list comprehension")?;
                return Ok(Expr::new(
                    ExprKind::Comprehension(Box::new(comp)),
                    p.span_from(start),
                ));
            }
            // Rust array repeat: `[0; n]`.
            if p.rust_ranges && p.at_op(";") {
                p.bump();
                let count = p.expr()?;
                p.expect_op("]", "to close the array")?;
                let span = p.span_from(start);
                return Ok(p.call_expr("__array_of", vec![count, first], span));
            }
            let mut items = vec![first];
            // OCaml/F# separate list items with `;`.
            while p.eat_op(",") || p.eat_op(";") {
                p.skip_newlines();
                if p.at_op("]") {
                    break;
                }
                items.push(p.expr()?);
                p.skip_newlines();
            }
            p.expect_op("]", "to close the list")?;
            let span = p.span_from(start);
            Ok(haskell_range_list(items, span))
        })
    }

    /// Whether a Haskell comprehension bar (`[e | x <- xs]`) appears at the
    /// top level of the list starting at the current token.
    fn haskell_comp_ahead(&self) -> bool {
        let toks = self.toks_ref();
        let mut i = self.pos_ref();
        let mut depth = 0;
        while i < toks.len() {
            let t = &toks[i];
            if t.is_op("(") || t.is_op("[") || t.is_op("{") {
                depth += 1;
            } else if t.is_op(")") || t.is_op("]") || t.is_op("}") {
                if depth == 0 {
                    return false;
                }
                depth -= 1;
            } else if depth == 0 && t.is_op("|") {
                let save = i;
                let _ = save;
                return self.generator_after(i);
            }
            i += 1;
        }
        false
    }

    fn generator_after(&self, bar: usize) -> bool {
        let toks = self.toks_ref();
        let mut i = bar + 1;
        let mut depth = 0;
        while i < toks.len() {
            let t = &toks[i];
            if t.is_op("(") || t.is_op("[") || t.is_op("{") {
                depth += 1;
            } else if t.is_op(")") || t.is_op("]") || t.is_op("}") {
                if depth == 0 {
                    return false;
                }
                depth -= 1;
            } else if depth == 0 && t.is_op("<-") {
                return true;
            }
            i += 1;
        }
        false
    }

    /// `x*2 | x <- xs, x > 1` (after the `|`).
    fn haskell_comprehension(&mut self, element: Expr) -> PResult<Comprehension> {
        let mut clauses = Vec::new();
        loop {
            self.skip_newlines();
            if self.at_op("]") {
                break;
            }
            let is_gen = self.attempt(|p| {
                let pat = p.binding_pattern()?;
                if !p.eat_op("<-") {
                    return Err(());
                }
                Ok(pat)
            });
            if let Some(pat) = is_gen {
                let iter = self.expr()?;
                clauses.push(CompClause::For(pat, iter));
            } else {
                clauses.push(CompClause::If(self.expr()?));
            }
            if !self.eat_op(",") {
                break;
            }
        }
        Ok(Comprehension {
            kind: CompKind::List,
            element,
            value: None,
            clauses,
        })
    }

    /// `for x in xs [if c] [for …]` after a comprehension element.
    fn comprehension_tail(
        &mut self,
        kind: CompKind,
        element: Expr,
        value: Option<Expr>,
    ) -> PResult<Comprehension> {
        let mut clauses = Vec::new();
        loop {
            self.skip_newlines();
            if self.eat_word("for") {
                let first = self.binding_pattern()?;
                let pat = if self.at_op(",") {
                    let mut items = vec![first];
                    while self.eat_op(",") {
                        items.push(self.binding_pattern()?);
                    }
                    Pattern::Tuple(items)
                } else {
                    first
                };
                if !(self.eat_word("in") || self.eat_word("of")) {
                    let span = self.peek().span;
                    return self.err(span, "expected `in` in the comprehension");
                }
                let iter = self.expr_bp(bp::OR)?;
                clauses.push(CompClause::For(pat, iter));
            } else if self.at_word("if") || self.at_word("when") {
                self.bump();
                let cond = self.expr_bp(bp::OR)?;
                clauses.push(CompClause::If(cond));
            } else {
                break;
            }
        }
        Ok(Comprehension {
            kind,
            element,
            value,
            clauses,
        })
    }

    /// `{` in expression position: map, set, comprehension, block lambda or
    /// block expression.
    fn brace_expr(&mut self) -> PResult<Expr> {
        let start = self.peek().span;
        if let Some(lambda) = self.attempt(|p| p.brace_lambda(false)) {
            return Ok(lambda);
        }
        if self.brace_looks_like_block() {
            let body = self.brace_block()?;
            return Ok(Expr::new(ExprKind::Block(body), self.span_from(start)));
        }
        self.bump();
        self.nested(|p| {
            p.skip_separators();
            if p.eat_op("}") {
                return Ok(Expr::new(ExprKind::Map(Vec::new()), p.span_from(start)));
            }
            let mut entries: Vec<(Expr, Expr)> = Vec::new();
            let mut set_items: Vec<Expr> = Vec::new();
            loop {
                p.skip_separators();
                if p.at_op("}") {
                    break;
                }
                if p.at_op("...") || p.at_op("**") {
                    let s = p.bump().span;
                    let inner = p.expr()?;
                    let span = s.to(inner.span);
                    entries.push((
                        Expr::new(ExprKind::Spread(Box::new(inner)), span),
                        Expr::new(ExprKind::Nil, span),
                    ));
                } else {
                    let key = p.map_key()?;
                    if p.at_op(":")
                        || p.at_op("=>")
                        || (p.at_op("=") && matches!(key.kind, ExprKind::Str(_)))
                    {
                        p.bump();
                        p.skip_newlines();
                        let value = p.expr()?;
                        if entries.is_empty() && set_items.is_empty() && p.at_word("for") {
                            // `{k: v for k, v in …}`: a bare key is a variable here.
                            let key = match &key.kind {
                                ExprKind::Str(s)
                                    if p.src.get(key.span.start..key.span.end) == Some(&**s) =>
                                {
                                    Expr::new(ExprKind::Name(Sym::new(s)), key.span)
                                }
                                _ => key,
                            };
                            let comp = p.comprehension_tail(CompKind::Map, key, Some(value))?;
                            p.skip_separators();
                            p.expect_op("}", "to close the map comprehension")?;
                            return Ok(Expr::new(
                                ExprKind::Comprehension(Box::new(comp)),
                                p.span_from(start),
                            ));
                        }
                        entries.push((key, value));
                    } else if p.at_word("for") && entries.is_empty() && set_items.is_empty() {
                        let comp = p.comprehension_tail(CompKind::Set, key, None)?;
                        p.skip_separators();
                        p.expect_op("}", "to close the set comprehension")?;
                        return Ok(Expr::new(
                            ExprKind::Comprehension(Box::new(comp)),
                            p.span_from(start),
                        ));
                    } else {
                        set_items.push(key);
                    }
                }
                p.skip_separators();
                if !p.eat_op(",") {
                    break;
                }
            }
            p.skip_separators();
            p.expect_op("}", "to close the braces")?;
            let span = p.span_from(start);
            let all_names = set_items
                .iter()
                .all(|e| matches!(e.kind, ExprKind::Name(_)));
            if !set_items.is_empty() && (entries.is_empty() && !all_names) {
                return Ok(Expr::new(ExprKind::Set(set_items), span));
            }
            // JS shorthand: `{a, b}` is `{a: a, b: b}`.
            for item in set_items {
                let ExprKind::Name(name) = item.kind else {
                    return p.err(item.span, "map entries need `key: value`");
                };
                let key = Expr::new(ExprKind::Str(name.as_str()), item.span);
                entries.push((key, item));
            }
            Ok(Expr::new(ExprKind::Map(entries), span))
        })
    }

    /// A map key: bare names become strings (JS/Ruby style); `[expr]` is a
    /// computed key; anything else is an expression.
    fn map_key(&mut self) -> PResult<Expr> {
        let t = self.peek().clone();
        if let Tok::Ident(w) = &t.tok
            && !matches!(
                w.as_str(),
                "true" | "false" | "nil" | "null" | "None" | "True" | "False"
            )
        {
            if (self.nth_is_op(1, ":") && !self.nth_is_op(1, "::")) || self.nth_is_op(1, "=") {
                self.bump();
                return Ok(Expr::new(ExprKind::Str(w.as_str().into()), t.span));
            }
            if self.nth_is_op(1, ",") || self.nth_is_op(1, "}") || self.nth_is_op(1, "=>") {
                // A variable: JS shorthand `{a, b}`, a set element, or a Ruby
                // `{ key => value }` entry.
                self.bump();
                return Ok(Expr::new(ExprKind::Name(Sym::new(w)), t.span));
            }
        }
        if t.is_op("[") {
            // `[expr]: value` computed key; otherwise a list element.
            if let Some(k) = self.attempt(|p| {
                p.bump();
                let k = p.expr()?;
                p.expect_op("]", "")?;
                if p.at_op(":") { Ok(k) } else { Err(()) }
            }) {
                return Ok(k);
            }
        }
        self.expr_bp(bp::TERNARY)
    }

    /// Heuristic: does `{` start a block expression rather than a map/set?
    fn brace_looks_like_block(&self) -> bool {
        let n1 = self.nth(1);
        let n2 = self.nth(2);
        if n1.is_op("}") {
            return false;
        }
        if matches!(n1.tok, Tok::Newline) {
            // Look at the first real token inside.
            let mut i = self.pos_ref() + 1;
            let toks = self.toks_ref();
            while i < toks.len() && matches!(toks[i].tok, Tok::Newline) {
                i += 1;
            }
            let a = &toks[i];
            let b = toks.get(i + 1);
            return !(b.is_some_and(|b| b.is_op(":") || b.is_op("=>") || b.is_op(","))
                || a.is_op("}")
                || a.is_op("..."));
        }
        if n2.is_op(":") || n2.is_op("=>") || n2.is_op(",") || n1.is_op("...") || n1.is_op("**") {
            return false;
        }
        if n1.is_op("[") || matches!(n1.tok, Tok::Str(_) | Tok::Int(_) | Tok::Float(_)) {
            return false;
        }
        // `{ x }` → set of one; `{ x for … }` → set comprehension.
        if n2.is_op("}") || n2.is_word("for") {
            return false;
        }
        // Lua tables: `{ name = "x", age = 3 }` — entries separated by
        // commas on one line, with no `;` or line break before the `}`.
        if n2.is_op("=") && n1.ident().is_some() {
            let toks = self.toks_ref();
            let mut depth = 0i32;
            let mut i = self.pos_ref() + 3;
            while let Some(t) = toks.get(i) {
                match &t.tok {
                    Tok::Op("(" | "[" | "{") => depth += 1,
                    Tok::Op(")" | "]") => depth -= 1,
                    Tok::Op("}") if depth == 0 => return false,
                    Tok::Op("}") => depth -= 1,
                    Tok::Op(";") | Tok::Newline | Tok::Eof if depth == 0 => return true,
                    _ => {}
                }
                i += 1;
            }
            return true;
        }
        n1.ident()
            .is_some_and(|w| is_reserved(w) || crate::parser::stmt::FN_WORDS.contains(&w))
            || n2.is_op("=")
            || n2.is_op(";")
            || matches!(n2.tok, Tok::Newline)
            || n2.is_op("(")
            || n2.is_op(".")
            || n2.ident().is_some()
    }

    /// `{ |x| … }`, `{ x -> … }`, `{ x, y in … }`, and (for trailing blocks)
    /// `{ … it … }` with an implicit parameter.
    fn brace_lambda(&mut self, implicit_ok: bool) -> PResult<Expr> {
        let start = self.expect_op("{", "")?;
        self.skip_newlines();
        let mut params = Vec::new();
        let mut implicit = false;
        if self.at_op("|") || self.at_op("||") {
            if self.eat_op("||") {
            } else {
                self.bump();
                while !self.at_op("|") {
                    params.push(self.param()?);
                    if !self.eat_op(",") {
                        break;
                    }
                }
                self.expect_op("|", "to close the block parameters")?;
            }
        } else if let Some(ps) = self.attempt(|p| {
            // `x, y ->` (Kotlin) or `x, y in` / `(x, y) in` (Swift).
            let ps = if p.at_op("(") {
                p.params()?.0
            } else {
                let mut ps = Vec::new();
                loop {
                    let (n, _) = p.expect_ident("")?;
                    if p.eat_op(":") {
                        p.type_expr()?;
                    }
                    ps.push(simple_param(n));
                    if !p.eat_op(",") {
                        break;
                    }
                }
                ps
            };
            if p.at_op("->") && !p.nth_is_op(1, "{") || p.at_word("in") {
                p.bump();
                Ok(ps)
            } else {
                Err(())
            }
        }) {
            params = ps;
        } else if implicit_ok {
            implicit = true;
            params = vec![Param {
                pattern: Pattern::Bind(Sym::new("it")),
                default: Some(Expr::new(ExprKind::Nil, start)),
                ty: None,
                kind: ParamKind::Normal,
                label: None,
            }];
        } else {
            return Err(());
        }
        let stmts = self.nested(|p| {
            let mut stmts = Vec::new();
            loop {
                p.skip_separators();
                if p.eat_op("}") {
                    break;
                }
                if p.at_eof() {
                    return p.err(start, "this `{` is never closed");
                }
                match p.statement() {
                    Ok(s) => stmts.push(s),
                    Err(()) => return Err(()),
                }
            }
            Ok(stmts)
        })?;
        let span = self.span_from(start);
        let mut def = self.new_fn(None, params, Block::new(stmts), span);
        def.implicit_params = implicit;
        Ok(Expr::new(ExprKind::Lambda(Rc::new(def)), span))
    }

    // ----- lambdas -----------------------------------------------------------

    /// `fn(x) => …`, `function(x) { … }`, `lambda x: …`, `fun x -> …`,
    /// `func(x int) int { … }`, `function(x) … end`, `proc { |x| … }`.
    fn keyword_lambda(&mut self) -> PResult<Expr> {
        let h = self.header();
        let kw = self.bump();
        let start = kw.span;
        let python = kw.is_word("lambda");
        if kw.is_word("proc") && self.at_op("{") {
            return self.brace_lambda(true);
        }
        let mut name = None;
        if self.peek().ident().is_some() && self.nth_is_op(1, "(") {
            name = Some(self.expect_ident("")?.0);
        }
        let params = if self.at_op("(") {
            self.params()?.0
        } else if self.at_op("|") || self.at_op("||") {
            return self.closure_lambda();
        } else {
            let mut ps = Vec::new();
            while self.peek().ident().is_some() || self.at_op("*") || self.at_op("(") {
                if self.at_any_word(&["do", "then"]) {
                    break;
                }
                if self.at_op("*") || self.at_op("**") {
                    ps.push(self.param()?);
                } else if self.at_op("(") {
                    ps.push(Param {
                        pattern: self.simple_pattern()?,
                        default: None,
                        ty: None,
                        kind: ParamKind::Normal,
                        label: None,
                    });
                } else {
                    let (n, _) = self.expect_ident("")?;
                    let default = if python && self.eat_op("=") {
                        Some(self.expr_bp(bp::TERNARY)?)
                    } else {
                        None
                    };
                    ps.push(Param {
                        pattern: Pattern::Bind(n),
                        default,
                        ty: None,
                        kind: ParamKind::Normal,
                        label: None,
                    });
                }
                if !self.eat_op(",") && python {
                    break;
                }
            }
            ps
        };
        // Return type for `fn(x) -> T { … }` / Go `func(x int) int { … }`.
        if self.at_op("->") && !python {
            if let Some(()) = self.attempt(|p| {
                p.bump();
                p.type_expr()?;
                if p.at_op("{") { Ok(()) } else { Err(()) }
            }) {}
        } else if self
            .peek()
            .ident()
            .is_some_and(|w| !matches!(w, "do" | "then"))
            && self.nth_is_op(1, "{")
        {
            self.bump();
        }
        let body = if python && self.at_op(":") {
            self.bump();
            Block::of_expr(self.expr()?)
        } else if self.at_op("->") || self.at_op("=>") || self.at_op(".") {
            self.bump();
            self.skip_newlines();
            if self.at_op("{") && self.prev().is_op("=>") {
                self.brace_block()?
            } else {
                Block::of_expr(self.nested(|p| p.expr())?)
            }
        } else {
            let b = self.body(h, true, terms::BASE)?;
            self.maybe_end(h);
            b
        };
        let span = self.span_from(start);
        let mut def = self.new_fn(name, params, body, span);
        if def.yields.is_empty() {
            // `function*() { yield … }` expressions are generators too.
            def.yields = self.claim_yields(span);
        }
        Ok(Expr::new(ExprKind::Lambda(Rc::new(def)), span))
    }

    /// `|a, b| expr` / `|| expr` (Rust, Ruby block params).
    fn closure_lambda(&mut self) -> PResult<Expr> {
        let start = self.peek().span;
        let mut params = Vec::new();
        if !self.eat_op("||") {
            self.expect_op("|", "")?;
            let saved = self.no_bitor;
            self.no_bitor = true;
            while !self.at_op("|") && !self.at_eof() {
                match self.param() {
                    Ok(p) => params.push(p),
                    Err(()) => {
                        self.no_bitor = saved;
                        return Err(());
                    }
                }
                if !self.eat_op(",") {
                    break;
                }
            }
            self.no_bitor = saved;
            self.expect_op("|", "to close the closure parameters")?;
        }
        if self.at_op("->") {
            self.bump();
            self.type_expr()?;
        }
        let body = if self.at_op("{") {
            self.brace_block()?
        } else {
            Block::of_expr(self.nested(|p| p.expr())?)
        };
        Ok(self.lambda(params, body, self.span_from(start)))
    }

    /// `\x y -> expr`.
    fn haskell_lambda(&mut self) -> PResult<Expr> {
        let start = self.bump().span;
        let mut params = Vec::new();
        while !self.at_op("->") && !self.at_op(".") && !self.at_eof() {
            params.push(Param {
                pattern: self.simple_pattern()?,
                default: None,
                ty: None,
                kind: ParamKind::Normal,
                label: None,
            });
        }
        self.bump();
        let body = Block::of_expr(self.nested(|p| p.expr())?);
        Ok(self.lambda(params, body, self.span_from(start)))
    }

    // ----- postfix -------------------------------------------------------------

    pub(crate) fn postfix_ops(&mut self, mut e: Expr) -> PResult<Expr> {
        loop {
            // Method chains continued on the next line: `xs\n  .map(f)`.
            if self.at_newline() {
                let n = self.peek_past_newlines();
                if n.line_start && (n.is_op(".") || n.is_op("?.")) && n.col > 0 {
                    self.skip_newlines();
                } else {
                    break;
                }
            }
            let t = self.peek().clone();
            let start = e.span;
            // Lua method calls: `s:upper()`, `("x"):rep(3)`, `obj:method "arg"`.
            if t.is_op(":")
                && !t.space_before
                && !self.in_index
                && self.nth(1).ident().is_some()
                && !self.nth(1).space_before
                && (self.nth_is_op(2, "(") && !self.nth(2).space_before
                    || matches!(self.nth(2).tok, Tok::Str(_)))
                && matches!(
                    e.kind,
                    ExprKind::Name(_)
                        | ExprKind::Str(_)
                        | ExprKind::Member { .. }
                        | ExprKind::Call { .. }
                        | ExprKind::Index { .. }
                )
            {
                self.bump();
                let (name, _) = self.expect_ident("")?;
                let args = if self.at_op("(") {
                    self.call_args()?
                } else {
                    vec![Arg::Pos(self.primary()?)]
                };
                let member = Expr::new(
                    ExprKind::Member {
                        object: Box::new(e),
                        name,
                        optional: false,
                    },
                    self.span_from(start),
                );
                e = Expr::new(
                    ExprKind::Call {
                        callee: Box::new(member),
                        args,
                    },
                    self.span_from(start),
                );
                continue;
            }
            match &t.tok {
                Tok::Op("(")
                    if !t.line_start
                        && (!t.space_before
                            || self.call_with_space_ok(&e) && !self.paren_followed_by_atom())
                        && !matches!(
                            e.kind,
                            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::BigInt(_)
                        ) =>
                {
                    let args = self.call_args()?;
                    e = Expr::new(
                        ExprKind::Call {
                            callee: Box::new(e),
                            args,
                        },
                        self.span_from(start),
                    );
                }
                Tok::Op("[") if !t.space_before && !t.line_start => {
                    self.bump();
                    let index = self.nested(|p| p.index_expr())?;
                    self.expect_op("]", "to close the index")?;
                    e = Expr::new(
                        ExprKind::Index {
                            object: Box::new(e),
                            index: Box::new(index),
                        },
                        self.span_from(start),
                    );
                }
                Tok::Op(".") | Tok::Op("?.") | Tok::Op("::") => {
                    let optional = t.is_op("?.");
                    let n1 = self.nth(1).clone();
                    let name = match &n1.tok {
                        Tok::Ident(w) => Sym::new(w),
                        Tok::Int(i) if !t.is_op("::") => Sym::new(&i.to_string()),
                        Tok::Op("(") if optional => {
                            // `f?.(x)`
                            self.bump();
                            let args = self.call_args()?;
                            e = Expr::new(
                                ExprKind::Call {
                                    callee: Box::new(e),
                                    args,
                                },
                                self.span_from(start),
                            );
                            continue;
                        }
                        Tok::Op("[") if optional => {
                            self.bump();
                            self.bump();
                            let index = self.nested(|p| p.index_expr())?;
                            self.expect_op("]", "")?;
                            e = Expr::new(
                                ExprKind::Index {
                                    object: Box::new(e),
                                    index: Box::new(index),
                                },
                                self.span_from(start),
                            );
                            continue;
                        }
                        Tok::Op("<") if t.is_op("::") => {
                            // Turbofish `::<T>` is ignored.
                            self.bump();
                            self.skip_generics();
                            continue;
                        }
                        _ => break,
                    };
                    self.bump();
                    self.bump();
                    e = Expr::new(
                        ExprKind::Member {
                            object: Box::new(e),
                            name,
                            optional,
                        },
                        self.span_from(start),
                    );
                    // Ruby: `Foo.new` without parentheses constructs.
                    if name.is("new")
                        && !(self.at_op("(") && !self.peek().space_before)
                        && !self.at_op("=")
                    {
                        e = Expr::new(
                            ExprKind::Call {
                                callee: Box::new(e),
                                args: Vec::new(),
                            },
                            self.span_from(start),
                        );
                    }
                }
                Tok::Op("!")
                    if !t.space_before && !self.nth_is_op(1, "=") && !self.nth_is_op(1, "!") =>
                {
                    self.bump();
                    e = Expr::new(
                        ExprKind::Unary(UnOp::Bang, Box::new(e)),
                        self.span_from(start),
                    );
                }
                Tok::Op("!") if !t.space_before && self.nth_is_op(1, "!") => {
                    // Kotlin `!!`: assert non-null.
                    self.bump();
                    self.bump();
                    e = Expr::new(
                        ExprKind::Unary(UnOp::Bang, Box::new(e)),
                        self.span_from(start),
                    );
                }
                Tok::Op("++") | Tok::Op("--") if !t.space_before && e.is_assignable() => {
                    let delta = if t.is_op("++") { 1 } else { -1 };
                    self.bump();
                    e = Expr::new(
                        ExprKind::IncDec {
                            target: Box::new(e),
                            delta,
                            prefix: false,
                        },
                        self.span_from(start),
                    );
                }
                Tok::Op("′") | Tok::Op("″") => {
                    let order = if t.is_op("′") { 1 } else { 2 };
                    self.bump();
                    e = Expr::new(
                        ExprKind::Derivative {
                            func: Box::new(e),
                            order,
                        },
                        self.span_from(start),
                    );
                }
                Tok::Op("²") | Tok::Op("³") => {
                    let p = if t.is_op("²") { 2 } else { 3 };
                    self.bump();
                    let span = self.span_from(start);
                    e = Expr::new(
                        ExprKind::Binary(
                            BinOp::Pow,
                            Box::new(e),
                            Box::new(Expr::new(ExprKind::Int(p), span)),
                        ),
                        span,
                    );
                }
                Tok::Op("?") if !t.space_before && !starts_expr(self.nth(1)) => {
                    // Rust `?` / Swift optional chaining marker: unwrap-or-return
                    // is approximated as a plain value.
                    self.bump();
                }
                Tok::Op("{") if !self.no_brace && !t.line_start => {
                    if let Some(lit) = self.struct_literal_after(&e)? {
                        e = lit;
                        continue;
                    }
                    if !self.trailing_block_ok(&e) {
                        break;
                    }
                    if self.brace_map_ahead() {
                        let map = self.brace_expr()?;
                        e = append_arg(e, map, self.span_from(start));
                        continue;
                    }
                    let block = self.brace_lambda(true)?;
                    e = append_arg(e, block, self.span_from(start));
                }
                Tok::Ident(w)
                    if w == "do"
                        && !self.no_brace
                        && self.same_line()
                        && self.trailing_block_ok(&e) =>
                {
                    let block = self.do_block()?;
                    e = append_arg(e, block, self.span_from(start));
                }
                Tok::Ident(w) if (w == "squared" || w == "cubed") && self.same_line() => {
                    let p = if w == "squared" { 2 } else { 3 };
                    self.bump();
                    let span = self.span_from(start);
                    e = Expr::new(
                        ExprKind::Binary(
                            BinOp::Pow,
                            Box::new(e),
                            Box::new(Expr::new(ExprKind::Int(p), span)),
                        ),
                        span,
                    );
                }
                Tok::Ident(w) if w == "match" && self.same_line() && self.nth_is_op(1, "{") => {
                    // Scala: `x match { case … }`.
                    self.bump();
                    let arms = self.brace_arms()?;
                    e = Expr::new(
                        ExprKind::Match {
                            subject: Some(Box::new(e)),
                            arms,
                        },
                        self.span_from(start),
                    );
                }
                _ if !self.no_juxtapose
                    && self.same_line()
                    && matches!(e.kind, ExprKind::Name(_) | ExprKind::Member { .. })
                    && !matches!(&e.kind, ExprKind::Name(n) if is_reserved(&n.as_str()))
                    && (atom_start(&t)
                        || t.is_op("(") && t.space_before && self.paren_followed_by_atom()) =>
                {
                    // Haskell/OCaml application by juxtaposition: `fact 5`,
                    // `even x`, `max a b`, `foldr (+) 0 xs`.
                    let mut args = Vec::new();
                    while self.same_line()
                        && (atom_start(self.peek()) || self.at_op("(") && self.peek().space_before)
                    {
                        let saved = self.no_juxtapose;
                        self.no_juxtapose = true;
                        let arg = self.postfix_expr();
                        self.no_juxtapose = saved;
                        args.push(Arg::Pos(arg?));
                    }
                    e = Expr::new(
                        ExprKind::Call {
                            callee: Box::new(e),
                            args,
                        },
                        self.span_from(start),
                    );
                }
                Tok::Ident(w)
                    if w == "as"
                        && !self.no_as_cast
                        && self.same_line()
                        && self.nth(1).ident().is_some_and(cast_type) =>
                {
                    self.bump();
                    let ty = self.bump();
                    let f = match ty.ident().unwrap_or("") {
                        "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "usize"
                        | "isize" | "int" | "Int" | "long" => "int",
                        "f32" | "f64" | "float" | "Float" | "double" | "Double" => "float",
                        "String" | "str" | "string" => "str",
                        "bool" | "Bool" | "boolean" => "bool",
                        _ => "__identity",
                    };
                    let span = self.span_from(start);
                    e = self.call_expr(f, vec![e], span);
                }
                _ => break,
            }
        }
        Ok(e)
    }

    /// `f (x) y`: a spaced parenthesised group followed by another argument
    /// is Haskell application, not call parentheses.
    fn paren_followed_by_atom(&self) -> bool {
        let toks = self.toks_ref();
        let mut depth = 0;
        let mut i = self.pos_ref();
        while let Some(t) = toks.get(i) {
            match &t.tok {
                Tok::Op("(" | "[" | "{") => depth += 1,
                Tok::Op(")" | "]" | "}") => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                Tok::Eof => return false,
                _ => {}
            }
            i += 1;
        }
        toks.get(i + 1).is_some_and(|n| {
            n.line == toks[i].line && (atom_start(n) || n.is_op("(") && n.space_before)
        })
    }

    /// Ruby's `puts (1..10).to_a`: a spaced `(` whose group is followed by
    /// `.`, `[` or an operator is the start of the argument, not call parens.
    fn paren_starts_argument_at(&self, offset: usize) -> bool {
        let toks = self.toks_ref();
        let mut depth = 0;
        let mut i = self.pos_ref() + offset;
        while let Some(t) = toks.get(i) {
            match &t.tok {
                Tok::Op("(" | "[" | "{") => depth += 1,
                Tok::Op(")" | "]" | "}") => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                Tok::Eof => return false,
                _ => {}
            }
            i += 1;
        }
        toks.get(i + 1).is_some_and(|n| {
            n.line == toks[i].line
                && matches!(
                    &n.tok,
                    Tok::Op(
                        "." | "[" | "+" | "-" | "*" | "/" | "%" | "**" | ".." | "..." | "&." | "<<"
                    )
                )
        })
    }

    fn call_with_space_ok(&self, callee: &Expr) -> bool {
        matches!(callee.kind, ExprKind::Name(_) | ExprKind::Member { .. })
            && !matches!(&callee.kind, ExprKind::Name(n) if is_reserved(&n.as_str()))
    }

    fn trailing_block_ok(&self, e: &Expr) -> bool {
        matches!(
            e.kind,
            ExprKind::Call { .. } | ExprKind::Member { .. } | ExprKind::Name(_)
        )
    }

    fn brace_map_ahead(&self) -> bool {
        let n1 = self.nth(1);
        let n2 = self.nth(2);
        n1.ident().is_some() && n2.is_op(":") && !n2.is_op("::")
            || matches!(n1.tok, Tok::Str(_)) && (n2.is_op(":") || n2.is_op("=>"))
    }

    /// `Point { x: 1, y: 2 }` after a capitalised name.
    fn struct_literal_after(&mut self, e: &Expr) -> PResult<Option<Expr>> {
        let capital = match &e.kind {
            ExprKind::Name(n) => looks_capitalized(&n.as_str()),
            ExprKind::Member { name, .. } => looks_capitalized(&name.as_str()),
            _ => false,
        };
        if !capital {
            return Ok(None);
        }
        let n1 = self.nth(1);
        let n2 = self.nth(2);
        let fieldy = n1.is_op("}")
            || n1.is_op("..")
            || n1.ident().is_some()
                && (n2.is_op(":") || n2.is_op(",") || n2.is_op("}") || n2.is_op("="))
            || matches!(n1.tok, Tok::Newline);
        if !fieldy {
            return Ok(None);
        }
        let Some(fields) = self.attempt(|p| p.struct_fields()) else {
            return Ok(None);
        };
        let span = self.span_from(e.span);
        Ok(Some(Expr::new(
            ExprKind::StructLit {
                ty: Box::new(e.clone()),
                fields,
            },
            span,
        )))
    }

    /// `{ x: 1, y, ..base }` / `{ X = 1 }`.
    fn struct_fields(&mut self) -> PResult<Vec<(Sym, Expr)>> {
        self.expect_op("{", "")?;
        self.nested(|p| {
            let mut fields = Vec::new();
            loop {
                p.skip_newlines();
                if p.at_op("}") {
                    break;
                }
                if p.eat_op("..") {
                    let base = p.expr()?;
                    fields.push((Sym::new("..base"), base));
                } else {
                    let (name, span) = p.expect_ident("for the field")?;
                    let value = if p.eat_op(":") || p.eat_op("=") {
                        p.skip_newlines();
                        p.expr()?
                    } else {
                        Expr::new(ExprKind::Name(name), span)
                    };
                    fields.push((name, value));
                }
                p.skip_newlines();
                if !p.eat_op(",") {
                    break;
                }
            }
            p.skip_newlines();
            p.expect_op("}", "to close the struct literal")?;
            Ok(fields)
        })
    }

    /// `do |x| … end` as a trailing block argument.
    fn do_block(&mut self) -> PResult<Expr> {
        let h = self.header();
        let start = self.bump().span;
        let mut params = Vec::new();
        if self.at_op("|") {
            self.bump();
            while !self.at_op("|") && !self.at_eof() {
                params.push(self.param()?);
                if !self.eat_op(",") {
                    break;
                }
            }
            self.expect_op("|", "to close the block parameters")?;
        }
        let body = if self.at_newline() {
            self.body(h, false, terms::BASE)?
        } else {
            self.body_inline_or_block(h)?
        };
        self.maybe_end(h);
        let span = self.span_from(start);
        let mut def = self.new_fn(None, params.clone(), body, span);
        if params.is_empty() {
            def.params = vec![Param {
                pattern: Pattern::Bind(Sym::new("it")),
                default: Some(Expr::new(ExprKind::Nil, span)),
                ty: None,
                kind: ParamKind::Normal,
                label: None,
            }];
            def.implicit_params = true;
        }
        Ok(Expr::new(ExprKind::Lambda(Rc::new(def)), span))
    }

    fn body_inline_or_block(&mut self, h: Header) -> PResult<Block> {
        let mut stmts = Vec::new();
        while !self.at_newline() && !self.at_end_word() && !self.at_eof() {
            stmts.push(self.statement()?);
            self.eat_op(";");
        }
        let _ = h;
        Ok(Block::new(stmts))
    }

    /// Index contents: plain index, slice `a:b:c`, or multi-dimensional `i, j`.
    fn index_expr(&mut self) -> PResult<Expr> {
        let saved = self.in_index;
        self.in_index = true;
        let r = self.index_expr_inner();
        self.in_index = saved;
        r
    }

    fn index_expr_inner(&mut self) -> PResult<Expr> {
        let first = self.slice_part()?;
        if !self.at_op(",") {
            return Ok(first);
        }
        let mut items = vec![first];
        while self.eat_op(",") {
            items.push(self.slice_part()?);
        }
        let span = items[0].span.to(items[items.len() - 1].span);
        Ok(Expr::new(ExprKind::Tuple(items), span))
    }

    fn slice_part(&mut self) -> PResult<Expr> {
        let start = self.peek().span;
        let lo = if self.at_op(":") || self.at_op("::") {
            None
        } else {
            Some(Box::new(self.expr()?))
        };
        if self.at_op("::") {
            // `xs[::2]`
            self.bump();
            let step = if self.at_op("]") || self.at_op(",") {
                None
            } else {
                Some(Box::new(self.expr()?))
            };
            return Ok(Expr::new(
                ExprKind::Slice {
                    start: lo,
                    end: None,
                    step,
                },
                self.span_from(start),
            ));
        }
        if !self.at_op(":") {
            return lo.map(|b| *b).ok_or(());
        }
        self.bump();
        let hi = if self.at_op(":") || self.at_op("]") || self.at_op(",") {
            None
        } else {
            Some(Box::new(self.expr()?))
        };
        let step = if self.eat_op(":") {
            if self.at_op("]") || self.at_op(",") {
                None
            } else {
                Some(Box::new(self.expr()?))
            }
        } else {
            None
        };
        Ok(Expr::new(
            ExprKind::Slice {
                start: lo,
                end: hi,
                step,
            },
            self.span_from(start),
        ))
    }

    /// `( args )` with positional, named (`x: 1`, `x = 1`) and spread args.
    pub(crate) fn call_args(&mut self) -> PResult<Vec<Arg>> {
        self.expect_op("(", "")?;
        self.nested(|p| {
            let mut args = Vec::new();
            loop {
                p.skip_newlines();
                if p.eat_op(")") {
                    break;
                }
                if p.at_eof() {
                    let span = p.peek().span;
                    return p.err(span, "unclosed argument list");
                }
                if p.at_op("*") && p.nth_is_op(1, ")") {
                    // SQL `count(*)`.
                    let t = p.bump();
                    args.push(Arg::Pos(Expr::new(ExprKind::Str("*".into()), t.span)));
                } else if p.at_op("**") {
                    p.bump();
                    args.push(Arg::KwSpread(p.expr()?));
                } else if p.at_op("*") || p.at_op("...") {
                    p.bump();
                    args.push(Arg::Spread(p.expr()?));
                } else if let Tok::Ident(w) = &p.peek().tok
                    && (p.nth_is_op(1, ":") && !p.nth_is_op(1, "::")
                        || p.nth_is_op(1, "=") && !p.nth_is_op(1, "=="))
                    && (!is_reserved(w)
                        // Swift labels may be keywords: `greet(from: "x")`, `index(of: 3)`.
                        || p.nth_is_op(1, ":")
                            && p.nth(2).space_before
                            && !matches!(w.as_str(), "lambda" | "if" | "else" | "for" | "while" | "case" | "default"))
                {
                    let name = Sym::new(w);
                    p.bump();
                    p.bump();
                    p.skip_newlines();
                    args.push(Arg::Named(name, p.expr()?));
                } else {
                    let e = p.expr()?;
                    p.skip_newlines();
                    if p.at_word("for") {
                        // `sum(x for x in xs)`
                        let comp = p.comprehension_tail(CompKind::List, e, None)?;
                        let span = comp.element.span;
                        args.push(Arg::Pos(Expr::new(
                            ExprKind::Comprehension(Box::new(comp)),
                            span,
                        )));
                    } else {
                        args.push(Arg::Pos(e));
                    }
                }
                p.skip_newlines();
                if !p.eat_op(",") {
                    p.skip_newlines();
                    p.expect_op(")", "to close the argument list")?;
                    break;
                }
            }
            Ok(args)
        })
    }

    // ----- command calls ---------------------------------------------------------

    /// An expression statement that may be a parenthesis-free call:
    /// `puts "hi"`, `print x, y`, `obj.method arg`.
    pub(crate) fn command_or_expr(&mut self) -> PResult<Expr> {
        // SQL statements are expressions, not command calls.
        let sql = match self.peek().ident().map(str::to_ascii_lowercase).as_deref() {
            Some("update") => {
                self.nth(1).ident().is_some()
                    && self
                        .nth(2)
                        .ident()
                        .is_some_and(|w| w.eq_ignore_ascii_case("set"))
            }
            Some("delete") => self
                .nth(1)
                .ident()
                .is_some_and(|w| w.eq_ignore_ascii_case("from")),
            Some("insert") => self
                .nth(1)
                .ident()
                .is_some_and(|w| w.eq_ignore_ascii_case("into")),
            _ => false,
        };
        if sql {
            return self.expr();
        }
        if let Some(w) = self.peek().ident().map(str::to_owned)
            && !is_reserved(&w)
            && self.command_arg_ahead(&w, 1)
        {
            let t = self.bump();
            let callee = Expr::new(ExprKind::Name(Sym::new(&w)), t.span);
            return self.command_call(callee);
        }
        let e = self.expr()?;
        if matches!(e.kind, ExprKind::Member { .. })
            && let ExprKind::Member { name, .. } = &e.kind
            && self.command_arg_ahead(&name.as_str(), 0)
        {
            return self.command_call(e);
        }
        // `a, b = b, a` and Python-style bare tuples.
        if self.at_op(",") && self.same_line() {
            if let ExprKind::Assign { target, value } = e.kind {
                let value = self.value_tuple(*value)?;
                let span = target.span.to(value.span);
                return Ok(Expr::new(
                    ExprKind::Assign {
                        target,
                        value: Box::new(value),
                    },
                    span,
                ));
            }
            let lhs = self.statement_level_tuple(e)?;
            if self.at_op("=") || self.at_op(":=") {
                let go_decl = self.bump().is_op(":=");
                self.skip_newlines();
                let first = self.expr()?;
                let mut rhs = self.statement_level_tuple(first)?;
                // Go's comma-ok: `v, ok := m[key]`.
                if go_decl
                    && matches!(&lhs.kind, ExprKind::Tuple(items) if items.len() == 2)
                    && let ExprKind::Index { object, index } = &rhs.kind
                {
                    let span = rhs.span;
                    rhs = self.call_expr(
                        "__comma_ok",
                        vec![(**object).clone(), (**index).clone()],
                        span,
                    );
                }
                let span = lhs.span.to(rhs.span);
                if !lhs.is_assignable() {
                    return self.err(span, "cannot assign to this expression");
                }
                return Ok(Expr::new(
                    ExprKind::Assign {
                        target: Box::new(lhs),
                        value: Box::new(rhs),
                    },
                    span,
                ));
            }
            return Ok(lhs);
        }
        Ok(e)
    }

    fn command_call(&mut self, callee: Expr) -> PResult<Expr> {
        let start = callee.span;

        let mut args = Vec::new();
        loop {
            if let Tok::Ident(w) = &self.peek().tok
                && self.nth_is_op(1, ":")
                && !self.nth_is_op(1, "::")
                && self.nth(2).space_before
                && !is_reserved(w)
            {
                let name = Sym::new(w);
                self.bump();
                self.bump();
                args.push(Arg::Named(name, self.expr()?));
            } else {
                let e = self.expr()?;
                // Ruby `puts greet`: a bare named method is called.
                let e = if matches!(e.kind, ExprKind::Name(_)) {
                    let span = e.span;
                    self.call_expr("__autocall", vec![e], span)
                } else {
                    e
                };
                args.push(Arg::Pos(e));
            }
            if self.eat_op(",") {
                self.skip_newlines();
                continue;
            }
            // Haskell-style juxtaposed arguments: `f x y`.
            if self.same_line() && self.juxtaposed_arg_ahead() {
                continue;
            }
            break;
        }
        let span = self.span_from(start);
        Ok(Expr::new(
            ExprKind::Call {
                callee: Box::new(callee),
                args,
            },
            span,
        ))
    }

    fn juxtaposed_arg_ahead(&self) -> bool {
        let t = self.peek();
        match &t.tok {
            Tok::Int(_) | Tok::Float(_) | Tok::BigInt(_) | Tok::Str(_) => true,
            Tok::Op("(") | Tok::Op("[") => t.space_before,
            Tok::Ident(w) => !is_operator_word(w) && !is_reserved(w),
            _ => false,
        }
    }

    /// Whether the token at offset `n` starts an argument of a command call
    /// whose callee is `callee`.
    fn command_arg_ahead(&self, callee: &str, n: usize) -> bool {
        let t = self.nth(n);
        let callee_tok = if n == 0 { self.prev() } else { self.peek() };
        if t.line != callee_tok.line || !t.space_before {
            return false;
        }
        let tight = |k: usize| !self.nth(n + k).space_before;
        match &t.tok {
            Tok::Str(_) | Tok::Int(_) | Tok::Float(_) | Tok::BigInt(_) | Tok::IVar(_) => true,
            Tok::Ident(w) => {
                !is_operator_word(w)
                    && !matches!(w.as_str(), "end" | "then" | "do" | "else" | "fi" | "done")
                    && !self.nth_is_op(n + 1, "=")
                    && !self.nth_is_op(n + 1, "<-")
            }
            Tok::Op("[") => print_like(callee),
            Tok::Op("(") => self.paren_starts_argument_at(n),
            Tok::Op("-") | Tok::Op("+") => print_like(callee) && tight(1),
            Tok::Op("!") => tight(1),
            Tok::Op(":") => self.nth(n + 1).ident().is_some() && tight(1),
            Tok::Op("√") | Tok::Op("∇") | Tok::Op("\\") => true,
            _ => false,
        }
    }

    // ----- conditionals -----------------------------------------------------------

    /// `if`/`unless` with `elif`/`elsif`/`else if`/`else`/`otherwise`.
    pub(crate) fn if_expr(&mut self) -> PResult<Expr> {
        let h = self.header();
        let kw = self.bump();
        let negate = kw.is_word("unless");
        let e = self.if_tail(h, kw.span, negate)?;
        self.maybe_end(h);
        Ok(e)
    }

    fn if_tail(&mut self, h: Header, start: Span, negate: bool) -> PResult<Expr> {
        // Go: `if v, ok := m[k]; ok { … }` runs the statement first.
        let line = self.peek().line;
        if let Some(init) = self.attempt(|p| {
            if !p.go_init_ahead() {
                return Err(());
            }
            let s = p.in_header(|p| p.statement())?;
            let assigns = match &s.kind {
                StmtKind::Let { .. } => true,
                StmtKind::Expr(e) => matches!(
                    e.kind,
                    ExprKind::Assign { .. } | ExprKind::CompoundAssign { .. }
                ),
                _ => false,
            };
            if assigns
                && p.prev().is_op(";")
                && p.peek().line == line
                && !p.at_op("{")
                && !p.at_word("else")
            {
                Ok(s)
            } else {
                Err(())
            }
        }) {
            let rest = self.if_tail(h, start, negate)?;
            let span = self.span_from(start);
            let rest_stmt = Stmt {
                span: rest.span,
                kind: StmtKind::Expr(rest),
            };
            return Ok(Expr::new(
                ExprKind::Block(Block::new(vec![init, rest_stmt])),
                span,
            ));
        }
        let mut cond = self.header_expr()?;
        if negate {
            let span = cond.span;
            cond = Expr::new(ExprKind::Unary(UnOp::Not, Box::new(cond)), span);
        }
        let then = self.body(h, false, terms::BASE)?;
        let otherwise = self.else_chain(h)?;
        let span = self.span_from(start);
        Ok(self.mk_if(cond, then, otherwise, span))
    }

    /// Whether a `;` follows on this line before any `{` (a Go `if` with an
    /// init statement).
    fn go_init_ahead(&self) -> bool {
        let line = self.peek().line;
        let mut i = 0;
        loop {
            let t = self.nth(i);
            if t.line != line || matches!(t.tok, Tok::Eof | Tok::Newline) || t.is_op("{") {
                return false;
            }
            if t.is_op(";") {
                return true;
            }
            i += 1;
        }
    }

    /// `[]T{` / `[N]T{`: a Go slice or array literal.
    fn go_slice_type_ahead(&self) -> bool {
        let close = if self.nth_is_op(1, "]") {
            1
        } else if matches!(self.nth(1).tok, Tok::Int(_)) && self.nth_is_op(2, "]") {
            2
        } else {
            return false;
        };
        let t = self.nth(close + 1);
        !t.space_before && (t.ident().is_some() || t.is_op("[") || t.is_op("*"))
    }

    /// `map[K]V{`: a Go map literal.
    fn go_map_type_ahead(&self) -> bool {
        let mut depth = 0;
        let mut i = 1;
        loop {
            let t = self.nth(i);
            if matches!(t.tok, Tok::Eof | Tok::Newline) {
                return false;
            }
            if t.is_op("[") {
                depth += 1;
            } else if t.is_op("]") {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            i += 1;
        }
        // The value type, then `{`.
        let mut j = i + 1;
        while self.nth(j).ident().is_some()
            || self.nth(j).is_op("[")
            || self.nth(j).is_op("]")
            || self.nth(j).is_op(".")
            || self.nth(j).is_op("*")
        {
            j += 1;
        }
        self.nth(j).is_op("{")
    }

    /// Go composite literals: `[]int{1, 2}`, `[][]int{{1}, {2}}`,
    /// `[]Point{{1, 2}}`, `map[string]int{"a": 1}`.
    fn go_composite(&mut self) -> PResult<Expr> {
        let start = self.peek().span;
        let ty = self.type_expr()?;
        if !self.at_op("{") {
            let span = self.span_from(start);
            return Ok(Expr::new(ExprKind::List(Vec::new()), span));
        }
        let lit = self.brace_expr()?;
        Ok(go_shape(lit, &ty))
    }

    fn else_chain(&mut self, h: Header) -> PResult<Option<Block>> {
        let next = self.peek_past_newlines().clone();
        let attach = next.line == self.prev().line || next.col >= h.indent;
        if !attach || next.ident().is_none() {
            return Ok(None);
        }
        let w = next.ident().unwrap_or("");
        let elif = matches!(w, "elif" | "elsif" | "elseif")
            || (w == "else" && self.nth_past_newlines_is_if());
        if elif {
            self.skip_newlines();
            let kw = self.bump();
            if kw.is_word("else") {
                self.bump(); // `if`
            }
            let eh = Header {
                indent: h.indent,
                line: kw.line,
            };
            let nested = self.if_tail(eh, kw.span, false)?;
            return Ok(Some(Block::of_expr(nested)));
        }
        if (w == "else" || w == "otherwise") && !self.word_used_as_value_past_newlines() {
            self.skip_newlines();
            let kw = self.bump();
            self.eat_op(",");
            let eh = Header {
                indent: h.indent,
                line: kw.line,
            };
            return Ok(Some(self.body(eh, false, terms::BASE)?));
        }
        Ok(None)
    }

    fn nth_past_newlines_is_if(&self) -> bool {
        let mut i = self.pos_ref();
        let toks = self.toks_ref();
        while i < toks.len() && matches!(toks[i].tok, Tok::Newline) {
            i += 1;
        }
        toks.get(i + 1)
            .is_some_and(|t| t.is_word("if") && t.line == toks[i].line)
    }

    fn word_used_as_value_past_newlines(&self) -> bool {
        let mut i = self.pos_ref();
        let toks = self.toks_ref();
        while i < toks.len() && matches!(toks[i].tok, Tok::Newline) {
            i += 1;
        }
        toks.get(i + 1)
            .is_some_and(|t| t.is_op("=") || t.is_op("."))
    }

    // ----- match ---------------------------------------------------------------

    /// `match`/`case`/`when` in their Rust, Scala, Python, OCaml, Haskell,
    /// Elixir, Ruby, Kotlin and SQL forms.
    pub(crate) fn match_expr(&mut self) -> PResult<Expr> {
        let h = self.header();
        let kw = self.bump();
        let start = kw.span;
        let word = kw.ident().unwrap_or("match").to_ascii_lowercase();

        if word == "when" {
            return self.kotlin_when(start);
        }
        if word == "case"
            && (self.at_word("when")
                || self.at_word("WHEN")
                || self.at_newline() && self.peek_past_newlines().is_word("when"))
        {
            // Subject-less `case when c then v … else v end` (SQL, Ruby).
            let arms = self.when_arms(h, None)?;
            self.maybe_end(h);
            return Ok(Expr::new(
                ExprKind::Match {
                    subject: None,
                    arms,
                },
                self.span_from(start),
            ));
        }
        let subject = self.header_expr()?;
        let arms = if self.at_op("{") {
            self.brace_arms()?
        } else if self.at_word("with") {
            self.bump();
            self.ocaml_arms()?
        } else if self.at_word("of") {
            self.bump();
            self.layout_arms(h, "->")?
        } else if word == "case"
            && (self.at_word("do") || self.at_op(":"))
            && self.elixir_arms_ahead()
        {
            self.bump();
            let arms = self.layout_arms(h, "->")?;
            self.maybe_end(h);
            arms
        } else if self.at_op(":") {
            // Python 3.10: `match x:` + `case p:` arms.
            self.bump();
            self.python_case_arms(h)?
        } else if word == "case" {
            // Ruby/SQL: `case x when 1 then … else … end`.
            let arms = self.when_arms(h, Some(()))?;
            self.maybe_end(h);
            arms
        } else {
            let span = self.peek().span;
            let found = describe(self.peek());
            return self.err(
                span,
                format!("expected `{{`, `with` or `:` after the `match` subject, found {found}"),
            );
        };
        Ok(Expr::new(
            ExprKind::Match {
                subject: Some(Box::new(subject)),
                arms,
            },
            self.span_from(start),
        ))
    }

    fn elixir_arms_ahead(&self) -> bool {
        // `case x do` followed by `pattern ->` lines.
        let mut i = self.pos_ref() + 1;
        let toks = self.toks_ref();
        while i < toks.len() && matches!(toks[i].tok, Tok::Newline) {
            i += 1;
        }
        let line = toks.get(i).map(|t| t.line);
        while i < toks.len() && Some(toks[i].line) == line {
            if toks[i].is_op("->") {
                return true;
            }
            i += 1;
        }
        false
    }

    fn guard(&mut self) -> PResult<Option<Expr>> {
        if self.at_word("if") || self.at_word("when") || self.at_word("where") {
            self.bump();
            return Ok(Some(self.nested(|p| {
                p.no_arrow_lambda = true;
                p.expr_no_assign()
            })?));
        }
        Ok(None)
    }

    fn arm_body(&mut self) -> PResult<Block> {
        self.skip_newlines();
        if self.at_op("{") && self.brace_looks_like_block() {
            return self.brace_block();
        }
        if self.at_op("{") && self.nth_is_op(1, "}") {
            self.bump();
            self.bump();
            return Ok(Block::default());
        }
        let e = self.nested(|p| p.expr())?;
        let e = self.statement_modifiers(e)?;
        Ok(Block::of_expr(e))
    }

    /// `{ pat [if g] => body, … }` (Rust, Scala with `case`).
    pub(crate) fn brace_arms(&mut self) -> PResult<Vec<MatchArm>> {
        self.expect_op("{", "")?;
        self.nested(|p| {
            let mut arms = Vec::new();
            loop {
                p.skip_separators();
                if p.eat_op("}") {
                    break;
                }
                if p.at_eof() {
                    let span = p.peek().span;
                    return p.err(span, "unclosed `match`");
                }
                let scala = p.eat_word("case");
                let pattern = p.match_pattern()?;
                let guard = p.guard()?;
                if !(p.eat_op("=>") || p.eat_op("->") || (scala && p.eat_op(":"))) {
                    let span = p.peek().span;
                    let found = describe(p.peek());
                    return p.err(
                        span,
                        format!("expected `=>` after the pattern, found {found}"),
                    );
                }
                let body = if scala && p.at_newline() {
                    // Scala arms may hold several statements, up to the next `case`.
                    let mut stmts = Vec::new();
                    loop {
                        p.skip_separators();
                        if p.at_op("}") || p.at_word("case") || p.at_eof() {
                            break;
                        }
                        stmts.push(p.statement()?);
                    }
                    Block::new(stmts)
                } else {
                    p.arm_body()?
                };
                arms.push(MatchArm {
                    pattern,
                    guard,
                    body,
                });
                p.eat_op(",");
            }
            Ok(arms)
        })
    }

    /// OCaml/F#: `match x with | p -> e | q when g -> e`.
    fn ocaml_arms(&mut self) -> PResult<Vec<MatchArm>> {
        let mut arms = Vec::new();
        self.skip_newlines();
        self.eat_op("|");
        loop {
            self.skip_newlines();
            let pattern = self.nested(|p| {
                p.no_arrow_lambda = true;
                p.match_pattern()
            })?;
            let guard = self.guard()?;
            self.expect_op("->", "after the pattern")?;
            let e = self.nested(|p| {
                p.no_arrow_lambda = true;
                p.expr()
            })?;
            arms.push(MatchArm {
                pattern,
                guard,
                body: Block::of_expr(e),
            });
            if self.at_op("|") || (self.at_newline() && self.peek_past_newlines().is_op("|")) {
                self.skip_newlines();
                self.bump();
                continue;
            }
            break;
        }
        Ok(arms)
    }

    /// Haskell `case x of` / Elixir `case x do` arms laid out one per line.
    fn layout_arms(&mut self, h: Header, arrow: &str) -> PResult<Vec<MatchArm>> {
        let mut arms = Vec::new();
        self.skip_newlines();
        let col = self.peek().col;
        let inline = !self.peek().line_start;
        loop {
            self.skip_separators();
            let t = self.peek();
            if self.at_eof()
                || self.at_end_word()
                || t.is_op("}")
                || (t.line_start && (t.col < col || t.col <= h.indent && !inline))
            {
                break;
            }
            let pattern = self.nested(|p| {
                p.no_arrow_lambda = true;
                p.match_pattern()
            })?;
            // Haskell guards: `n | n < 0 -> …`.
            let guard = if self.at_op("|") {
                self.bump();
                Some(self.nested(|p| {
                    p.no_arrow_lambda = true;
                    p.expr_no_assign()
                })?)
            } else {
                self.guard()?
            };
            self.expect_op(arrow, "after the pattern")?;
            let body = if self.at_newline() {
                let ah = self.header();
                self.body(
                    Header {
                        indent: col,
                        line: ah.line,
                    },
                    false,
                    terms::BASE,
                )?
            } else {
                Block::of_expr(self.nested(|p| {
                    p.no_arrow_lambda = true;
                    p.expr()
                })?)
            };
            arms.push(MatchArm {
                pattern,
                guard,
                body,
            });
            if inline && !self.at_op(";") && !self.at_newline() {
                break;
            }
        }
        Ok(arms)
    }

    /// Python 3.10 `case pattern [if guard]:` arms.
    fn python_case_arms(&mut self, h: Header) -> PResult<Vec<MatchArm>> {
        let mut arms = Vec::new();
        self.skip_newlines();
        let col = self.peek().col;
        while self.at_word("case") && self.peek().col == col && col > h.indent {
            let ch = self.header();
            self.bump();
            let pattern = self.match_pattern()?;
            let pattern = if self.at_op(",") {
                // `case 1, 2:` → tuple-ish alternatives are tuple patterns in
                // Python; keep that meaning.
                let mut items = vec![pattern];
                while self.eat_op(",") {
                    items.push(self.match_pattern()?);
                }
                Pattern::Tuple(items)
            } else {
                pattern
            };
            let guard = self.guard()?;
            let body = self.body(ch, false, terms::BASE)?;
            arms.push(MatchArm {
                pattern,
                guard,
                body,
            });
            self.skip_newlines();
        }
        Ok(arms)
    }

    /// Ruby/SQL `when v1, v2 then …` arms, ending at `end`. Without a
    /// subject each `when` holds a condition.
    fn when_arms(&mut self, h: Header, subject: Option<()>) -> PResult<Vec<MatchArm>> {
        let mut arms = Vec::new();
        loop {
            self.skip_separators();
            if self.at_word("when") || self.at_word("WHEN") {
                let wh = self.header();
                self.bump();
                if subject.is_none() {
                    let cond = self.in_header(|p| p.expr_no_assign())?;
                    let body = self.when_body(wh)?;
                    arms.push(MatchArm {
                        pattern: Pattern::Wildcard,
                        guard: Some(cond),
                        body,
                    });
                    continue;
                }
                let mut alts = Vec::new();
                loop {
                    if self.at_word("in") || self.at_word("is") {
                        alts.push(self.match_pattern()?);
                    } else {
                        alts.push(Pattern::CaseEq(self.in_header(|p| p.expr_no_assign())?));
                    }
                    if !self.eat_op(",") {
                        break;
                    }
                }
                let pattern = if alts.len() == 1 {
                    alts.pop().unwrap_or(Pattern::Wildcard)
                } else {
                    Pattern::Or(alts)
                };
                let body = self.when_body(wh)?;
                arms.push(MatchArm {
                    pattern,
                    guard: None,
                    body,
                });
            } else if self.at_word("else") || self.at_word("ELSE") {
                let wh = self.header();
                self.bump();
                let body = self.when_body(wh)?;
                arms.push(MatchArm {
                    pattern: Pattern::Wildcard,
                    guard: None,
                    body,
                });
            } else {
                break;
            }
        }
        let _ = h;
        Ok(arms)
    }

    fn when_body(&mut self, wh: Header) -> PResult<Block> {
        if (self.eat_word("then") || self.eat_word("THEN") || self.eat_op(":") || self.eat_op("->"))
            && !self.at_newline()
        {
            let mut stmts = Vec::new();
            while !self.at_newline()
                && !self.at_eof()
                && !self.at_any_word(&["when", "WHEN", "else", "ELSE", "end", "END"])
            {
                stmts.push(self.statement()?);
                self.eat_op(";");
            }
            return Ok(Block::new(stmts));
        }
        self.skip_separators();
        let mut stmts = Vec::new();
        while !self.at_eof()
            && !self.at_terminator(terms::END | terms::WHEN | terms::ELSE)
            && !self.at_any_word(&["END", "WHEN", "ELSE"])
        {
            stmts.push(self.statement()?);
            self.skip_separators();
        }
        let _ = wh;
        Ok(Block::new(stmts))
    }

    /// Kotlin `when (x) { 1 -> …; in 2..5 -> …; is String -> …; else -> … }`
    /// and subject-less `when { cond -> … }`.
    fn kotlin_when(&mut self, start: Span) -> PResult<Expr> {
        let subject = if self.at_op("(") {
            self.bump();
            let s = self.nested(|p| p.expr())?;
            self.expect_op(")", "")?;
            Some(s)
        } else {
            None
        };
        self.expect_op("{", "to open the `when` arms")?;
        let arms = self.nested(|p| {
            p.no_arrow_lambda = true;
            let mut arms = Vec::new();
            loop {
                p.skip_separators();
                if p.eat_op("}") {
                    break;
                }
                if p.at_eof() {
                    let span = p.peek().span;
                    return p.err(span, "unclosed `when`");
                }
                let mut alts = Vec::new();
                let mut guard = None;
                if p.eat_word("else") {
                    alts.push(Pattern::Wildcard);
                } else if subject.is_some() {
                    loop {
                        if p.at_word("in") || p.at_word("is") {
                            alts.push(p.match_pattern()?);
                        } else {
                            alts.push(Pattern::CaseEq(p.expr_no_assign()?));
                        }
                        if !p.eat_op(",") {
                            break;
                        }
                    }
                } else {
                    guard = Some(p.expr_no_assign()?);
                    alts.push(Pattern::Wildcard);
                }
                p.expect_op("->", "after the `when` condition")?;
                let body = p.arm_body()?;
                let pattern = if alts.len() == 1 {
                    alts.pop().unwrap_or(Pattern::Wildcard)
                } else {
                    Pattern::Or(alts)
                };
                arms.push(MatchArm {
                    pattern,
                    guard,
                    body,
                });
            }
            Ok(arms)
        })?;
        Ok(Expr::new(
            ExprKind::Match {
                subject: subject.map(Box::new),
                arms,
            },
            self.span_from(start),
        ))
    }
}

/// Results of reading a comparison-ish operator.
enum CmpOpOrSpecial {
    Op(CmpOp),
    Predicate(String),
    Between,
    DivisibleBy,
    Like(bool),
    Contains,
}

/// Whether `t` can begin an expression.
pub(crate) fn starts_expr(t: &crate::lexer::Token) -> bool {
    match &t.tok {
        Tok::Int(_) | Tok::Float(_) | Tok::BigInt(_) | Tok::Str(_) | Tok::IVar(_) => true,
        Tok::Ident(w) => !matches!(
            w.as_str(),
            "then"
                | "do"
                | "else"
                | "end"
                | "elif"
                | "elsif"
                | "fi"
                | "done"
                | "times"
                | "by"
                | "step"
        ),
        Tok::Op(op) => matches!(
            *op,
            "(" | "["
                | "{"
                | "-"
                | "+"
                | "!"
                | "~"
                | "√"
                | "∇"
                | "\\"
                | "|"
                | "||"
                | "..."
                | "*"
                | "&"
                | "->"
                | "++"
                | "--"
                | "#"
        ),
        _ => false,
    }
}

/// Whether `t` can be a juxtaposed argument (`f x`, `f 1`, `f [1, 2]`).
fn atom_start(t: &crate::lexer::Token) -> bool {
    if !t.space_before {
        return false;
    }
    match &t.tok {
        Tok::Int(_) | Tok::Float(_) | Tok::BigInt(_) | Tok::Str(_) | Tok::IVar(_) => true,
        Tok::Op("[") | Tok::Op("\\") => true,
        Tok::Ident(w) => {
            !is_operator_word(w)
                && (!is_reserved(w)
                    || matches!(
                        w.as_str(),
                        "true"
                            | "false"
                            | "nil"
                            | "null"
                            | "None"
                            | "True"
                            | "False"
                            | "self"
                            | "this"
                    ))
        }
        _ => false,
    }
}

fn cast_type(w: &str) -> bool {
    matches!(
        w,
        "i8" | "i16"
            | "i32"
            | "i64"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "usize"
            | "isize"
            | "int"
            | "Int"
            | "long"
            | "f32"
            | "f64"
            | "float"
            | "Float"
            | "double"
            | "Double"
            | "String"
            | "str"
            | "string"
            | "bool"
            | "Bool"
            | "boolean"
    )
}

/// Appends a trailing block/map argument to a call (or makes one).
fn append_arg(callee: Expr, arg: Expr, span: Span) -> Expr {
    match callee.kind {
        ExprKind::Call { callee, mut args } => {
            args.push(Arg::Pos(arg));
            Expr::new(ExprKind::Call { callee, args }, span)
        }
        _ => Expr::new(
            ExprKind::Call {
                callee: Box::new(callee),
                args: vec![Arg::Pos(arg)],
            },
            span,
        ),
    }
}

/// `[1..10]` and `[1,3..9]` (Haskell) expand to ranges.
fn haskell_range_list(items: Vec<Expr>, span: Span) -> Expr {
    if items.len() == 1
        && let ExprKind::Range { .. } = &items[0].kind
    {
        let range = items
            .into_iter()
            .next()
            .unwrap_or(Expr::new(ExprKind::Nil, span));
        return Expr::new(
            ExprKind::Call {
                callee: Box::new(Expr::new(ExprKind::Name(Sym::new("list")), span)),
                args: vec![Arg::Pos(range)],
            },
            span,
        );
    }
    if items.len() == 2
        && let ExprKind::Range {
            start: Some(second),
            end,
            inclusive,
            step: None,
        } = &items[1].kind
        && matches!(items[0].kind, ExprKind::Int(_) | ExprKind::Float(_))
        && matches!(second.kind, ExprKind::Int(_) | ExprKind::Float(_))
    {
        let first = items[0].clone();
        let step = Expr::new(
            ExprKind::Binary(BinOp::Sub, second.clone(), Box::new(first.clone())),
            span,
        );
        let range = Expr::new(
            ExprKind::Range {
                start: Some(Box::new(first)),
                end: end.clone(),
                inclusive: *inclusive,
                step: Some(Box::new(step)),
            },
            span,
        );
        return Expr::new(
            ExprKind::Call {
                callee: Box::new(Expr::new(ExprKind::Name(Sym::new("list")), span)),
                args: vec![Arg::Pos(range)],
            },
            span,
        );
    }
    Expr::new(ExprKind::List(items), span)
}

/// Gives a brace literal the shape of a Go composite type: `{1, 2}` is a
/// list for `[]int` and a constructor call for `Point`.
fn go_shape(lit: Expr, ty: &TypeExpr) -> Expr {
    let span = lit.span;
    match (lit.kind, ty) {
        (ExprKind::Set(items), TypeExpr::List(elem)) => Expr::new(
            ExprKind::List(items.into_iter().map(|e| go_shape(e, elem)).collect()),
            span,
        ),
        (ExprKind::Map(items), TypeExpr::List(_)) if items.is_empty() => {
            Expr::new(ExprKind::List(Vec::new()), span)
        }
        (ExprKind::Set(items), TypeExpr::Named(n, _))
            if n.as_str().chars().next().is_some_and(char::is_uppercase) && !n.is("Map") =>
        {
            Expr::new(
                ExprKind::Call {
                    callee: Box::new(Expr::new(ExprKind::Name(*n), span)),
                    args: items.into_iter().map(Arg::Pos).collect(),
                },
                span,
            )
        }
        (ExprKind::Map(items), TypeExpr::Named(n, _))
            if n.as_str().chars().next().is_some_and(char::is_uppercase)
                && !n.is("Map")
                && !items.is_empty() =>
        {
            // `Point{X: 1, Y: 2}` written inside a slice literal with the type elided.
            let fields = items
                .into_iter()
                .filter_map(|(k, v)| match k.kind {
                    ExprKind::Str(s) => Some((Sym::new(&s), v)),
                    _ => None,
                })
                .collect();
            Expr::new(
                ExprKind::StructLit {
                    ty: Box::new(Expr::new(ExprKind::Name(*n), span)),
                    fields,
                },
                span,
            )
        }
        (kind, _) => Expr::new(kind, span),
    }
}

/// Operators that can form a Haskell section: `(+)`, `(* 2)`, `(x <)`.
fn is_section_op(op: &str) -> bool {
    matches!(
        op,
        "+" | "-"
            | "*"
            | "/"
            | "^"
            | "**"
            | "%"
            | "++"
            | "=="
            | "/="
            | "!="
            | "<"
            | ">"
            | "<="
            | ">="
            | "&&"
            | "||"
            | "."
            | "<>"
            | "!!"
    )
}
