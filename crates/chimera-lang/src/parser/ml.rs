//! Machine-learning syntax: `model` declarations and `train` statements.
//!
//! ```text
//! model XorNet:
//!     dense 2 -> 8, tanh
//!     dense 8 -> 1, sigmoid
//!
//! train net on X, y for 2000 epochs using adam(lr: 0.05) minimizing bce reporting every 500
//! ```

use std::rc::Rc;

use super::expr::bp;
use super::{Header, PResult, Parser, terms};
use crate::ast::*;
use crate::lexer::Tok;
use crate::symbol::Sym;

/// Activation names accepted after a comma in a layer line.
const ACTIVATIONS: &[&str] = &[
    "relu", "sigmoid", "tanh", "softmax", "gelu", "leaky_relu", "elu", "linear", "identity", "swish",
    "silu", "softplus",
];

impl Parser<'_> {
    /// `model Name` followed by a body.
    pub(crate) fn model_ahead(&self) -> bool {
        let n2 = self.nth(2);
        n2.is_op(":") || n2.is_op("{") || matches!(n2.tok, Tok::Newline) || n2.is_word("is") || n2.is_word("do")
    }

    pub(crate) fn model_decl(&mut self, h: Header) -> PResult<Stmt> {
        let start = self.bump().span;
        let (name, _) = self.expect_ident("for the model")?;
        self.eat_word("is");
        let body = self.body_with(h, false, terms::END, Self::layer_line)?;
        self.maybe_end(h);
        let mut layers = Vec::new();
        for stmt in body.stmts {
            if let StmtKind::Expr(Expr {
                kind: ExprKind::Call { callee, args },
                span,
            }) = stmt.kind
                && let ExprKind::Name(kind) = callee.kind
            {
                layers.push(LayerSpec { kind, args, span });
            }
        }
        let span = self.span_from(start);
        Ok(Stmt {
            kind: StmtKind::Model(Rc::new(ModelDef { name, layers, span })),
            span,
        })
    }

    /// One layer: `dense 2 -> 8, relu`, `linear(4, 2)`, `dropout 0.1`, `relu`.
    /// Returned as a call statement and unpacked by [`Parser::model_decl`].
    fn layer_line(&mut self) -> PResult<Stmt> {
        let start = self.peek().span;
        let (kind, kind_span) = self.expect_ident("for the layer")?;
        let mut args = Vec::new();
        if self.at_op("(") && !self.peek().space_before {
            args = self.call_args()?;
        } else {
            while self.same_line() && !self.at_op(";") {
                if let Some(w) = self.peek().ident()
                    && ACTIVATIONS.contains(&w)
                    && !self.nth_is_op(1, "(")
                {
                    let t = self.bump();
                    let act = Expr::new(ExprKind::Str(w_to_rc(t.ident().unwrap_or(""))), t.span);
                    args.push(Arg::Named(Sym::new("activation"), act));
                } else if self.peek().ident().is_some() && (self.nth_is_op(1, ":") || self.nth_is_op(1, "=")) {
                    let (n, _) = self.expect_ident("")?;
                    self.bump();
                    let v = self.layer_value()?;
                    args.push(Arg::Named(n, v));
                } else {
                    let v = self.layer_value()?;
                    args.push(Arg::Pos(v));
                    if self.eat_op("->") || self.eat_word("to") {
                        let out = self.layer_value()?;
                        args.push(Arg::Pos(out));
                    }
                }
                if !self.eat_op(",") {
                    break;
                }
            }
        }
        self.end_stmt()?;
        let span = self.span_from(start);
        Ok(Stmt {
            kind: StmtKind::Expr(Expr::new(
                ExprKind::Call {
                    callee: Box::new(Expr::new(ExprKind::Name(kind), kind_span)),
                    args,
                },
                span,
            )),
            span,
        })
    }

    fn layer_value(&mut self) -> PResult<Expr> {
        self.nested(|p| {
            p.no_arrow_lambda = true;
            p.expr_bp(bp::RANGE + 1)
        })
    }

    /// `train MODEL on X, Y [for N epochs] [using OPT] [minimizing LOSS]
    /// [with learning rate R] [in batches of B] [reporting every K]`.
    pub(crate) fn train_stmt(&mut self) -> PResult<Stmt> {
        let start = self.bump().span;
        let model = self.postfix_expr()?;
        if !self.eat_word("on") {
            return Err(());
        }
        let inputs = self.expr_bp(bp::RANGE + 1)?;
        if !(self.eat_op(",") || self.eat_word("and") || self.eat_word("to")) {
            let span = self.peek().span;
            return self.err(span, "expected `, targets` after the training inputs");
        }
        let targets = self.expr_bp(bp::RANGE + 1)?;
        let mut options: Vec<(Sym, Expr)> = Vec::new();
        let opt = |name: &str| Sym::new(name);
        while self.same_line() && !self.at_op(";") {
            if self.eat_word("for") {
                let n = self.expr_bp(bp::RANGE + 1)?;
                if !(self.eat_word("epochs") || self.eat_word("epoch") || self.eat_word("iterations") || self.eat_word("steps")) {
                    let span = self.peek().span;
                    return self.err(span, "expected `epochs` after the count");
                }
                options.push((opt("epochs"), n));
            } else if matches!(self.peek().tok, Tok::Int(_)) && (self.nth_is_word(1, "epochs") || self.nth_is_word(1, "epoch")) {
                let n = self.expr_bp(bp::RANGE + 1)?;
                self.bump();
                options.push((opt("epochs"), n));
            } else if self.at_word("with") && self.nth_is_word(1, "learning") || self.at_word("learning") {
                self.eat_word("with");
                self.bump();
                self.expect_word("rate", "")?;
                self.eat_op("=");
                options.push((opt("lr"), self.expr_bp(bp::RANGE + 1)?));
            } else if self.at_word("lr") {
                self.bump();
                self.eat_op("=");
                self.eat_op(":");
                options.push((opt("lr"), self.expr_bp(bp::RANGE + 1)?));
            } else if self.eat_word("using") || self.eat_word("with") || self.eat_word("optimizer") {
                let o = self.postfix_expr()?;
                options.push((opt("optimizer"), name_to_str(o)));
            } else if self.eat_word("minimizing") || self.eat_word("minimising") || self.eat_word("loss") {
                self.eat_op("=");
                let l = self.postfix_expr()?;
                options.push((opt("loss"), name_to_str(l)));
            } else if self.eat_word("in") {
                self.expect_word("batches", "")?;
                self.expect_word("of", "")?;
                options.push((opt("batch"), self.expr_bp(bp::RANGE + 1)?));
            } else if self.eat_word("batch") {
                self.eat_word("size");
                self.eat_op("=");
                options.push((opt("batch"), self.expr_bp(bp::RANGE + 1)?));
            } else if self.eat_word("reporting") || self.eat_word("report") || self.eat_word("printing") || self.eat_word("logging") {
                self.eat_word("every");
                options.push((opt("report"), self.expr_bp(bp::RANGE + 1)?));
            } else if self.at_word("verbose") || self.at_word("verbosely") {
                let t = self.bump();
                options.push((opt("report"), Expr::new(ExprKind::Int(0), t.span)));
            } else if self.at_word("silently") || self.at_word("quietly") {
                let t = self.bump();
                options.push((opt("report"), Expr::new(ExprKind::Int(-1), t.span)));
            } else {
                let span = self.peek().span;
                let found = super::describe(self.peek());
                return self.err(span, format!("unexpected {found} in the `train` statement"));
            }
            self.eat_op(",");
            self.eat_word("and");
        }
        self.end_stmt()?;
        let span = self.span_from(start);
        Ok(Stmt {
            kind: StmtKind::Train(Rc::new(TrainSpec {
                model,
                inputs,
                targets,
                options,
                span,
            })),
            span,
        })
    }
}

fn w_to_rc(w: &str) -> Rc<str> {
    Rc::from(w)
}

/// `adam` → `"adam"` so optimizers/losses can be named without calling them.
fn name_to_str(e: Expr) -> Expr {
    match e.kind {
        ExprKind::Name(n) => Expr::new(ExprKind::Str(n.as_str()), e.span),
        _ => e,
    }
}
