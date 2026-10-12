//! SQL and LINQ over in-memory collections.
//!
//! ```text
//! select name, age from people where age > 30 order by age desc limit 3
//! from p in people where p.age > 30 orderby p.name select p.name
//! insert into people (name, age) values ("Ada", 36)
//! update people set age = age + 1 where name = "Ada"
//! delete from people where age < 18
//! ```
//!
//! Keywords are case-insensitive. Inside SQL, `=` compares and `AS` names a
//! column instead of casting.

use std::rc::Rc;

use super::expr::bp;
use super::{PResult, Parser};
use crate::ast::*;
use crate::lexer::Tok;
use crate::symbol::Sym;

/// Words that end a SQL clause (and so cannot be table aliases).
const CLAUSE_WORDS: &[&str] = &[
    "from", "where", "group", "having", "order", "orderby", "limit", "offset", "join", "inner",
    "left", "right", "outer", "cross", "on", "select", "union", "into", "values", "set", "and",
    "or", "as", "asc", "desc", "ascending", "descending", "full", "natural", "by",
];

impl Parser<'_> {
    fn at_kw(&self, kw: &str) -> bool {
        self.peek().ident().is_some_and(|w| w.eq_ignore_ascii_case(kw))
    }

    fn nth_kw(&self, n: usize, kw: &str) -> bool {
        self.nth(n).ident().is_some_and(|w| w.eq_ignore_ascii_case(kw))
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.at_kw(kw) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_kw(&mut self, kw: &str) -> PResult<()> {
        if self.eat_kw(kw) {
            Ok(())
        } else {
            let span = self.peek().span;
            let found = super::describe(self.peek());
            self.err(span, format!("expected `{}`, found {found}", kw.to_uppercase()))
        }
    }

    /// Runs `f` with SQL expression rules (`=` compares, no `as` casts).
    fn sql<T>(&mut self, f: impl FnOnce(&mut Self) -> PResult<T>) -> PResult<T> {
        let saved = (self.eq_is_cmp, self.no_as_cast, self.no_brace, self.no_juxtapose);
        self.eq_is_cmp = true;
        self.no_as_cast = true;
        self.no_brace = false;
        self.no_juxtapose = true;
        let r = f(self);
        (self.eq_is_cmp, self.no_as_cast, self.no_brace, self.no_juxtapose) = saved;
        r
    }

    fn sql_expr(&mut self) -> PResult<Expr> {
        self.sql(|p| p.expr_bp(bp::PIPE))
    }

    /// Whether `select` here starts a SQL query rather than naming something.
    pub(crate) fn sql_select_ahead(&self) -> bool {
        let n = self.nth(1);
        if n.line != self.peek().line {
            return false;
        }
        !(n.is_op("(") && !n.space_before
            || n.is_op(".")
            || n.is_op("=")
            || n.is_op(",")
            || n.is_op(")")
            || matches!(n.tok, Tok::Newline | Tok::Eof))
    }

    pub(crate) fn select_query(&mut self) -> PResult<Expr> {
        let start = self.bump().span;
        let mut q = SelectQuery {
            distinct: self.eat_kw("distinct"),
            ..SelectQuery::default()
        };
        self.eat_kw("all");
        // Columns.
        loop {
            if self.at_op("*") {
                self.bump();
                q.columns.push(SelectItem::Star(None));
            } else if self.peek().ident().is_some() && self.nth_is_op(1, ".") && self.nth_is_op(2, "*") {
                let (alias, _) = self.expect_ident("")?;
                self.bump();
                self.bump();
                q.columns.push(SelectItem::Star(Some(alias)));
            } else {
                let e = self.sql_expr()?;
                let alias = self.alias()?;
                q.columns.push(SelectItem::Expr(e, alias));
            }
            if !self.eat_op(",") {
                break;
            }
            self.skip_newlines();
        }
        self.skip_query_newlines();
        if self.eat_kw("from") {
            self.sources(&mut q)?;
        }
        self.clauses(&mut q)?;
        Ok(Expr::new(ExprKind::Query(Rc::new(Query::Select(q))), self.span_from(start)))
    }

    /// Lets a query continue on following lines when they start with a
    /// clause keyword.
    fn skip_query_newlines(&mut self) {
        if self.at_newline() {
            let next = self.peek_past_newlines();
            if next
                .ident()
                .is_some_and(|w| CLAUSE_WORDS.contains(&w.to_ascii_lowercase().as_str()) && w != "and" && w != "or")
            {
                self.skip_newlines();
            }
        }
    }

    fn alias(&mut self) -> PResult<Option<Sym>> {
        if self.eat_kw("as") {
            return Ok(Some(self.alias_name()?));
        }
        if let Some(w) = self.peek().ident()
            && !CLAUSE_WORDS.contains(&w.to_ascii_lowercase().as_str())
            && self.same_line()
        {
            return Ok(Some(self.alias_name()?));
        }
        Ok(None)
    }

    fn alias_name(&mut self) -> PResult<Sym> {
        let t = self.bump();
        match &t.tok {
            Tok::Ident(w) => Ok(Sym::new(w)),
            Tok::Str(s) => Ok(Sym::new(&s.as_plain().unwrap_or_default())),
            _ => {
                let found = super::describe(&t);
                self.err(t.span, format!("expected a name, found {found}"))
            }
        }
    }

    fn source(&mut self, join: JoinKind) -> PResult<Source> {
        let expr = if self.at_op("(") {
            self.sql(|p| p.postfix_expr())?
        } else {
            self.sql(|p| p.postfix_expr())?
        };
        let alias = self.alias()?;
        Ok(Source {
            expr,
            alias,
            join,
            on: None,
        })
    }

    fn sources(&mut self, q: &mut SelectQuery) -> PResult<()> {
        q.sources.push(self.source(JoinKind::Cross)?);
        loop {
            self.skip_query_newlines();
            if self.eat_op(",") {
                q.sources.push(self.source(JoinKind::Cross)?);
                continue;
            }
            let kind = if self.at_kw("join") {
                JoinKind::Inner
            } else if self.at_kw("inner") && self.nth_kw(1, "join") {
                self.bump();
                JoinKind::Inner
            } else if self.at_kw("left") {
                self.bump();
                self.eat_kw("outer");
                JoinKind::Left
            } else if self.at_kw("cross") && self.nth_kw(1, "join") {
                self.bump();
                JoinKind::Cross
            } else {
                break;
            };
            self.expect_kw("join")?;
            let mut src = self.source(kind)?;
            self.skip_query_newlines();
            if self.eat_kw("on") {
                src.on = Some(self.sql_expr()?);
            }
            q.sources.push(src);
        }
        Ok(())
    }

    fn clauses(&mut self, q: &mut SelectQuery) -> PResult<()> {
        loop {
            self.skip_query_newlines();
            if self.eat_kw("where") {
                q.filter = Some(self.sql_expr()?);
            } else if self.at_kw("group") && self.nth_kw(1, "by") {
                self.bump();
                self.bump();
                loop {
                    q.group_by.push(self.sql_expr()?);
                    if !self.eat_op(",") {
                        break;
                    }
                }
            } else if self.eat_kw("having") {
                q.having = Some(self.sql_expr()?);
            } else if (self.at_kw("order") && self.nth_kw(1, "by")) || self.at_kw("orderby") {
                if self.eat_kw("orderby") {
                } else {
                    self.bump();
                    self.bump();
                }
                loop {
                    let e = self.sql_expr()?;
                    let desc = if self.eat_kw("desc") || self.eat_kw("descending") {
                        true
                    } else {
                        self.eat_kw("asc");
                        self.eat_kw("ascending");
                        false
                    };
                    q.order_by.push((e, desc));
                    if !self.eat_op(",") {
                        break;
                    }
                }
            } else if self.eat_kw("limit") || self.eat_kw("take") {
                q.limit = Some(self.sql_expr()?);
                if self.eat_op(",") {
                    // MySQL `LIMIT offset, count`.
                    let count = self.sql_expr()?;
                    q.offset = q.limit.take();
                    q.limit = Some(count);
                }
            } else if self.eat_kw("offset") || self.eat_kw("skip") {
                q.offset = Some(self.sql_expr()?);
                self.eat_kw("rows");
            } else {
                return Ok(());
            }
        }
    }

    /// LINQ: `from x in xs [join y in ys on a equals b] [where …]
    /// [orderby k [descending]] select e`.
    pub(crate) fn linq_query(&mut self) -> PResult<Expr> {
        let start = self.bump().span;
        let mut q = SelectQuery::default();
        let (alias, _) = self.expect_ident("for the range variable")?;
        self.expect_kw("in")?;
        let expr = self.sql(|p| p.expr_bp(bp::COMPARE + 1))?;
        q.sources.push(Source {
            expr,
            alias: Some(alias),
            join: JoinKind::Cross,
            on: None,
        });
        loop {
            self.skip_query_newlines();
            if self.at_kw("from") && self.nth(1).ident().is_some() && self.nth_kw(2, "in") {
                self.bump();
                let (a, _) = self.expect_ident("")?;
                self.bump();
                let e = self.sql(|p| p.expr_bp(bp::COMPARE + 1))?;
                q.sources.push(Source {
                    expr: e,
                    alias: Some(a),
                    join: JoinKind::Cross,
                    on: None,
                });
            } else if self.eat_kw("join") {
                let (a, _) = self.expect_ident("")?;
                self.expect_kw("in")?;
                let e = self.sql(|p| p.expr_bp(bp::COMPARE + 1))?;
                self.expect_kw("on")?;
                let l = self.sql(|p| p.expr_bp(bp::COMPARE + 1))?;
                self.expect_kw("equals")?;
                let r = self.sql(|p| p.expr_bp(bp::COMPARE + 1))?;
                let span = l.span.to(r.span);
                let on = Expr::new(ExprKind::Compare(Box::new(l), vec![(CmpOp::Eq, r)]), span);
                q.sources.push(Source {
                    expr: e,
                    alias: Some(a),
                    join: JoinKind::Inner,
                    on: Some(on),
                });
            } else if self.eat_kw("where") {
                q.filter = Some(self.sql_expr()?);
            } else if self.eat_kw("orderby") || (self.at_kw("order") && self.nth_kw(1, "by")) {
                if self.prev().is_word("order") || self.at_kw("by") {
                    self.eat_kw("by");
                }
                loop {
                    let e = self.sql_expr()?;
                    let desc = self.eat_kw("descending") || self.eat_kw("desc");
                    self.eat_kw("ascending");
                    q.order_by.push((e, desc));
                    if !self.eat_op(",") {
                        break;
                    }
                }
            } else if self.eat_kw("select") {
                q.value = Some(self.sql_expr()?);
                break;
            } else if self.eat_kw("group") {
                // `group x by key` → list of {key, items}.
                let e = self.sql_expr()?;
                self.expect_kw("by")?;
                let key = self.sql_expr()?;
                q.group_by.push(key);
                q.value = Some(e);
                break;
            } else {
                let span = self.peek().span;
                let found = super::describe(self.peek());
                return self.err(span, format!("expected `select` to finish the query, found {found}"));
            }
        }
        Ok(Expr::new(ExprKind::Query(Rc::new(Query::Select(q))), self.span_from(start)))
    }

    pub(crate) fn insert_query(&mut self) -> PResult<Expr> {
        let start = self.bump().span;
        self.expect_kw("into")?;
        let target = self.sql(|p| p.postfix_expr())?;
        let mut columns = Vec::new();
        if self.at_op("(") {
            self.bump();
            loop {
                columns.push(self.alias_name()?);
                if !self.eat_op(",") {
                    break;
                }
            }
            self.expect_op(")", "after the column list")?;
        }
        self.skip_query_newlines();
        self.expect_kw("values")?;
        let mut rows = Vec::new();
        loop {
            self.skip_newlines();
            self.expect_op("(", "to start a row of values")?;
            let row = self.nested(|p| {
                let mut row = Vec::new();
                while !p.at_op(")") && !p.at_eof() {
                    row.push(p.expr()?);
                    if !p.eat_op(",") {
                        break;
                    }
                }
                p.expect_op(")", "after the row")?;
                Ok(row)
            })?;
            rows.push(row);
            if !self.eat_op(",") {
                break;
            }
        }
        Ok(Expr::new(
            ExprKind::Query(Rc::new(Query::Insert { target, columns, rows })),
            self.span_from(start),
        ))
    }

    pub(crate) fn update_query(&mut self) -> PResult<Expr> {
        let start = self.bump().span;
        let target = self.sql(|p| p.postfix_expr())?;
        let alias = if !self.at_kw("set") { Some(self.alias_name()?) } else { None };
        self.expect_kw("set")?;
        let mut sets = Vec::new();
        loop {
            let (name, _) = self.expect_ident("for the column")?;
            if !self.eat_op("=") {
                self.expect_op(":=", "in SET")?;
            }
            let value = self.sql_expr()?;
            sets.push((name, value));
            if !self.eat_op(",") {
                break;
            }
        }
        self.skip_query_newlines();
        let filter = if self.eat_kw("where") { Some(self.sql_expr()?) } else { None };
        Ok(Expr::new(
            ExprKind::Query(Rc::new(Query::Update {
                target,
                alias,
                sets,
                filter,
            })),
            self.span_from(start),
        ))
    }

    pub(crate) fn delete_query(&mut self) -> PResult<Expr> {
        let start = self.bump().span;
        self.expect_kw("from")?;
        let target = self.sql(|p| p.postfix_expr())?;
        let alias = if self.peek().ident().is_some_and(|w| !w.eq_ignore_ascii_case("where")) && self.same_line() {
            Some(self.alias_name()?)
        } else {
            None
        };
        self.skip_query_newlines();
        let filter = if self.eat_kw("where") { Some(self.sql_expr()?) } else { None };
        Ok(Expr::new(
            ExprKind::Query(Rc::new(Query::Delete { target, alias, filter })),
            self.span_from(start),
        ))
    }
}
