//! Executing SQL and LINQ queries over in-memory collections.
//!
//! Rows are maps, class instances or plain values. Inside a query, bare
//! names resolve to columns of the current row first (so `where age > 30`
//! works), then to ordinary variables. Aggregates (`count`, `sum`, `avg`,
//! `min`, `max`, `group_concat`) are evaluated per group up front and looked
//! up by expression identity while the rest of the row is computed.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use chimera_core::Span;

use super::{Env, Interp};
use crate::ast::*;
use crate::symbol::Sym;
use crate::value::*;

type Row = Vec<(Option<Sym>, Value)>;

const AGGREGATES: &[&str] = &[
    "count",
    "sum",
    "avg",
    "average",
    "mean",
    "min",
    "max",
    "group_concat",
    "string_agg",
    "array_agg",
    "collect",
];

fn agg_name(e: &Expr) -> Option<String> {
    if let ExprKind::Call { callee, .. } = &e.kind
        && let ExprKind::Name(n) = &callee.kind
    {
        let lower = n.as_str().to_ascii_lowercase();
        if AGGREGATES.contains(&lower.as_str()) {
            return Some(lower);
        }
    }
    None
}

/// Collects aggregate calls inside an expression (not inside nested queries).
fn find_aggregates<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    if agg_name(e).is_some() {
        out.push(e);
        return;
    }
    match &e.kind {
        ExprKind::Unary(_, a) => find_aggregates(a, out),
        ExprKind::Binary(_, a, b)
        | ExprKind::And(a, b)
        | ExprKind::Or(a, b)
        | ExprKind::Coalesce(a, b) => {
            find_aggregates(a, out);
            find_aggregates(b, out);
        }
        ExprKind::Compare(a, rest) => {
            find_aggregates(a, out);
            for (_, r) in rest {
                find_aggregates(r, out);
            }
        }
        ExprKind::Call { args, callee } => {
            find_aggregates(callee, out);
            for a in args {
                if let Arg::Pos(x) | Arg::Named(_, x) = a {
                    find_aggregates(x, out);
                }
            }
        }
        ExprKind::If {
            cond,
            then,
            otherwise,
        } => {
            find_aggregates(cond, out);
            for s in then
                .stmts
                .iter()
                .chain(otherwise.iter().flat_map(|b| b.stmts.iter()))
            {
                if let StmtKind::Expr(x) = &s.kind {
                    find_aggregates(x, out);
                }
            }
        }
        ExprKind::Match { subject, arms } => {
            if let Some(s) = subject {
                find_aggregates(s, out);
            }
            for arm in arms {
                if let Some(g) = &arm.guard {
                    find_aggregates(g, out);
                }
                for s in &arm.body.stmts {
                    if let StmtKind::Expr(x) = &s.kind {
                        find_aggregates(x, out);
                    }
                }
            }
        }
        ExprKind::Interp(parts) => {
            for p in parts {
                if let InterpPart::Expr(x, _) = p {
                    find_aggregates(x, out);
                }
            }
        }
        _ => {}
    }
}

fn expr_key(e: &Expr) -> usize {
    e as *const Expr as usize
}

impl Interp {
    /// Looks up a bare name among the current query rows.
    pub(crate) fn row_lookup(&self, n: Sym) -> Option<Value> {
        let rows = self.rows.last()?;
        for (alias, row) in rows {
            if *alias == Some(n) {
                return Some(row.clone());
            }
        }
        for (_, row) in rows {
            let found = match row {
                Value::Map(m) => m.borrow().get_str(&n.as_str()),
                Value::Instance(i) => i.get(n),
                Value::Variant(v) => {
                    let info = v.info();
                    info.field_names
                        .iter()
                        .position(|f| *f == n)
                        .map(|i| v.fields[i].clone())
                }
                _ => None,
            };
            if found.is_some() {
                return found;
            }
        }
        None
    }

    fn with_row<T>(&mut self, row: Row, f: impl FnOnce(&mut Self) -> R<T>) -> R<T> {
        self.rows.push(row);
        let r = f(self);
        self.rows.pop();
        r
    }

    fn eval_in_row(&mut self, e: &Expr, row: &Row, env: &Rc<Env>) -> R {
        self.with_row(row.clone(), |interp| interp.eval(e, env))
    }

    /// Runs a query.
    pub fn eval_query(&mut self, q: &Query, env: &Rc<Env>, span: Span) -> R {
        match q {
            Query::Select(s) => self.select(s, env, span),
            Query::Insert {
                target,
                columns,
                rows,
            } => {
                let t = self.eval(target, env)?;
                let Value::List(list) = &t else {
                    return self.type_err("INSERT needs a list to insert into", target.span);
                };
                let template: Vec<Value> = match list.borrow().first() {
                    Some(Value::Map(m)) => m.borrow().keys(),
                    _ => Vec::new(),
                };
                let mut count = 0;
                for row in rows {
                    let mut values = Vec::with_capacity(row.len());
                    for e in row {
                        values.push(self.eval(e, env)?);
                    }
                    let keys: Vec<Value> = if !columns.is_empty() {
                        columns.iter().map(|c| Value::Str(c.as_str())).collect()
                    } else {
                        template.clone()
                    };
                    let item = if keys.is_empty() {
                        if values.len() == 1 {
                            values.pop().unwrap_or(Value::Nil)
                        } else {
                            Value::tuple(values)
                        }
                    } else {
                        if keys.len() != values.len() {
                            return self.value_err(
                                format!(
                                    "INSERT has {} columns but {} values",
                                    keys.len(),
                                    values.len()
                                ),
                                span,
                            );
                        }
                        let mut d = Dict::new();
                        for (k, v) in keys.into_iter().zip(values) {
                            d.insert(k, v);
                        }
                        Value::Map(Rc::new(RefCell::new(d)))
                    };
                    list.borrow_mut().push(item);
                    count += 1;
                }
                Ok(Value::Int(count))
            }
            Query::Update {
                target,
                alias,
                sets,
                filter,
            } => {
                let t = self.eval(target, env)?;
                let Value::List(list) = &t else {
                    return self.type_err("UPDATE needs a list of rows", target.span);
                };
                let alias = alias.or_else(|| name_of(target));
                let rows = list.borrow().clone();
                let mut count = 0;
                for row in rows {
                    let r: Row = vec![(alias, row.clone())];
                    if let Some(f) = filter
                        && !self.eval_in_row(f, &r, env)?.truthy()
                    {
                        continue;
                    }
                    let mut updates = Vec::with_capacity(sets.len());
                    for (col, e) in sets {
                        updates.push((*col, self.eval_in_row(e, &r, env)?));
                    }
                    for (col, v) in updates {
                        match &row {
                            Value::Map(m) => {
                                m.borrow_mut().insert(Value::Str(col.as_str()), v);
                            }
                            Value::Instance(i) => i.set(col, v),
                            _ => return self.type_err("UPDATE rows must be maps or objects", span),
                        }
                    }
                    count += 1;
                }
                Ok(Value::Int(count))
            }
            Query::Delete {
                target,
                alias,
                filter,
            } => {
                let t = self.eval(target, env)?;
                let Value::List(list) = &t else {
                    return self.type_err("DELETE needs a list of rows", target.span);
                };
                let alias = alias.or_else(|| name_of(target));
                let rows = list.borrow().clone();
                let mut keep = Vec::with_capacity(rows.len());
                let mut count = 0;
                for row in rows {
                    let doomed = match filter {
                        Some(f) => self
                            .eval_in_row(f, &vec![(alias, row.clone())], env)?
                            .truthy(),
                        None => true,
                    };
                    if doomed {
                        count += 1;
                    } else {
                        keep.push(row);
                    }
                }
                *list.borrow_mut() = keep;
                Ok(Value::Int(count))
            }
        }
    }

    fn select(&mut self, q: &SelectQuery, env: &Rc<Env>, span: Span) -> R {
        // 1. Sources and joins.
        let mut rows: Vec<Row> = vec![Vec::new()];
        for (i, src) in q.sources.iter().enumerate() {
            let coll = self.eval(&src.expr, env)?;
            let items = self.collect(&coll, src.expr.span)?;
            let alias = src.alias.or_else(|| name_of(&src.expr));
            let mut next = Vec::new();
            for base in &rows {
                let mut matched = false;
                for item in &items {
                    let mut r = base.clone();
                    r.push((alias, item.clone()));
                    if i > 0
                        && let Some(on) = &src.on
                        && !self.eval_in_row(on, &r, env)?.truthy()
                    {
                        continue;
                    }
                    matched = true;
                    next.push(r);
                }
                if !matched && src.join == JoinKind::Left {
                    let mut r = base.clone();
                    r.push((alias, Value::Nil));
                    next.push(r);
                }
            }
            rows = next;
        }
        if q.sources.is_empty() {
            rows = vec![Vec::new()];
        }

        // 2. WHERE.
        if let Some(f) = &q.filter {
            let mut kept = Vec::with_capacity(rows.len());
            for r in rows {
                if self.eval_in_row(f, &r, env)?.truthy() {
                    kept.push(r);
                }
            }
            rows = kept;
        }

        // 3. Aggregates and grouping.
        let mut aggs: Vec<&Expr> = Vec::new();
        for c in &q.columns {
            if let SelectItem::Expr(e, _) = c {
                find_aggregates(e, &mut aggs);
            }
        }
        if let Some(h) = &q.having {
            find_aggregates(h, &mut aggs);
        }
        for (e, _) in &q.order_by {
            find_aggregates(e, &mut aggs);
        }
        if let Some(v) = &q.value {
            find_aggregates(v, &mut aggs);
        }
        let grouping = !q.group_by.is_empty() || !aggs.is_empty();
        let groups: Vec<Vec<Row>> = if grouping {
            let mut order: Vec<Vec<Value>> = Vec::new();
            let mut buckets: HashMap<Key, usize> = HashMap::new();
            let mut grouped: Vec<Vec<Row>> = Vec::new();
            for r in rows {
                let mut key = Vec::with_capacity(q.group_by.len());
                for g in &q.group_by {
                    key.push(self.eval_in_row(g, &r, env)?);
                }
                let k = key_of(&Value::tuple(key.clone())).unwrap_or(Key::Nil);
                match buckets.get(&k) {
                    Some(&i) => grouped[i].push(r),
                    None => {
                        buckets.insert(k, grouped.len());
                        order.push(key);
                        grouped.push(vec![r]);
                    }
                }
            }
            if grouped.is_empty() && q.group_by.is_empty() {
                // Aggregates over no rows still produce one row (count = 0).
                grouped.push(Vec::new());
            }
            grouped
        } else {
            rows.into_iter().map(|r| vec![r]).collect()
        };

        // 4. Project, HAVING, ORDER BY keys.
        let mut out: Vec<(Value, Vec<Value>)> = Vec::new();
        for group in &groups {
            let mut agg_values = HashMap::new();
            for a in &aggs {
                let v = self.aggregate(a, group, env)?;
                agg_values.insert(expr_key(a), v);
            }
            let first: Row = group.first().cloned().unwrap_or_default();
            self.aggregates.push(agg_values);
            let produced = (|| -> R<Option<(Value, Vec<Value>)>> {
                if let Some(h) = &q.having
                    && !self.eval_in_row(h, &first, env)?.truthy()
                {
                    return Ok(None);
                }
                let value = if let Some(v) = &q.value {
                    if !q.group_by.is_empty() && agg_name(v).is_none() && !contains_aggregate(v) {
                        // LINQ `group x by key`.
                        let key = self.eval_in_row(&q.group_by[0], &first, env)?;
                        let mut items = Vec::new();
                        for r in group {
                            items.push(self.eval_in_row(v, r, env)?);
                        }
                        Value::map_from(vec![("key", key), ("items", Value::list(items))])
                    } else {
                        self.eval_in_row(v, &first, env)?
                    }
                } else {
                    self.project(q, &first, env)?
                };
                let mut sort_row = vec![(None, value.clone())];
                sort_row.extend(first.iter().cloned());
                let mut keys = Vec::with_capacity(q.order_by.len());
                for (e, _) in &q.order_by {
                    keys.push(self.eval_in_row(e, &sort_row, env)?);
                }
                Ok(Some((value, keys)))
            })();
            self.aggregates.pop();
            if let Some(p) = produced? {
                out.push(p);
            }
        }

        // 5. DISTINCT.
        if q.distinct {
            let mut seen = Vec::new();
            let mut unique = Vec::new();
            for (v, k) in out {
                let mut dup = false;
                for s in &seen {
                    if self.values_equal(s, &v)? {
                        dup = true;
                        break;
                    }
                }
                if !dup {
                    seen.push(v.clone());
                    unique.push((v, k));
                }
            }
            out = unique;
        }

        // 6. ORDER BY (stable).
        if !q.order_by.is_empty() {
            let mut err = None;
            let dirs: Vec<bool> = q.order_by.iter().map(|(_, d)| *d).collect();
            let mut indexed: Vec<(usize, (Value, Vec<Value>))> =
                out.into_iter().enumerate().collect();
            // Insertion-free merge sort via sort_by with an error slot.
            indexed.sort_by(|(ia, (_, ka)), (ib, (_, kb))| {
                for (i, (x, y)) in ka.iter().zip(kb).enumerate() {
                    let o = match (x, y) {
                        (Value::Nil, Value::Nil) => std::cmp::Ordering::Equal,
                        (Value::Nil, _) => std::cmp::Ordering::Less,
                        (_, Value::Nil) => std::cmp::Ordering::Greater,
                        _ => match self.compare_values(x, y, span) {
                            Ok(o) => o,
                            Err(e) => {
                                err.get_or_insert(e);
                                std::cmp::Ordering::Equal
                            }
                        },
                    };
                    let o = if dirs[i] { o.reverse() } else { o };
                    if o != std::cmp::Ordering::Equal {
                        return o;
                    }
                }
                ia.cmp(ib)
            });
            if let Some(e) = err {
                return Err(e);
            }
            out = indexed.into_iter().map(|(_, p)| p).collect();
        }

        // 7. OFFSET / LIMIT.
        let mut values: Vec<Value> = out.into_iter().map(|(v, _)| v).collect();
        if let Some(o) = &q.offset {
            let n = self.eval(o, env)?.as_int().unwrap_or(0).max(0) as usize;
            values = values.into_iter().skip(n).collect();
        }
        if let Some(l) = &q.limit {
            let n = self.eval(l, env)?.as_int().unwrap_or(0).max(0) as usize;
            values.truncate(n);
        }
        Ok(Value::list(values))
    }

    fn project(&mut self, q: &SelectQuery, row: &Row, env: &Rc<Env>) -> R {
        let mut d = Dict::new();
        for col in &q.columns {
            match col {
                SelectItem::Star(alias) => {
                    for (a, r) in row {
                        if alias.is_some() && a != alias {
                            continue;
                        }
                        match r {
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
                                let name =
                                    a.map_or_else(|| "value".into(), |s| s.as_str().to_string());
                                d.insert(Value::str(&name), other.clone());
                            }
                        }
                    }
                }
                SelectItem::Expr(e, alias) => {
                    let v = self.eval_in_row(e, row, env)?;
                    let name = match alias {
                        Some(a) => a.as_str().to_string(),
                        None => self.column_name(e),
                    };
                    d.insert(Value::str(&name), v);
                }
            }
        }
        // `SELECT expr` with a single unnamed scalar and no FROM yields rows
        // too; keep the map form for consistency.
        Ok(Value::Map(Rc::new(RefCell::new(d))))
    }

    fn column_name(&self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Name(n) => n.as_str().to_string(),
            ExprKind::Member { name, .. } => name.as_str().to_string(),
            _ => self
                .files
                .get(self.cur_file as usize)
                .and_then(|f| f.text().get(e.span.range()))
                .map(|s| s.trim().to_owned())
                .unwrap_or_else(|| "?column?".into()),
        }
    }

    fn aggregate(&mut self, call: &Expr, group: &[Row], env: &Rc<Env>) -> R {
        let Some(name) = agg_name(call) else {
            return Ok(Value::Nil);
        };
        let ExprKind::Call { args, .. } = &call.kind else {
            return Ok(Value::Nil);
        };
        let arg = args.iter().find_map(|a| match a {
            Arg::Pos(e) => Some(e),
            _ => None,
        });
        let star = arg.is_none_or(|e| matches!(&e.kind, ExprKind::Str(s) if &**s == "*"));
        if name == "count" && star {
            return Ok(Value::Int(group.len() as i64));
        }
        let Some(arg) = arg else {
            return Ok(Value::Nil);
        };
        let mut values = Vec::with_capacity(group.len());
        for r in group {
            let v = self.eval_in_row(arg, r, env)?;
            if !matches!(v, Value::Nil) {
                values.push(v);
            }
        }
        let span = call.span;
        Ok(match name.as_str() {
            "count" => Value::Int(values.len() as i64),
            "sum" => {
                let mut acc = Value::Int(0);
                for v in values {
                    acc = self.binary_op(BinOp::Add, acc, v, span)?;
                }
                acc
            }
            "avg" | "average" | "mean" => {
                if values.is_empty() {
                    return Ok(Value::Nil);
                }
                let n = values.len();
                let mut acc = Value::Int(0);
                for v in values {
                    acc = self.binary_op(BinOp::Add, acc, v, span)?;
                }
                self.binary_op(BinOp::Div, acc, Value::Int(n as i64), span)?
            }
            "min" | "max" => {
                let mut best: Option<Value> = None;
                for v in values {
                    best = Some(match best {
                        None => v,
                        Some(b) => {
                            let o = self.compare_values(&v, &b, span)?;
                            let better = if name == "min" { o.is_lt() } else { o.is_gt() };
                            if better { v } else { b }
                        }
                    });
                }
                best.unwrap_or(Value::Nil)
            }
            "group_concat" | "string_agg" => {
                let sep = match args.get(1) {
                    Some(Arg::Pos(e)) => self.eval(e, env)?,
                    _ => Value::str(","),
                };
                let sep = self.to_str(&sep)?;
                let mut parts = Vec::with_capacity(values.len());
                for v in &values {
                    parts.push(self.to_str(v)?);
                }
                Value::str(&parts.join(&sep))
            }
            _ => Value::list(values),
        })
    }
}

fn contains_aggregate(e: &Expr) -> bool {
    let mut v = Vec::new();
    find_aggregates(e, &mut v);
    !v.is_empty()
}

fn name_of(e: &Expr) -> Option<Sym> {
    match &e.kind {
        ExprKind::Name(n) => Some(*n),
        ExprKind::Member { name, .. } => Some(*name),
        _ => None,
    }
}

/// Precomputed aggregate for `e`, if we are inside a grouped query.
pub(crate) fn precomputed(interp: &Interp, e: &Expr) -> Option<Value> {
    interp
        .aggregates
        .last()
        .and_then(|m| m.get(&expr_key(e)).cloned())
}
