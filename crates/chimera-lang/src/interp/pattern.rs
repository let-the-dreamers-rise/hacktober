//! Pattern matching and destructuring.

use std::rc::Rc;

use chimera_core::Span;

use super::{Env, EnvKind, Interp};
use crate::ast::*;
use crate::symbol::Sym;
use crate::value::*;

impl Interp {
    /// Binds an irrefutable pattern (`let`, parameters, `for`); a mismatch is
    /// a `ValueError`.
    pub fn bind_pattern(
        &mut self,
        pat: &Pattern,
        value: Value,
        env: &Rc<Env>,
        span: Span,
    ) -> R<()> {
        if let Pattern::Bind(n) = pat {
            env.define(*n, value);
            return Ok(());
        }
        if self.match_into(pat, &value, env, span)? {
            Ok(())
        } else {
            let shown = super::format::repr_plain(&value);
            self.value_err(
                format!("cannot destructure {shown} with this pattern"),
                span,
            )
        }
    }

    /// Tests `value` against `pat`, binding names into `env` on success.
    pub fn match_into(
        &mut self,
        pat: &Pattern,
        value: &Value,
        env: &Rc<Env>,
        span: Span,
    ) -> R<bool> {
        match pat {
            Pattern::Wildcard => Ok(true),
            Pattern::Bind(n) => {
                env.define(*n, value.clone());
                Ok(true)
            }
            Pattern::Literal(e) => {
                let lit = self.eval(e, env)?;
                self.values_equal(&lit, value)
            }
            // OCaml/Haskell unit: `let () = print_endline "x"`.
            Pattern::Tuple(items) if items.is_empty() && matches!(value, Value::Nil) => Ok(true),
            Pattern::Tuple(items) => {
                let elems: Vec<Value> = match value {
                    Value::Tuple(t) => t.to_vec(),
                    Value::List(l) => l.borrow().clone(),
                    _ => return Ok(false),
                };
                if elems.len() != items.len() {
                    return Ok(false);
                }
                for (p, v) in items.iter().zip(&elems) {
                    if !self.match_into(p, v, env, span)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Pattern::List { items, rest } => {
                let elems: Vec<Value> = match value {
                    Value::List(l) => l.borrow().clone(),
                    Value::Tuple(t) => t.to_vec(),
                    Value::Str(s) => s.chars().map(|c| Value::str(&c.to_string())).collect(),
                    Value::Range(_) => {
                        let mut it = self.iterate(value, span)?;
                        let mut out = Vec::new();
                        while let Some(x) = it.next(self)? {
                            out.push(x);
                        }
                        out
                    }
                    _ => return Ok(false),
                };
                match rest {
                    None => {
                        // Missing trailing elements are fine when they have defaults.
                        let optional = |p: &Pattern| matches!(p, Pattern::Default(..));
                        if elems.len() > items.len()
                            || items[elems.len()..].iter().any(|p| !optional(p))
                        {
                            return Ok(false);
                        }
                        for (i, p) in items.iter().enumerate() {
                            let v = elems.get(i).cloned().unwrap_or(Value::Nil);
                            if !self.match_into(p, &v, env, span)? {
                                return Ok(false);
                            }
                        }
                        Ok(true)
                    }
                    Some((at, name)) => {
                        let before = &items[..*at];
                        let after = &items[*at..];
                        if elems.len() < before.len() + after.len() {
                            return Ok(false);
                        }
                        for (p, v) in before.iter().zip(&elems) {
                            if !self.match_into(p, v, env, span)? {
                                return Ok(false);
                            }
                        }
                        let tail_start = elems.len() - after.len();
                        for (p, v) in after.iter().zip(&elems[tail_start..]) {
                            if !self.match_into(p, v, env, span)? {
                                return Ok(false);
                            }
                        }
                        if let Some(n) = name {
                            let middle = elems[before.len()..tail_start].to_vec();
                            let rest_value = match value {
                                Value::Str(_) => {
                                    let s: String = middle
                                        .iter()
                                        .map(|v| super::format::display_plain(v, false))
                                        .collect();
                                    Value::str(&s)
                                }
                                _ => Value::list(middle),
                            };
                            env.define(*n, rest_value);
                        }
                        Ok(true)
                    }
                }
            }
            Pattern::Default(inner, default) => {
                let v = if matches!(value, Value::Nil) {
                    self.eval(default, env)?
                } else {
                    value.clone()
                };
                self.match_into(inner, &v, env, span)
            }
            Pattern::Map(entries, rest) => {
                let mut used = Vec::with_capacity(entries.len());
                for (key_expr, p) in entries {
                    let key = self.eval(key_expr, env)?;
                    let field = match value {
                        Value::Map(m) => m.borrow().get(&key),
                        Value::Instance(i) => match &key {
                            Value::Str(s) => i.get(Sym::new(s)),
                            _ => None,
                        },
                        Value::Variant(v) => match &key {
                            Value::Str(s) => {
                                let n = Sym::new(s);
                                v.info()
                                    .field_names
                                    .iter()
                                    .position(|f| *f == n)
                                    .map(|i| v.fields[i].clone())
                            }
                            _ => None,
                        },
                        _ => return Ok(false),
                    };
                    let field = match field {
                        Some(f) => f,
                        None if matches!(p, Pattern::Default(..)) => Value::Nil,
                        None => return Ok(false),
                    };
                    used.push(key);
                    if !self.match_into(p, &field, env, span)? {
                        return Ok(false);
                    }
                }
                if let Some(r) = rest {
                    let mut remaining = Dict::new();
                    let pairs: Vec<(Value, Value)> = match value {
                        Value::Map(m) => m
                            .borrow()
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                        Value::Instance(i) => i
                            .fields
                            .borrow()
                            .iter()
                            .map(|(k, v)| (Value::Str(k.as_str()), v.clone()))
                            .collect(),
                        _ => Vec::new(),
                    };
                    for (k, v) in pairs {
                        let mut taken = false;
                        for u in &used {
                            if self.values_equal(u, &k)? {
                                taken = true;
                                break;
                            }
                        }
                        if !taken {
                            remaining.insert(k, v);
                        }
                    }
                    env.define(*r, Value::Map(Rc::new(std::cell::RefCell::new(remaining))));
                }
                Ok(true)
            }
            Pattern::Ctor {
                path,
                args,
                fields,
                has_args,
            } => self.match_ctor(path, args, fields, *has_args, value, env, span),
            Pattern::Or(alts) => {
                for alt in alts {
                    let scope = Env::child(env, EnvKind::Block);
                    if self.match_into(alt, value, &scope, span)? {
                        for (n, v) in scope.local_bindings() {
                            env.define(n, v);
                        }
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Pattern::Range { lo, hi, inclusive } => {
                let lo = self.eval(lo, env)?;
                let hi = self.eval(hi, env)?;
                if !(value.is_number() || matches!(value, Value::Str(_))) {
                    return Ok(false);
                }
                let above = self
                    .compare_values(value, &lo, span)
                    .map(|o| o != std::cmp::Ordering::Less);
                let below = self.compare_values(value, &hi, span).map(|o| {
                    if *inclusive {
                        o != std::cmp::Ordering::Greater
                    } else {
                        o == std::cmp::Ordering::Less
                    }
                });
                Ok(matches!((above, below), (Ok(true), Ok(true))))
            }
            Pattern::CaseEq(e) => self.case_eq(e, value, env),
            Pattern::In(e) => {
                let container = self.eval(e, env)?;
                self.contains(&container, value, span)
            }
            Pattern::Type(t, sub) => {
                if !self.type_matches(value, t)? {
                    return Ok(false);
                }
                match sub {
                    Some(p) => self.match_into(p, value, env, span),
                    None => Ok(true),
                }
            }
            Pattern::As(p, n) => {
                if self.match_into(p, value, env, span)? {
                    env.define(*n, value.clone());
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn match_ctor(
        &mut self,
        path: &[Sym],
        args: &[Pattern],
        fields: &[(Sym, Pattern)],
        has_args: bool,
        value: &Value,
        env: &Rc<Env>,
        span: Span,
    ) -> R<bool> {
        let last = *path.last().unwrap_or(&Sym::new("_"));
        let name = last.as_str();
        // Option: `Some(x)` matches any non-nil value; `None` matches nil.
        match &*name {
            "Some" if !matches!(value, Value::Variant(v) if v.name().is("Some")) => {
                if matches!(value, Value::Nil) {
                    return Ok(false);
                }
                return match args.first() {
                    Some(p) => self.match_into(p, value, env, span),
                    None => Ok(true),
                };
            }
            "None" | "Nothing" if path.len() == 1 => return Ok(matches!(value, Value::Nil)),
            // Plain values are successes and exceptions are failures, so
            // `Ok(x)` / `Err(e)` arms also work on functions that return
            // values or raise.
            "Ok" if path.len() == 1
                && !matches!(value, Value::Variant(v) if v.name().is("Ok") || v.name().is("Err")) =>
            {
                if matches!(value, Value::Instance(i) if i.class.has_ancestor_named("Error")) {
                    return Ok(false);
                }
                return match args.first() {
                    Some(p) => self.match_into(p, value, env, span),
                    None => Ok(true),
                };
            }
            "Err"
                if path.len() == 1
                    && !matches!(value, Value::Variant(v) if v.name().is("Ok") || v.name().is("Err")) =>
            {
                if !matches!(value, Value::Instance(i) if i.class.has_ancestor_named("Error")) {
                    return Ok(false);
                }
                return match args.first() {
                    Some(p) => self.match_into(p, value, env, span),
                    None => Ok(true),
                };
            }
            _ => {}
        }
        // Enum variants match by name (and enum name when qualified).
        if let Value::Variant(v) = value {
            if v.name() != last {
                // A qualified path naming another enum, or a different variant.
                return Ok(false);
            }
            if path.len() >= 2 && !path[path.len() - 2].is(&v.ty.name.as_str()) {
                return Ok(false);
            }
            let info = v.info().clone();
            if !args.is_empty() {
                if args.len() != v.fields.len() {
                    return Ok(false);
                }
                for (p, f) in args.iter().zip(&v.fields) {
                    if !self.match_into(p, f, env, span)? {
                        return Ok(false);
                    }
                }
            }
            for (fname, p) in fields {
                let Some(i) = info.field_names.iter().position(|n| n == fname) else {
                    return Ok(false);
                };
                if !self.match_into(p, &v.fields[i], env, span)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        let resolved = self.resolve_path(path, env);
        match resolved {
            Some(Value::Class(c)) => {
                let Value::Instance(inst) = value else {
                    return Ok(false);
                };
                if !inst.class.is_subclass_of(&c) {
                    return Ok(false);
                }
                if !args.is_empty() {
                    let names = match inst.class.find_static(Sym::new("__match_args__")) {
                        Some(Value::Tuple(t)) => t
                            .iter()
                            .filter_map(|v| match v {
                                Value::Str(s) => Some(Sym::new(s)),
                                _ => None,
                            })
                            .collect(),
                        _ => {
                            let declared = inst.class.all_field_names();
                            if declared.is_empty() {
                                inst.fields.borrow().iter().map(|(n, _)| *n).collect()
                            } else {
                                declared
                            }
                        }
                    };
                    if args.len() > names.len() {
                        return Ok(false);
                    }
                    for (p, n) in args.iter().zip(&names) {
                        let v = inst.get(*n).unwrap_or(Value::Nil);
                        if !self.match_into(p, &v, env, span)? {
                            return Ok(false);
                        }
                    }
                }
                for (n, p) in fields {
                    let Some(v) = inst.get(*n) else {
                        return Ok(false);
                    };
                    if !self.match_into(p, &v, env, span)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Some(Value::Type(t)) => {
                if !self.type_tag_matches(value, t) {
                    return Ok(false);
                }
                match args.first() {
                    Some(p) => self.match_into(p, value, env, span),
                    None => Ok(true),
                }
            }
            Some(Value::Enum(_)) => Ok(false),
            Some(Value::Native(_)) if has_args => Ok(false),
            Some(constant) if !has_args => self.values_equal(&constant, value),
            Some(_) => Ok(false),
            None => {
                if !has_args && path.len() == 1 {
                    // An unknown capitalised name binds, like a variable.
                    env.define(first_of(path), value.clone());
                    return Ok(true);
                }
                Ok(false)
            }
        }
    }

    fn resolve_path(&mut self, path: &[Sym], env: &Rc<Env>) -> Option<Value> {
        let first = first_of(path);
        let mut v = env
            .get(first)
            .or_else(|| self.builtins.get(&first).cloned())?;
        for seg in &path[1..] {
            v = self.get_attr(&v, *seg, Span::default()).ok()?;
        }
        Some(v)
    }

    /// Ruby/Kotlin/C case equality (`===`): types test membership, ranges
    /// test containment, everything else compares with `==`.
    pub fn case_eq(&mut self, e: &Expr, value: &Value, env: &Rc<Env>) -> R<bool> {
        let pattern = self.eval(e, env)?;
        Ok(match &pattern {
            Value::Class(_) | Value::Type(_) | Value::Enum(_) => self.isinstance(value, &pattern),
            Value::Range(r) => match value {
                Value::Int(i) => r.contains(*i) || (r.step == 1 && *i >= r.start && *i < r.end),
                Value::Float(f) => *f >= r.start as f64 && *f < r.end as f64,
                _ => false,
            },
            Value::Func(_) | Value::Native(_) => {
                let r = self.call_value(
                    pattern.clone(),
                    super::Args::new(vec![value.clone()], e.span),
                )?;
                r.truthy()
            }
            _ => self.values_equal(&pattern, value)?,
        })
    }

    /// Evaluates a `match`/`case`/`when` expression.
    pub fn eval_match(
        &mut self,
        subject: Option<&Expr>,
        arms: &[MatchArm],
        env: &Rc<Env>,
        span: Span,
    ) -> R {
        let value = match subject {
            Some(s) => self.eval(s, env)?,
            None => Value::Bool(true),
        };
        for arm in arms {
            let scope = Env::child(env, EnvKind::Block);
            let matched = match (&arm.pattern, subject) {
                (Pattern::Wildcard, None) => true,
                (p, _) => self.match_into(p, &value, &scope, span)?,
            };
            if !matched {
                continue;
            }
            if let Some(g) = &arm.guard
                && !self.eval(g, &scope)?.truthy()
            {
                continue;
            }
            return self.eval_block(&arm.body, &scope);
        }
        Ok(Value::Nil)
    }
}

fn first_of(path: &[Sym]) -> Sym {
    path.first().copied().unwrap_or_else(|| Sym::new("_"))
}
