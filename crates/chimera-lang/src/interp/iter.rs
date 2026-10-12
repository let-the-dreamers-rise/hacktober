//! Iteration over values.

use chimera_core::Span;

use super::{Args, Interp, sym};
use crate::value::*;

/// An in-progress iteration.
#[derive(Debug)]
pub enum ValueIter {
    /// A materialised sequence.
    Items(std::vec::IntoIter<Value>),
    /// A generator, resumed on demand.
    Gen(Value),
    /// A Python/JS iterator object: `__next__()` / `next()` called on demand.
    Protocol {
        /// The bound `next` method.
        next: Value,
        /// Where the iteration started.
        span: Span,
    },
    /// A lazy integer range.
    Range {
        /// Next value.
        cur: i64,
        /// Exclusive end.
        end: i64,
        /// Step.
        step: i64,
    },
}

impl ValueIter {
    /// The next element, if any.
    pub fn next(&mut self, interp: &mut Interp) -> R<Option<Value>> {
        Ok(match self {
            ValueIter::Items(it) => it.next(),
            ValueIter::Gen(g) => {
                let Value::Object(o) = g else { return Ok(None) };
                let o = o.clone();
                if let Some(g) = o.as_any().downcast_ref::<super::generator::Generator>() {
                    g.resume(interp, Value::Nil, Span::default())?
                } else if let Some(i) = o.as_any().downcast_ref::<super::generator::IterObject>() {
                    i.advance(interp)?
                } else {
                    None
                }
            }
            ValueIter::Protocol { next, span } => {
                let span = *span;
                match interp.call_value(next.clone(), Args::new(Vec::new(), span)) {
                    Ok(Value::Map(m)) if m.borrow().get_str("done").is_some() => {
                        let m = m.borrow();
                        if m.get_str("done").is_some_and(|d| d.truthy()) {
                            None
                        } else {
                            Some(m.get_str("value").unwrap_or(Value::Nil))
                        }
                    }
                    Ok(v) => Some(v),
                    Err(super::Flow::Throw(t)) if matches!(&t.value, Value::Instance(i) if i.class.has_ancestor_named("StopIteration")) => {
                        None
                    }
                    Err(e) => return Err(e),
                }
            }
            ValueIter::Range { cur, end, step } => {
                let more = if *step > 0 { *cur < *end } else { *cur > *end };
                if more {
                    let v = *cur;
                    *cur += *step;
                    Some(Value::Int(v))
                } else {
                    None
                }
            }
        })
    }
}

impl Interp {
    /// Starts iterating over a value.
    pub fn iterate(&mut self, v: &Value, span: Span) -> R<ValueIter> {
        let items = match v {
            Value::List(l) => l.borrow().clone(),
            Value::Tuple(t) => t.to_vec(),
            Value::Range(r) => {
                return Ok(ValueIter::Range {
                    cur: r.start,
                    end: r.end,
                    step: r.step,
                });
            }
            Value::Str(s) => s.chars().map(|c| Value::str(&c.to_string())).collect(),
            Value::Map(m) | Value::Set(m) => m.borrow().keys(),
            Value::Int(n) => {
                return Ok(ValueIter::Range {
                    cur: 0,
                    end: *n,
                    step: 1,
                });
            }
            Value::Tensor(t) => {
                if t.ndim() == 0 {
                    vec![Value::Float(t.item())]
                } else if t.ndim() == 1 {
                    t.data().into_iter().map(Value::Float).collect()
                } else {
                    let rows = t.shape()[0];
                    let mut out = Vec::with_capacity(rows);
                    for i in 0..rows {
                        let row = t
                            .slice(&[crate::ml::tensor::SliceSpec::Index(i as i64)])
                            .map_err(|e| self.error("ValueError", e, span))?;
                        out.push(Value::Tensor(row));
                    }
                    out
                }
            }
            Value::Instance(i) => {
                for name in ["__iter__", "iter", "iterator", "to_list", "to_a", "items"] {
                    if let Some(m) = i.class.find_method(sym(name)) {
                        let bound = Value::Bound(std::rc::Rc::new(Bound {
                            receiver: v.clone(),
                            func: m,
                        }));
                        let r = self.call_value(bound, Args::new(Vec::new(), span))?;
                        if matches!(r, Value::Instance(_)) {
                            return self.drain_iterator(&r, span);
                        }
                        return self.iterate(&r, span);
                    }
                }
                let n = i.class.name;
                return self.type_err(
                    format!("`{n}` objects are not iterable (define `__iter__`)"),
                    span,
                );
            }
            Value::Object(o)
                if o.as_any().is::<super::generator::Generator>()
                    || o.as_any().is::<super::generator::IterObject>() =>
            {
                return Ok(ValueIter::Gen(v.clone()));
            }
            Value::Object(o) => {
                let this = v.clone();
                match o.call_method(self, "__iter__", Args::new(Vec::new(), span), &this) {
                    Some(r) => {
                        let r = r?;
                        return self.iterate(&r, span);
                    }
                    None => {
                        let t = o.type_name().to_owned();
                        return self.type_err(format!("`{t}` is not iterable"), span);
                    }
                }
            }
            other => {
                let t = other.type_name();
                return self.type_err(format!("`{t}` is not iterable"), span);
            }
        };
        Ok(ValueIter::Items(items.into_iter()))
    }

    /// Iterates a Python/JS-style iterator object (`__next__`/`next`)
    /// lazily.
    fn drain_iterator(&mut self, it: &Value, span: Span) -> R<ValueIter> {
        let Value::Instance(inst) = it else {
            return self.iterate(it, span);
        };
        let Some(next) = inst
            .class
            .find_method(sym("__next__"))
            .or_else(|| inst.class.find_method(sym("next")))
        else {
            let n = inst.class.name;
            return self.type_err(
                format!("`{n}` iterator has no `__next__`/`next` method"),
                span,
            );
        };
        let bound = Value::Bound(std::rc::Rc::new(Bound {
            receiver: it.clone(),
            func: next,
        }));
        Ok(ValueIter::Protocol { next: bound, span })
    }

    /// Collects a value's elements into a vector.
    pub fn collect(&mut self, v: &Value, span: Span) -> R<Vec<Value>> {
        let mut it = self.iterate(v, span)?;
        let mut out = Vec::new();
        while let Some(x) = it.next(self)? {
            out.push(x);
        }
        Ok(out)
    }
}
