//! Dual numbers for forward-mode automatic differentiation.
//!
//! `f′(x)` evaluates `f` at the dual number `x + ε` (with `ε² = 0`); the
//! coefficient of `ε` in the result is the derivative. Components are
//! ordinary [`Value`]s, so duals nest: `f″` differentiates `f′`, giving exact
//! higher-order derivatives.
//!
//! Nesting needs care ("perturbation confusion"): every differentiation gets
//! a fresh *tag*, and a dual with a newer (larger) tag always sits outside
//! duals with older tags. Arithmetic between duals of different tags treats
//! the older one as a constant with respect to the newer perturbation.

use std::cell::Cell;
use std::fmt;
use std::rc::Rc;

use crate::value::Value;

thread_local! {
    static NEXT_TAG: Cell<u64> = const { Cell::new(1) };
}

/// Allocates a new perturbation tag.
pub fn fresh_tag() -> u64 {
    NEXT_TAG.with(|c| {
        let t = c.get();
        c.set(t + 1);
        t
    })
}

/// `re + du·ε` for the perturbation `tag`.
pub struct Dual {
    /// Which differentiation this perturbation belongs to.
    pub tag: u64,
    /// Primal (value) part.
    pub re: Value,
    /// Tangent (derivative) part.
    pub du: Value,
}

impl fmt::Debug for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dual({:?} + {:?}ε{})", self.re, self.du, self.tag)
    }
}

impl Dual {
    /// Wraps the parts as a value.
    pub fn value(tag: u64, re: Value, du: Value) -> Value {
        Value::Dual(Rc::new(Dual { tag, re, du }))
    }
}

/// The highest tag among the operands, if any is a dual.
pub fn top_tag(a: &Value, b: &Value) -> Option<u64> {
    match (a, b) {
        (Value::Dual(x), Value::Dual(y)) => Some(x.tag.max(y.tag)),
        (Value::Dual(x), _) => Some(x.tag),
        (_, Value::Dual(y)) => Some(y.tag),
        _ => None,
    }
}

/// Splits `v` into (primal, tangent) with respect to `tag`.
pub fn split(v: &Value, tag: u64) -> (Value, Value) {
    match v {
        Value::Dual(d) if d.tag == tag => (d.re.clone(), d.du.clone()),
        other => (other.clone(), Value::Int(0)),
    }
}

/// The tangent of `v` with respect to `tag` (0 if `v` does not depend on it).
pub fn tangent(v: &Value, tag: u64) -> Value {
    match v {
        Value::Dual(d) if d.tag == tag => d.du.clone(),
        Value::Dual(d) if d.tag > tag => {
            // A newer perturbation wraps ours: keep its structure.
            Dual::value(d.tag, tangent(&d.re, tag), tangent(&d.du, tag))
        }
        _ => Value::Int(0),
    }
}
