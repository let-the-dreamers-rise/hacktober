//! `f′`, `f″`, `derivative(f)` (forward mode) and `∇f`, `grad(f)` (reverse
//! mode).

use chimera_core::Span;

use super::dual::{self, Dual};
use super::tensor::Tensor;
use crate::interp::{Args, Interp};
use crate::value::*;

/// The `order`-th derivative of `f` at `x`, using nested dual numbers.
pub fn derivative_at(interp: &mut Interp, f: &Value, x: Value, order: u32, span: Span) -> R {
    if order == 0 {
        return interp.call_value(f.clone(), Args::new(vec![x], span));
    }
    let tag = dual::fresh_tag();
    let xd = Dual::value(tag, x, Value::Int(1));
    let y = derivative_at(interp, f, xd, order - 1, span)?;
    Ok(dual::tangent(&y, tag))
}

/// A function computing the `order`-th derivative of `f`.
pub fn derivative_fn(f: Value, order: u32) -> Value {
    let name = match order {
        1 => "f′".to_string(),
        2 => "f″".to_string(),
        n => format!("f^({n})"),
    };
    Value::native(&name, move |interp, args| {
        let Some(x) = args.get(0).cloned() else {
            return interp.type_err("a derivative needs a point to evaluate at", args.span);
        };
        if let Value::List(_) | Value::Tensor(_) | Value::Range(_) = &x {
            // Elementwise over many points (handy for plotting).
            let points = interp.collect(&x, args.span)?;
            let mut out = Vec::with_capacity(points.len());
            for p in points {
                let p = match p {
                    Value::Tensor(t) if t.numel() == 1 => Value::Float(t.item()),
                    other => other,
                };
                out.push(derivative_at(interp, &f, p, order, args.span)?);
            }
            return Ok(Value::list(out));
        }
        derivative_at(interp, &f, x, order, args.span)
    })
}

/// A function computing the gradient of a scalar-valued `f`.
///
/// `∇f(x, y)` with numbers returns `[∂f/∂x, ∂f/∂y]` (or a single number for
/// one argument); `∇f(t)` with a tensor returns a tensor of the same shape.
pub fn gradient_fn(f: Value) -> Value {
    Value::native("∇f", move |interp, args| {
        let span = args.span;
        let mut params = Vec::with_capacity(args.pos.len());
        let mut kinds = Vec::with_capacity(args.pos.len());
        for a in &args.pos {
            let (t, kind) = match a {
                Value::Tensor(t) => (t.detach(), 0u8),
                Value::List(_) | Value::Tuple(_) => (interp.to_tensor(a, span)?.detach(), 1),
                other => match other.as_f64() {
                    Some(x) => (Tensor::scalar(x), 2),
                    None => {
                        let t = other.type_name();
                        return interp
                            .type_err(format!("cannot differentiate with respect to a {t}"), span);
                    }
                },
            };
            t.set_requires_grad(true);
            params.push(t);
            kinds.push(kind);
        }
        let call_args: Vec<Value> = params.iter().cloned().map(Value::Tensor).collect();
        let y = interp.call_value(f.clone(), Args::new(call_args, span))?;
        let y = match y {
            Value::Tensor(t) => t,
            other => {
                // A constant output has zero gradient everywhere.
                if other.as_f64().is_none() {
                    let t = other.type_name();
                    return interp.type_err(
                        format!("∇ needs a function returning a number, got {t}"),
                        span,
                    );
                }
                Tensor::scalar(0.0)
            }
        };
        if y.numel() != 1 {
            return interp.value_err(
                "∇ needs a scalar-valued function (sum or mean the output first)",
                span,
            );
        }
        if y.requires_grad() {
            y.backward();
        }
        let mut grads = Vec::with_capacity(params.len());
        for (p, kind) in params.iter().zip(&kinds) {
            let g = p.grad().unwrap_or_else(|| Tensor::zeros(p.shape()));
            grads.push(match kind {
                2 => Value::Float(g.item()),
                1 => super::tensor_to_list(&g),
                _ => Value::Tensor(g),
            });
        }
        Ok(if grads.len() == 1 {
            grads.pop().unwrap_or(Value::Nil)
        } else {
            Value::list(grads)
        })
    })
}
