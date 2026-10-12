//! Math and randomness. Every unary function works on numbers, lists
//! (elementwise), tensors (with autograd) and dual numbers (so `f′` can
//! differentiate through it).

use std::rc::Rc;

use chimera_core::Span;

use super::{arg, constant, def, float_arg, int_arg};
use crate::ast::BinOp;
use crate::bigint::BigInt;
use crate::interp::{Args, Interp};
use crate::ml::dual::Dual;
use crate::ml::tensor::Unary;
use crate::value::*;

/// A math function and the names it goes by.
type UnaryEntry = (&'static [&'static str], fn(f64) -> f64);

/// Unary math functions: (names, f64 implementation).
const UNARY: &[UnaryEntry] = &[
    (&["sqrt", "Sqrt", "sqrtf"], f64::sqrt),
    (&["cbrt"], f64::cbrt),
    (&["exp", "Exp"], f64::exp),
    (&["log", "ln", "Log"], f64::ln),
    (&["log2"], f64::log2),
    (&["log10", "Log10"], f64::log10),
    (&["sin", "Sin"], f64::sin),
    (&["cos", "Cos"], f64::cos),
    (&["tan", "Tan"], f64::tan),
    (&["asin", "arcsin"], f64::asin),
    (&["acos", "arccos"], f64::acos),
    (&["atan", "arctan"], f64::atan),
    (&["sinh"], f64::sinh),
    (&["cosh"], f64::cosh),
    (&["tanh"], f64::tanh),
    (&["sigmoid", "logistic"], |x| 1.0 / (1.0 + (-x).exp())),
    (&["relu"], |x| x.max(0.0)),
    (
        &["softplus"],
        |x| if x > 30.0 { x } else { x.exp().ln_1p() },
    ),
    (&["log1p"], f64::ln_1p),
    (&["expm1"], f64::exp_m1),
    (&["radians", "to_radians", "deg2rad"], f64::to_radians),
    (&["degrees", "to_degrees", "rad2deg"], f64::to_degrees),
];

fn tensor_unary(name: &str) -> Option<Unary> {
    Some(match name {
        "exp" => Unary::Exp,
        "log" | "ln" => Unary::Log,
        "tanh" => Unary::Tanh,
        "sigmoid" | "logistic" => Unary::Sigmoid,
        "relu" => Unary::Relu,
        "sqrt" => Unary::Sqrt,
        "abs" => Unary::Abs,
        "sin" => Unary::Sin,
        "cos" => Unary::Cos,
        "softplus" => Unary::Softplus,
        _ => return None,
    })
}

/// Applies a named unary function to any supported value.
fn apply(interp: &mut Interp, name: &'static str, f: fn(f64) -> f64, x: Value, span: Span) -> R {
    match x {
        Value::Dual(d) => dual_rule(interp, name, &d, span),
        Value::Tensor(t) => Ok(Value::Tensor(match tensor_unary(name) {
            Some(u) => t.unary(u),
            None => t.map(f),
        })),
        Value::List(l) => {
            let items = l.borrow().clone();
            let mut out = Vec::with_capacity(items.len());
            for v in items {
                out.push(apply(interp, name, f, v, span)?);
            }
            Ok(Value::list(out))
        }
        other => match other.as_f64() {
            Some(v) => {
                if matches!(name, "sqrt") && v < 0.0 {
                    return Err(interp.error(
                        "ValueError",
                        "math domain error: sqrt of a negative number",
                        span,
                    ));
                }
                if matches!(name, "log" | "ln" | "log2" | "log10") && v <= 0.0 {
                    return Err(interp.error(
                        "ValueError",
                        "math domain error: log of a non-positive number",
                        span,
                    ));
                }
                Ok(Value::Float(f(v)))
            }
            None => {
                let t = other.type_name();
                interp.type_err(format!("`{name}` needs a number, got {t}"), span)
            }
        },
    }
}

/// Forward-mode derivative rules: f(re + du·ε) = f(re) + f′(re)·du·ε.
fn dual_rule(interp: &mut Interp, name: &'static str, d: &Rc<Dual>, span: Span) -> R {
    let re = d.re.clone();
    let du = d.du.clone();
    let mul = |interp: &mut Interp, a: Value, b: Value| interp.binary_op(BinOp::Mul, a, b, span);
    let div = |interp: &mut Interp, a: Value, b: Value| interp.binary_op(BinOp::Div, a, b, span);
    let fx = interp.math_fn(name, re.clone(), span)?;
    let deriv = match name {
        "sqrt" => {
            let two_fx = mul(interp, Value::Int(2), fx.clone())?;
            div(interp, Value::Int(1), two_fx)?
        }
        "exp" => fx.clone(),
        "log" | "ln" => div(interp, Value::Int(1), re.clone())?,
        "log2" => {
            let r = mul(interp, re.clone(), Value::Float(std::f64::consts::LN_2))?;
            div(interp, Value::Int(1), r)?
        }
        "log10" => {
            let r = mul(interp, re.clone(), Value::Float(std::f64::consts::LN_10))?;
            div(interp, Value::Int(1), r)?
        }
        "sin" => interp.math_fn("cos", re.clone(), span)?,
        "cos" => {
            let s = interp.math_fn("sin", re.clone(), span)?;
            interp.unary_op(crate::ast::UnOp::Neg, s, span)?
        }
        "tan" => {
            let c = interp.math_fn("cos", re.clone(), span)?;
            let c2 = mul(interp, c.clone(), c)?;
            div(interp, Value::Int(1), c2)?
        }
        "tanh" => {
            let t2 = mul(interp, fx.clone(), fx.clone())?;
            interp.binary_op(BinOp::Sub, Value::Int(1), t2, span)?
        }
        "sigmoid" | "logistic" => {
            let one_minus = interp.binary_op(BinOp::Sub, Value::Int(1), fx.clone(), span)?;
            mul(interp, fx.clone(), one_minus)?
        }
        "sinh" => interp.math_fn("cosh", re.clone(), span)?,
        "cosh" => interp.math_fn("sinh", re.clone(), span)?,
        "asin" | "acos" => {
            let r2 = mul(interp, re.clone(), re.clone())?;
            let inner = interp.binary_op(BinOp::Sub, Value::Int(1), r2, span)?;
            let s = interp.math_fn("sqrt", inner, span)?;
            let base = div(interp, Value::Int(1), s)?;
            if name == "acos" {
                interp.unary_op(crate::ast::UnOp::Neg, base, span)?
            } else {
                base
            }
        }
        "atan" => {
            let r2 = mul(interp, re.clone(), re.clone())?;
            let denom = interp.binary_op(BinOp::Add, Value::Int(1), r2, span)?;
            div(interp, Value::Int(1), denom)?
        }
        "relu" => {
            let positive = re.as_f64().is_some_and(|x| x > 0.0);
            Value::Int(i64::from(positive))
        }
        "softplus" => interp.math_fn("sigmoid", re.clone(), span)?,
        "cbrt" => {
            let c2 = mul(interp, fx.clone(), fx.clone())?;
            let three_c2 = mul(interp, Value::Int(3), c2)?;
            div(interp, Value::Int(1), three_c2)?
        }
        "log1p" => {
            let denom = interp.binary_op(BinOp::Add, Value::Int(1), re.clone(), span)?;
            div(interp, Value::Int(1), denom)?
        }
        "expm1" => interp.math_fn("exp", re.clone(), span)?,
        "radians" | "to_radians" | "deg2rad" => Value::Float(std::f64::consts::PI / 180.0),
        "degrees" | "to_degrees" | "rad2deg" => Value::Float(180.0 / std::f64::consts::PI),
        _ => Value::Int(0),
    };
    let new_du = mul(interp, deriv, du)?;
    Ok(Dual::value(d.tag, fx, new_du))
}

fn round_value(interp: &mut Interp, x: &Value, digits: Option<i64>, span: Span) -> R {
    match x {
        Value::Int(_) | Value::Big(_) if digits.is_none_or(|d| d >= 0) => Ok(x.clone()),
        Value::Dual(d) => round_value(interp, &d.re.clone(), digits, span),
        Value::Tensor(t) => {
            let p = 10f64.powi(digits.unwrap_or(0) as i32);
            Ok(Value::Tensor(t.map(|v| (v * p).round_ties_even() / p)))
        }
        other => {
            let Some(f) = other.as_f64() else {
                let t = other.type_name();
                return interp.type_err(format!("round() needs a number, got {t}"), span);
            };
            match digits {
                None => {
                    // Python rounds half to even.
                    let r = f.round_ties_even();
                    if r.abs() < 9.0e18 {
                        Ok(Value::Int(r as i64))
                    } else {
                        Ok(Value::from_big(BigInt::from_f64(r).unwrap_or_default()))
                    }
                }
                Some(d) if (0..=17).contains(&d) && f.is_finite() => {
                    // Round the exact decimal value, like Python:
                    // round(2.675, 2) == 2.67 because 2.675 is really 2.67499…
                    let s = format!("{f:.*}", d as usize);
                    Ok(Value::Float(s.parse().unwrap_or(f)))
                }
                Some(d) => {
                    let p = 10f64.powi(d as i32);
                    Ok(Value::Float((f * p).round_ties_even() / p))
                }
            }
        }
    }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Installs math and random functions.
pub fn install(interp: &mut Interp) {
    for (names, f) in UNARY {
        let name: &'static str = names[0];
        let f = *f;
        def(interp, names, move |interp, args| {
            let x = arg(interp, &args, 0, name)?;
            if name == "log" && args.len() > 1 {
                // log(x, base)
                let base = float_arg(interp, &args, 1, "log")?;
                let ln = apply(interp, "log", f64::ln, x, args.span)?;
                return interp.binary_op(BinOp::Div, ln, Value::Float(base.ln()), args.span);
            }
            apply(interp, name, f, x, args.span)
        });
    }
    def(
        interp,
        &["abs", "Abs", "fabs", "absolute"],
        |interp, args| {
            let x = arg(interp, &args, 0, "abs")?;
            match x {
                Value::Int(i) => Ok(i
                    .checked_abs()
                    .map(Value::Int)
                    .unwrap_or_else(|| Value::from_big(BigInt::from_i64(i).abs()))),
                Value::Big(b) => Ok(Value::from_big(b.abs())),
                Value::Dual(d) => {
                    let neg = d.re.as_f64().is_some_and(|v| v < 0.0);
                    if neg {
                        interp.unary_op(crate::ast::UnOp::Neg, Value::Dual(d), args.span)
                    } else {
                        Ok(Value::Dual(d))
                    }
                }
                Value::Tensor(t) => Ok(Value::Tensor(t.unary(Unary::Abs))),
                Value::Instance(_) => {
                    match interp.call_overload(&x, &["__abs__", "abs"], Vec::new(), args.span)? {
                        Some(v) => Ok(v),
                        None => interp.type_err("abs() is not defined for this object", args.span),
                    }
                }
                other => match other.as_f64() {
                    Some(f) => Ok(Value::Float(f.abs())),
                    None => {
                        let t = other.type_name();
                        interp.type_err(format!("abs() needs a number, got {t}"), args.span)
                    }
                },
            }
        },
    );
    for (names, op) in [
        (&["floor", "Floor"][..], 0u8),
        (&["ceil", "ceiling", "Ceiling"], 1),
        (&["trunc", "truncate", "Truncate"], 2),
    ] {
        def(interp, names, move |interp, args| {
            let x = arg(interp, &args, 0, "floor")?;
            let x = match x {
                Value::Dual(d) => d.re.clone(),
                other => other,
            };
            if let Value::Tensor(t) = &x {
                return Ok(Value::Tensor(t.map(match op {
                    0 => f64::floor,
                    1 => f64::ceil,
                    _ => f64::trunc,
                })));
            }
            if matches!(x, Value::Int(_) | Value::Big(_)) {
                return Ok(x);
            }
            let Some(f) = x.as_f64() else {
                return interp.type_err("expected a number", args.span);
            };
            let r = match op {
                0 => f.floor(),
                1 => f.ceil(),
                _ => f.trunc(),
            };
            if !r.is_finite() {
                return Ok(Value::Float(r));
            }
            Ok(Value::from_big(BigInt::from_f64(r).unwrap_or_default()))
        });
    }
    def(interp, &["round", "Round", "rint"], |interp, args| {
        let x = arg(interp, &args, 0, "round")?;
        let digits = args.arg(1, "ndigits").and_then(|d| d.as_int());
        round_value(interp, &x, digits, args.span)
    });
    def(
        interp,
        &["sign", "signum", "sgn", "Sign"],
        |interp, args| {
            let x = arg(interp, &args, 0, "sign")?;
            let f = x.as_f64().unwrap_or(0.0);
            Ok(match x {
                Value::Float(_) => Value::Float(if f > 0.0 {
                    1.0
                } else if f < 0.0 {
                    -1.0
                } else {
                    0.0
                }),
                _ => Value::Int(if f > 0.0 {
                    1
                } else if f < 0.0 {
                    -1
                } else {
                    0
                }),
            })
        },
    );
    def(interp, &["pow", "Pow", "power", "expt"], |interp, args| {
        let a = arg(interp, &args, 0, "pow")?;
        let b = arg(interp, &args, 1, "pow")?;
        if let Some(m) = args.get(2) {
            // Modular exponentiation.
            let (Some(base), Some(exp), Some(m)) = (a.as_int(), b.as_int(), m.as_int()) else {
                return interp.type_err("pow(a, b, m) needs integers", args.span);
            };
            if m == 0 {
                return Err(interp.error("ValueError", "pow() modulus cannot be zero", args.span));
            }
            let (mut result, mut base, mut exp) =
                (1i128, (base as i128).rem_euclid(m as i128), exp);
            while exp > 0 {
                if exp & 1 == 1 {
                    result = result * base % m as i128;
                }
                base = base * base % m as i128;
                exp >>= 1;
            }
            return Ok(Value::Int(result as i64));
        }
        interp.binary_op(BinOp::Pow, a, b, args.span)
    });
    def(interp, &["atan2", "Atan2"], |interp, args| {
        let y = float_arg(interp, &args, 0, "atan2")?;
        let x = float_arg(interp, &args, 1, "atan2")?;
        Ok(Value::Float(y.atan2(x)))
    });
    def(interp, &["hypot", "Hypot"], |interp, args| {
        let mut acc = 0.0f64;
        for (i, _) in args.pos.iter().enumerate() {
            let v = float_arg(interp, &args, i, "hypot")?;
            acc += v * v;
        }
        Ok(Value::Float(acc.sqrt()))
    });
    def(interp, &["gcd"], |interp, args| {
        let mut g = 0;
        for (i, _) in args.pos.iter().enumerate() {
            g = gcd(g, int_arg(interp, &args, i, "gcd")?);
        }
        Ok(Value::Int(g))
    });
    def(interp, &["lcm"], |interp, args| {
        let mut l: i64 = 1;
        for (i, _) in args.pos.iter().enumerate() {
            let v = int_arg(interp, &args, i, "lcm")?;
            if v == 0 {
                return Ok(Value::Int(0));
            }
            l = (l / gcd(l, v)).saturating_mul(v.abs());
        }
        Ok(Value::Int(l))
    });
    def(interp, &["factorial", "fact"], |interp, args| {
        let n = arg(interp, &args, 0, "factorial")?;
        interp.unary_op(crate::ast::UnOp::Bang, n, args.span)
    });
    def(
        interp,
        &["comb", "choose", "binomial", "nCr"],
        |interp, args| {
            let n = int_arg(interp, &args, 0, "comb")?;
            let k = int_arg(interp, &args, 1, "comb")?;
            if k < 0 || n < 0 || k > n {
                return Ok(Value::Int(0));
            }
            let k = k.min(n - k);
            let mut acc = BigInt::from_i64(1);
            for i in 0..k {
                acc = acc.mul(&BigInt::from_i64(n - i));
                acc = acc
                    .divmod_trunc(&BigInt::from_i64(i + 1))
                    .map(|(q, _)| q)
                    .unwrap_or_default();
            }
            Ok(Value::from_big(acc))
        },
    );
    def(interp, &["perm", "nPr"], |interp, args| {
        let n = int_arg(interp, &args, 0, "perm")?;
        let k = args.get(1).and_then(Value::as_int).unwrap_or(n);
        if k < 0 || n < 0 || k > n {
            return Ok(Value::Int(0));
        }
        let mut acc = BigInt::from_i64(1);
        for i in 0..k {
            acc = acc.mul(&BigInt::from_i64(n - i));
        }
        Ok(Value::from_big(acc))
    });
    def(
        interp,
        &["isclose", "is_close", "approx_eq"],
        |interp, args| {
            let a = float_arg(interp, &args, 0, "isclose")?;
            let b = float_arg(interp, &args, 1, "isclose")?;
            let rel = args
                .named("rel_tol")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-9);
            let abs = args
                .named("abs_tol")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            Ok(Value::Bool(
                (a - b).abs() <= (rel * a.abs().max(b.abs())).max(abs),
            ))
        },
    );
    def(interp, &["mean", "average", "avg"], |interp, args| {
        let xs = if args.len() == 1 {
            interp.collect(&args.pos[0].clone(), args.span)?
        } else {
            args.pos.clone()
        };
        if let Some(Value::Tensor(t)) = args.get(0) {
            return t
                .mean(None)
                .map(Value::Tensor)
                .map_err(|e| interp.error("ValueError", e, args.span));
        }
        if xs.is_empty() {
            return interp.value_err("mean of an empty sequence", args.span);
        }
        let n = xs.len();
        let mut acc = Value::Int(0);
        for x in xs {
            acc = interp.binary_op(BinOp::Add, acc, x, args.span)?;
        }
        interp.binary_op(BinOp::Div, acc, Value::Int(n as i64), args.span)
    });
    def(interp, &["median"], |interp, args| {
        let a = arg(interp, &args, 0, "median")?;
        let xs = interp.collect(&a, args.span)?;
        let mut v: Vec<f64> = xs.iter().filter_map(Value::as_f64).collect();
        if v.is_empty() {
            return interp.value_err("median of an empty sequence", args.span);
        }
        v.sort_by(f64::total_cmp);
        let n = v.len();
        Ok(Value::Float(if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        }))
    });
    let spreads: [(&[&str], f64, bool); 4] = [
        (&["variance", "sample_variance"], 1.0, false),
        (&["pvariance", "var", "population_variance"], 0.0, false),
        (&["stdev", "standard_deviation", "sample_std"], 1.0, true),
        (&["pstdev", "std", "population_std"], 0.0, true),
    ];
    for (names, ddof, is_std) in spreads {
        def(interp, names, move |interp, args| {
            let a = arg(interp, &args, 0, "variance")?;
            let xs = interp.collect(&a, args.span)?;
            let v: Vec<f64> = xs.iter().filter_map(Value::as_f64).collect();
            if (v.len() as f64) < ddof + 1.0 {
                return interp.value_err(
                    format!("{} needs at least {} values", names[0], ddof as usize + 1),
                    args.span,
                );
            }
            let n = v.len() as f64;
            let mean = v.iter().sum::<f64>() / n;
            let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - ddof);
            Ok(Value::Float(if is_std { var.sqrt() } else { var }))
        });
    }
    def(interp, &["mode"], |interp, args| {
        let a = arg(interp, &args, 0, "mode")?;
        let xs = interp.collect(&a, args.span)?;
        let mut best: Option<(Value, usize)> = None;
        let mut counts = Dict::new();
        for x in &xs {
            let n = counts.get(x).and_then(|v| v.as_int()).unwrap_or(0) + 1;
            counts.insert(x.clone(), Value::Int(n));
            if best.as_ref().is_none_or(|(_, c)| n as usize > *c) {
                best = Some((x.clone(), n as usize));
            }
        }
        Ok(best.map_or(Value::Nil, |(v, _)| v))
    });

    let pi = Value::Float(std::f64::consts::PI);
    constant(interp, &["pi", "PI", "π", "M_PI", "Pi"], pi);
    constant(
        interp,
        &["e", "E", "M_E", "euler"],
        Value::Float(std::f64::consts::E),
    );
    constant(
        interp,
        &["tau", "TAU", "τ"],
        Value::Float(std::f64::consts::TAU),
    );
    constant(
        interp,
        &["inf", "Infinity", "INFINITY", "∞"],
        Value::Float(f64::INFINITY),
    );
    constant(interp, &["nan", "NaN", "NAN"], Value::Float(f64::NAN));
    constant(
        interp,
        &["phi", "φ", "golden_ratio"],
        Value::Float((1.0 + 5f64.sqrt()) / 2.0),
    );

    // ----- random ---------------------------------------------------------------
    def(
        interp,
        &["random", "rand", "Math.random", "random_float", "drand48"],
        |interp, args| match args.get(0) {
            None => Ok(Value::Float(interp.rng.next_f64())),
            Some(Value::Int(n)) if *n > 0 => Ok(Value::Int(interp.rng.below(*n as u64) as i64)),
            Some(Value::Range(r)) => {
                if r.is_empty() {
                    return interp.value_err("rand() of an empty range", args.span);
                }
                let i = interp.rng.index(r.len());
                Ok(Value::Int(r.get(i).unwrap_or(0)))
            }
            Some(Value::Float(f)) => Ok(Value::Float(interp.rng.next_f64() * f)),
            Some(other) => {
                let items = interp.collect(&other.clone(), args.span)?;
                if items.is_empty() {
                    return Ok(Value::Nil);
                }
                let i = interp.rng.index(items.len());
                Ok(items[i].clone())
            }
        },
    );
    def(
        interp,
        &["randint", "random_int", "rand_int", "randrange", "nextInt"],
        |interp, args| {
            let a = int_arg(interp, &args, 0, "randint")?;
            let (lo, hi) = match args.get(1).and_then(Value::as_int) {
                Some(b) => (a, b),
                None => (0, a - 1),
            };
            if lo > hi {
                return interp.value_err(format!("empty range for randint({lo}, {hi})"), args.span);
            }
            Ok(Value::Int(interp.rng.range_i64(lo, hi.saturating_add(1))))
        },
    );
    def(interp, &["uniform", "random_uniform"], |interp, args| {
        let a = float_arg(interp, &args, 0, "uniform")?;
        let b = float_arg(interp, &args, 1, "uniform")?;
        Ok(Value::Float(interp.rng.range_f64(a, b)))
    });
    def(
        interp,
        &["gauss", "normalvariate", "random_normal", "randn_scalar"],
        |interp, args| {
            let mu = args.get(0).and_then(Value::as_f64).unwrap_or(0.0);
            let sigma = args.get(1).and_then(Value::as_f64).unwrap_or(1.0);
            Ok(Value::Float(interp.rng.normal(mu, sigma)))
        },
    );
    def(interp, &["choice", "pick", "sample_one"], |interp, args| {
        let xs = arg(interp, &args, 0, "choice")?;
        let items = interp.collect(&xs, args.span)?;
        if items.is_empty() {
            return Err(interp.error(
                "IndexError",
                "cannot choose from an empty sequence",
                args.span,
            ));
        }
        let i = interp.rng.index(items.len());
        Ok(items[i].clone())
    });
    def(interp, &["shuffle", "shuffled"], |interp, args| {
        let xs = arg(interp, &args, 0, "shuffle")?;
        if let Value::List(l) = &xs {
            let mut items = l.borrow().clone();
            interp.rng.shuffle(&mut items);
            *l.borrow_mut() = items;
            return Ok(Value::Nil);
        }
        let mut items = interp.collect(&xs, args.span)?;
        interp.rng.shuffle(&mut items);
        Ok(Value::list(items))
    });
    def(interp, &["sample", "random_sample"], |interp, args| {
        let xs = arg(interp, &args, 0, "sample")?;
        let k = int_arg(interp, &args, 1, "sample")?.max(0) as usize;
        let mut items = interp.collect(&xs, args.span)?;
        if k > items.len() {
            return interp.value_err("sample larger than the population", args.span);
        }
        interp.rng.shuffle(&mut items);
        items.truncate(k);
        Ok(Value::list(items))
    });
    def(
        interp,
        &["seed", "srand", "random_seed", "set_seed", "manual_seed"],
        |interp, args| {
            let s = args.get(0).and_then(Value::as_int).unwrap_or(0);
            interp.rng = chimera_core::Rng::new(s as u64);
            Ok(Value::Nil)
        },
    );
}

#[allow(dead_code)]
fn unused(_: Args) {}
