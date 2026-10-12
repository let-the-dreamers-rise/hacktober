//! Operator semantics: arithmetic (with transparent big-integer promotion),
//! comparison, equality, membership, operator overloading, tensor
//! broadcasting and dual-number differentiation.

use std::cmp::Ordering;
use std::rc::Rc;

use chimera_core::Span;

use super::{Args, Interp, sym};
use crate::ast::{BinOp, CmpOp, UnOp};
use crate::bigint::BigInt;
use crate::ml::dual::{self, Dual};
use crate::ml::tensor::Tensor;
use crate::value::*;

/// A number in one of its three representations.
enum Num {
    I(i64),
    B(BigInt),
    F(f64),
}

fn num(v: &Value) -> Option<Num> {
    match v {
        Value::Int(i) => Some(Num::I(*i)),
        Value::Bool(b) => Some(Num::I(i64::from(*b))),
        Value::Float(f) => Some(Num::F(*f)),
        Value::Big(b) => Some(Num::B((**b).clone())),
        _ => None,
    }
}

fn big(n: &Num) -> BigInt {
    match n {
        Num::I(i) => BigInt::from_i64(*i),
        Num::B(b) => b.clone(),
        Num::F(f) => BigInt::from_f64(*f).unwrap_or_default(),
    }
}

fn float(n: &Num) -> f64 {
    match n {
        Num::I(i) => *i as f64,
        Num::B(b) => b.to_f64(),
        Num::F(f) => *f,
    }
}

/// Method names that overload each binary operator.
fn overload_names(op: BinOp) -> &'static [&'static str] {
    match op {
        BinOp::Add => &["__add__", "+", "add", "plus"],
        BinOp::Sub => &["__sub__", "-", "sub", "minus"],
        BinOp::Mul => &["__mul__", "*", "mul", "times"],
        BinOp::Div => &["__truediv__", "__div__", "/", "div"],
        BinOp::FloorDiv => &["__floordiv__"],
        BinOp::Mod => &["__mod__", "%", "rem"],
        BinOp::Pow => &["__pow__", "**", "pow"],
        BinOp::MatMul => &["__matmul__", "@"],
        BinOp::BitAnd => &["__and__", "&"],
        BinOp::BitOr => &["__or__", "|"],
        BinOp::BitXor => &["__xor__", "^"],
        BinOp::Shl => &["__lshift__", "<<"],
        BinOp::Shr => &["__rshift__", ">>"],
        BinOp::Concat => &["__concat__", "++"],
        BinOp::Pipe => &[],
    }
}

fn reflected_name(op: BinOp) -> Option<&'static str> {
    Some(match op {
        BinOp::Add => "__radd__",
        BinOp::Sub => "__rsub__",
        BinOp::Mul => "__rmul__",
        BinOp::Div => "__rtruediv__",
        BinOp::Pow => "__rpow__",
        _ => return None,
    })
}

fn op_symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::FloorDiv => "div",
        BinOp::Mod => "%",
        BinOp::Pow => "**",
        BinOp::MatMul => "@",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "xor",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        BinOp::Concat => "++",
        BinOp::Pipe => "|>",
    }
}

impl Interp {
    // ----- unary ------------------------------------------------------------

    /// Applies a prefix (or postfix `!`) operator.
    pub fn unary_op(&mut self, op: UnOp, v: Value, span: Span) -> R {
        match op {
            UnOp::Not => Ok(Value::Bool(!v.truthy())),
            UnOp::TypeOf => Ok(Value::str(&js_typeof(&v))),
            UnOp::Pos => match &v {
                Value::Str(s) => Ok(s
                    .trim()
                    .parse::<f64>()
                    .map(Value::Float)
                    .unwrap_or(Value::Float(f64::NAN))),
                _ => Ok(v),
            },
            UnOp::Neg => match v {
                Value::Int(i) => Ok(i
                    .checked_neg()
                    .map(Value::Int)
                    .unwrap_or_else(|| Value::from_big(BigInt::from_i64(i).neg()))),
                Value::Float(f) => Ok(Value::Float(-f)),
                Value::Big(b) => Ok(Value::from_big(b.neg())),
                Value::Bool(b) => Ok(Value::Int(-i64::from(b))),
                Value::Tensor(t) => Ok(Value::Tensor(t.neg())),
                Value::Dual(d) => {
                    let re = self.unary_op(UnOp::Neg, d.re.clone(), span)?;
                    let du = self.unary_op(UnOp::Neg, d.du.clone(), span)?;
                    Ok(Dual::value(d.tag, re, du))
                }
                Value::Instance(_) => self
                    .call_overload(&v, &["__neg__", "neg", "-@"], Vec::new(), span)?
                    .map_or_else(|| self.type_err("cannot negate this object", span), Ok),
                other => {
                    let t = other.type_name();
                    self.type_err(format!("cannot negate a {t}"), span)
                }
            },
            UnOp::BitNot => match v {
                Value::Int(i) => Ok(Value::Int(!i)),
                Value::Bool(b) => Ok(Value::Int(!i64::from(b))),
                other => {
                    let t = other.type_name();
                    self.type_err(format!("`~` needs an integer, got {t}"), span)
                }
            },
            UnOp::Sqrt => {
                let f = self
                    .builtins
                    .get(&sym("sqrt"))
                    .cloned()
                    .unwrap_or(Value::Nil);
                self.call_value(f, Args::new(vec![v], span))
            }
            UnOp::Try => match &v {
                Value::Nil => Err(super::Flow::Return(Value::Nil)),
                Value::Variant(var) if var.name().is("Err") => Err(super::Flow::Return(v.clone())),
                Value::Variant(var) if var.name().is("Ok") || var.name().is("Some") => {
                    Ok(var.fields.first().cloned().unwrap_or(Value::Nil))
                }
                _ => Ok(v),
            },
            UnOp::Bang => match &v {
                Value::Int(n) => {
                    if *n < 0 {
                        return self.value_err("factorial of a negative number", span);
                    }
                    if *n > 100_000 {
                        return self.value_err("factorial argument is too large", span);
                    }
                    let mut acc = BigInt::from_i64(1);
                    for i in 2..=*n {
                        acc = acc.mul(&BigInt::from_i64(i));
                    }
                    Ok(Value::from_big(acc))
                }
                Value::Nil => Err(self.error(
                    "ValueError",
                    "unexpectedly found nil while unwrapping a value",
                    span,
                )),
                Value::Variant(var) if var.name().is("Err") => {
                    let msg = var
                        .fields
                        .first()
                        .map(super::format::repr_plain)
                        .unwrap_or_default();
                    Err(self.error(
                        "ValueError",
                        format!("called unwrap on an Err value: {msg}"),
                        span,
                    ))
                }
                Value::Variant(var) if var.name().is("Ok") || var.name().is("Some") => {
                    Ok(var.fields.first().cloned().unwrap_or(Value::Nil))
                }
                _ => Ok(v),
            },
        }
    }

    // ----- binary --------------------------------------------------------

    /// Applies a binary operator.
    pub fn binary_op(&mut self, op: BinOp, a: Value, b: Value, span: Span) -> R {
        if matches!(a, Value::Dual(_)) || matches!(b, Value::Dual(_)) {
            return self.dual_binary(op, a, b, span);
        }
        if matches!(a, Value::Tensor(_)) || matches!(b, Value::Tensor(_)) || op == BinOp::MatMul {
            return self.tensor_binary(op, &a, &b, span);
        }
        if let (Value::Bool(p), Value::Bool(q)) = (&a, &b)
            && matches!(op, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor)
        {
            // `true & false`, `a ^ b` on booleans stay boolean.
            return Ok(Value::Bool(match op {
                BinOp::BitAnd => *p & *q,
                BinOp::BitOr => *p | *q,
                _ => *p ^ *q,
            }));
        }
        if op != BinOp::Concat
            && let (Some(x), Some(y)) = (num(&a), num(&b))
        {
            return self.numeric(op, x, y, span);
        }
        match (op, &a, &b) {
            // Lua `..` on numbers: `1 .. 2` is "12".
            (BinOp::Concat, x, y)
                if !matches!(
                    x,
                    Value::List(_) | Value::Tuple(_) | Value::Str(_) | Value::Instance(_)
                ) && !matches!(
                    y,
                    Value::List(_) | Value::Tuple(_) | Value::Str(_) | Value::Instance(_)
                ) =>
            {
                let (x, y) = (self.to_str(x)?, self.to_str(y)?);
                Ok(Value::str(&format!("{x}{y}")))
            }
            (BinOp::Add | BinOp::Concat, Value::Str(x), Value::Str(y)) => {
                let mut s = String::with_capacity(x.len() + y.len());
                s.push_str(x);
                s.push_str(y);
                Ok(Value::str(&s))
            }
            (BinOp::Add | BinOp::Concat, Value::Str(x), other)
                if !matches!(other, Value::Instance(_)) =>
            {
                let y = self.to_str(other)?;
                Ok(Value::str(&format!("{x}{y}")))
            }
            (BinOp::Add | BinOp::Concat, other, Value::Str(y))
                if !matches!(other, Value::Instance(_)) =>
            {
                let x = self.to_str(other)?;
                Ok(Value::str(&format!("{x}{y}")))
            }
            (BinOp::Pow, Value::Str(x), Value::Str(y)) => Ok(Value::str(&format!("{x}{y}"))),
            (BinOp::Add | BinOp::Concat, Value::List(x), Value::List(y)) => {
                let mut v = x.borrow().clone();
                v.extend(y.borrow().iter().cloned());
                Ok(Value::list(v))
            }
            (BinOp::Add | BinOp::Concat, Value::Tuple(x), Value::Tuple(y)) => {
                let mut v = x.to_vec();
                v.extend(y.iter().cloned());
                Ok(Value::tuple(v))
            }
            (BinOp::Add | BinOp::BitOr, Value::Map(x), Value::Map(y)) => {
                let mut d = x.borrow().clone();
                for (k, v) in y.borrow().iter() {
                    d.insert(k.clone(), v.clone());
                }
                Ok(Value::Map(Rc::new(std::cell::RefCell::new(d))))
            }
            (BinOp::Sub, Value::List(x), Value::List(y)) => {
                let y = y.borrow().clone();
                let mut out = Vec::new();
                for item in x.borrow().clone() {
                    let mut found = false;
                    for o in &y {
                        if self.values_equal(&item, o)? {
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        out.push(item);
                    }
                }
                Ok(Value::list(out))
            }
            (BinOp::Mul, Value::Str(s), n) | (BinOp::Mul, n, Value::Str(s))
                if n.as_int().is_some() =>
            {
                let n = n.as_int().unwrap_or(0).max(0) as usize;
                if s.len().saturating_mul(n) > 100_000_000 {
                    return self.value_err("string repetition is too large", span);
                }
                Ok(Value::str(&s.repeat(n)))
            }
            (BinOp::Mul, Value::List(l), n) | (BinOp::Mul, n, Value::List(l))
                if n.as_int().is_some() =>
            {
                let n = n.as_int().unwrap_or(0).max(0) as usize;
                let items = l.borrow();
                if items.len().saturating_mul(n) > 50_000_000 {
                    return self.value_err("list repetition is too large", span);
                }
                let mut out = Vec::with_capacity(items.len() * n);
                for _ in 0..n {
                    out.extend(items.iter().cloned());
                }
                Ok(Value::list(out))
            }
            (BinOp::Mul, Value::Tuple(t), n) if n.as_int().is_some() => {
                let n = n.as_int().unwrap_or(0).max(0) as usize;
                let mut out = Vec::with_capacity(t.len() * n);
                for _ in 0..n {
                    out.extend(t.iter().cloned());
                }
                Ok(Value::tuple(out))
            }
            (BinOp::Mod, Value::Str(fmt), args) => {
                let args: Vec<Value> = match args {
                    Value::Tuple(t) => t.to_vec(),
                    other => vec![other.clone()],
                };
                let s = self.printf(fmt, &args, span)?;
                Ok(Value::str(&s))
            }
            (
                BinOp::BitOr | BinOp::BitAnd | BinOp::Sub | BinOp::BitXor,
                Value::Set(x),
                Value::Set(y),
            ) => {
                let (x, y) = (x.borrow(), y.borrow());
                let mut d = Dict::new();
                match op {
                    BinOp::BitOr => {
                        for k in x.keys().into_iter().chain(y.keys()) {
                            d.insert(k, Value::Nil);
                        }
                    }
                    BinOp::BitAnd => {
                        for k in x.keys() {
                            if y.contains(&k) {
                                d.insert(k, Value::Nil);
                            }
                        }
                    }
                    BinOp::Sub => {
                        for k in x.keys() {
                            if !y.contains(&k) {
                                d.insert(k, Value::Nil);
                            }
                        }
                    }
                    _ => {
                        for k in x.keys() {
                            if !y.contains(&k) {
                                d.insert(k, Value::Nil);
                            }
                        }
                        for k in y.keys() {
                            if !x.contains(&k) {
                                d.insert(k, Value::Nil);
                            }
                        }
                    }
                }
                Ok(Value::Set(Rc::new(std::cell::RefCell::new(d))))
            }
            (BinOp::Shl, Value::List(l), x) => {
                // Ruby: `xs << item` appends.
                l.borrow_mut().push(x.clone());
                Ok(a.clone())
            }
            (BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor, Value::Bool(x), Value::Bool(y)) => {
                Ok(Value::Bool(match op {
                    BinOp::BitAnd => *x & *y,
                    BinOp::BitOr => *x | *y,
                    _ => *x ^ *y,
                }))
            }
            (_, Value::Object(o), _) => {
                let this = a.clone();
                match o.call_method(self, op_symbol(op), Args::new(vec![b.clone()], span), &this) {
                    Some(r) => r,
                    None => self.op_error(op, &a, &b, span),
                }
            }
            (_, Value::Instance(_), _) => {
                match self.call_overload(&a, overload_names(op), vec![b.clone()], span)? {
                    Some(v) => Ok(v),
                    None => self.op_error(op, &a, &b, span),
                }
            }
            (_, _, Value::Instance(_)) => {
                let reflected = reflected_name(op).map(|n| vec![n]).unwrap_or_default();
                match self.call_overload(&b, &reflected, vec![a.clone()], span)? {
                    Some(v) => Ok(v),
                    None => self.op_error(op, &a, &b, span),
                }
            }
            _ => self.op_error(op, &a, &b, span),
        }
    }

    fn op_error(&mut self, op: BinOp, a: &Value, b: &Value, span: Span) -> R {
        let (ta, tb) = (a.type_name(), b.type_name());
        self.type_err(
            format!(
                "unsupported operand types for `{}`: {ta} and {tb}",
                op_symbol(op)
            ),
            span,
        )
    }

    /// Calls the first operator method an instance defines, if any.
    pub(crate) fn call_overload(
        &mut self,
        v: &Value,
        names: &[&str],
        args: Vec<Value>,
        span: Span,
    ) -> R<Option<Value>> {
        let Value::Instance(inst) = v else {
            return Ok(None);
        };
        for name in names {
            if let Some(m) = inst.class.find_method(sym(name)) {
                let bound = Value::Bound(Rc::new(Bound {
                    receiver: v.clone(),
                    func: m,
                }));
                return self.call_value(bound, Args::new(args, span)).map(Some);
            }
        }
        Ok(None)
    }

    fn numeric(&mut self, op: BinOp, x: Num, y: Num, span: Span) -> R {
        use Num::*;
        let any_float = matches!(x, F(_)) || matches!(y, F(_));
        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Concat => {
                if any_float {
                    let (a, b) = (float(&x), float(&y));
                    return Ok(Value::Float(match op {
                        BinOp::Sub => a - b,
                        BinOp::Mul => a * b,
                        _ => a + b,
                    }));
                }
                if let (I(a), I(b)) = (&x, &y) {
                    let r = match op {
                        BinOp::Sub => a.checked_sub(*b),
                        BinOp::Mul => a.checked_mul(*b),
                        _ => a.checked_add(*b),
                    };
                    if let Some(r) = r {
                        return Ok(Value::Int(r));
                    }
                }
                let (a, b) = (big(&x), big(&y));
                Ok(Value::from_big(match op {
                    BinOp::Sub => a.sub(&b),
                    BinOp::Mul => a.mul(&b),
                    _ => a.add(&b),
                }))
            }
            BinOp::Div => {
                let d = float(&y);
                if d == 0.0 {
                    return Err(self.error("ZeroDivisionError", "division by zero", span));
                }
                // Exact big-integer division when it divides evenly.
                if let (B(_), _) | (_, B(_)) = (&x, &y)
                    && !any_float
                    && let Some((q, r)) = big(&x).divmod_trunc(&big(&y))
                    && r.is_zero()
                {
                    return Ok(Value::from_big(q));
                }
                Ok(Value::Float(float(&x) / d))
            }
            BinOp::FloorDiv | BinOp::Mod => {
                if any_float {
                    let (a, b) = (float(&x), float(&y));
                    if b == 0.0 {
                        return Err(self.error("ZeroDivisionError", "division by zero", span));
                    }
                    let q = (a / b).floor();
                    return Ok(Value::Float(if op == BinOp::FloorDiv {
                        q
                    } else {
                        a - b * q
                    }));
                }
                if let (I(a), I(b)) = (&x, &y) {
                    if *b == 0 {
                        return Err(self.error(
                            "ZeroDivisionError",
                            "integer division or modulo by zero",
                            span,
                        ));
                    }
                    if let (Some(q), Some(r)) = (a.checked_div_euclid(*b), a.checked_rem_euclid(*b))
                    {
                        // Floor semantics (sign of the divisor), like Python.
                        let (q, r) = if r != 0 && *b < 0 {
                            (q + 1, r + b)
                        } else {
                            (q, r)
                        };
                        return Ok(Value::Int(if op == BinOp::FloorDiv { q } else { r }));
                    }
                }
                let (a, b) = (big(&x), big(&y));
                match a.divmod_floor(&b) {
                    Some((q, r)) => Ok(Value::from_big(if op == BinOp::FloorDiv { q } else { r })),
                    None => Err(self.error(
                        "ZeroDivisionError",
                        "integer division or modulo by zero",
                        span,
                    )),
                }
            }
            BinOp::Pow => {
                if !any_float {
                    let exp = big(&y);
                    if !exp.is_negative() {
                        let Some(e) = exp.to_i64() else {
                            return self.value_err("exponent is too large", span);
                        };
                        let base = big(&x);
                        if base.decimal_len() as f64 * e as f64 > 2_000_000.0 {
                            return self
                                .value_err("result of `**` would be astronomically large", span);
                        }
                        if let I(b) = x
                            && let Ok(e32) = u32::try_from(e)
                            && let Some(r) = b.checked_pow(e32)
                        {
                            return Ok(Value::Int(r));
                        }
                        return Ok(Value::from_big(base.pow(e as u64)));
                    }
                }
                Ok(Value::Float(float(&x).powf(float(&y))))
            }
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                let (I(a), I(b)) = (&x, &y) else {
                    if let (I(a), I(b)) = (
                        match &x {
                            F(f) if f.fract() == 0.0 => I(*f as i64),
                            B(_) => {
                                return self.type_err(
                                    "bitwise operators need machine-sized integers",
                                    span,
                                );
                            }
                            F(_) => return self.type_err("bitwise operators need integers", span),
                            I(i) => I(*i),
                        },
                        match &y {
                            F(f) if f.fract() == 0.0 => I(*f as i64),
                            B(_) | F(_) => {
                                return self.type_err("bitwise operators need integers", span);
                            }
                            I(i) => I(*i),
                        },
                    ) {
                        return self.numeric(op, I(a), I(b), span);
                    }
                    return self.type_err("bitwise operators need integers", span);
                };
                Ok(Value::Int(match op {
                    BinOp::BitAnd => a & b,
                    BinOp::BitOr => a | b,
                    BinOp::BitXor => a ^ b,
                    BinOp::Shl => {
                        if *b < 0 || *b >= 63 {
                            return Ok(Value::from_big(
                                BigInt::from_i64(*a)
                                    .mul(&BigInt::from_i64(2).pow((*b).max(0) as u64)),
                            ));
                        }
                        match a.checked_mul(1i64 << b) {
                            Some(v) => v,
                            None => {
                                return Ok(Value::from_big(
                                    BigInt::from_i64(*a).mul(&BigInt::from_i64(2).pow(*b as u64)),
                                ));
                            }
                        }
                    }
                    _ => a >> b.clamp(&0, &63),
                }))
            }
            BinOp::MatMul | BinOp::Pipe => self.type_err("unsupported numeric operator", span),
        }
    }

    // ----- tensors ---------------------------------------------------------

    /// Converts a value to a tensor (numbers, nested lists, tensors).
    pub fn to_tensor(&mut self, v: &Value, span: Span) -> R<Tensor> {
        if let Value::Tensor(t) = v {
            return Ok(t.clone());
        }
        if let Some(x) = v.as_f64()
            && !matches!(v, Value::Str(_))
        {
            return Ok(Tensor::scalar(x));
        }
        let mut data = Vec::new();
        let mut shape = Vec::new();
        self.flatten_nested(v, 0, &mut data, &mut shape, span)?;
        Tensor::new(data, shape).map_err(|e| self.error("ValueError", e, span))
    }

    fn flatten_nested(
        &mut self,
        v: &Value,
        depth: usize,
        data: &mut Vec<f64>,
        shape: &mut Vec<usize>,
        span: Span,
    ) -> R<()> {
        let items: Vec<Value> = match v {
            Value::List(l) => l.borrow().clone(),
            Value::Tuple(t) => t.to_vec(),
            Value::Range(_) => {
                let mut it = self.iterate(v, span)?;
                let mut out = Vec::new();
                while let Some(x) = it.next(self)? {
                    out.push(x);
                }
                out
            }
            Value::Tensor(t) => {
                if depth == shape.len() {
                    shape.extend_from_slice(t.shape());
                }
                data.extend(t.data());
                return Ok(());
            }
            other => {
                let Some(x) = other.as_f64() else {
                    let t = other.type_name();
                    return self.type_err(format!("tensors hold numbers, not {t}"), span);
                };
                if depth < shape.len() {
                    return self.value_err("ragged nested list: rows have different lengths", span);
                }
                data.push(x);
                return Ok(());
            }
        };
        if depth == shape.len() {
            shape.push(items.len());
        } else if shape[depth] != items.len() {
            return self.value_err("ragged nested list: rows have different lengths", span);
        }
        for item in &items {
            self.flatten_nested(item, depth + 1, data, shape, span)?;
        }
        Ok(())
    }

    fn tensor_binary(&mut self, op: BinOp, a: &Value, b: &Value, span: Span) -> R {
        let x = self.to_tensor(a, span)?;
        let y = self.to_tensor(b, span)?;
        let r = match op {
            BinOp::Add => x.add(&y),
            BinOp::Sub => x.sub(&y),
            BinOp::Mul => x.mul(&y),
            BinOp::Div => x.div(&y),
            BinOp::Pow => x.pow(&y),
            BinOp::MatMul => x.matmul(&y),
            BinOp::Mod => x.binary_map(&y, |p, q| p - q * (p / q).floor()),
            BinOp::FloorDiv => x.binary_map(&y, |p, q| (p / q).floor()),
            _ => return self.op_error(op, a, b, span),
        };
        r.map(Value::Tensor)
            .map_err(|e| self.error("ValueError", e, span))
    }

    // ----- duals -----------------------------------------------------------

    fn dual_binary(&mut self, op: BinOp, a: Value, b: Value, span: Span) -> R {
        let Some(tag) = dual::top_tag(&a, &b) else {
            return self.binary_op(op, a, b, span);
        };
        let (ar, ad) = dual::split(&a, tag);
        let (br, bd) = dual::split(&b, tag);
        let (re, du) = match op {
            BinOp::Add | BinOp::Sub => {
                let re = self.binary_op(op, ar, br, span)?;
                let du = self.binary_op(op, ad, bd, span)?;
                (re, du)
            }
            BinOp::Mul => {
                let re = self.binary_op(BinOp::Mul, ar.clone(), br.clone(), span)?;
                let t1 = self.binary_op(BinOp::Mul, ar, bd, span)?;
                let t2 = self.binary_op(BinOp::Mul, ad, br, span)?;
                (re, self.binary_op(BinOp::Add, t1, t2, span)?)
            }
            BinOp::Div => {
                let re = self.binary_op(BinOp::Div, ar.clone(), br.clone(), span)?;
                let num1 = self.binary_op(BinOp::Mul, ad, br.clone(), span)?;
                let num2 = self.binary_op(BinOp::Mul, ar, bd, span)?;
                let numer = self.binary_op(BinOp::Sub, num1, num2, span)?;
                let denom = self.binary_op(BinOp::Mul, br.clone(), br, span)?;
                (re, self.binary_op(BinOp::Div, numer, denom, span)?)
            }
            BinOp::Pow => {
                let re = self.binary_op(BinOp::Pow, ar.clone(), br.clone(), span)?;
                // d(a^b) = b·a^(b-1)·da + a^b·ln(a)·db
                let bm1 = self.binary_op(BinOp::Sub, br.clone(), Value::Int(1), span)?;
                let pow_bm1 = self.binary_op(BinOp::Pow, ar.clone(), bm1, span)?;
                let t1 = self.binary_op(BinOp::Mul, br.clone(), pow_bm1, span)?;
                let t1 = self.binary_op(BinOp::Mul, t1, ad, span)?;
                let db_zero = matches!(bd, Value::Int(0));
                let du = if db_zero {
                    t1
                } else {
                    let ln = self.math_fn("log", ar, span)?;
                    let t2 = self.binary_op(BinOp::Mul, re.clone(), ln, span)?;
                    let t2 = self.binary_op(BinOp::Mul, t2, bd, span)?;
                    self.binary_op(BinOp::Add, t1, t2, span)?
                };
                (re, du)
            }
            _ => {
                let t = format!("`{}`", op_symbol(op));
                return self.type_err(format!("{t} is not differentiable"), span);
            }
        };
        Ok(Dual::value(tag, re, du))
    }

    /// Calls a math builtin by name (works on numbers, duals and tensors).
    pub(crate) fn math_fn(&mut self, name: &str, x: Value, span: Span) -> R {
        let f = self.builtins.get(&sym(name)).cloned().unwrap_or(Value::Nil);
        self.call_value(f, Args::new(vec![x], span))
    }

    // ----- comparison ------------------------------------------------------

    /// Applies a comparison operator.
    pub fn compare_op(&mut self, op: CmpOp, a: &Value, b: &Value, span: Span) -> R {
        Ok(Value::Bool(match op {
            CmpOp::Eq => self.values_equal(a, b)?,
            CmpOp::Ne => !self.values_equal(a, b)?,
            CmpOp::StrictEq => strict_type(a) == strict_type(b) && self.values_equal(a, b)?,
            CmpOp::StrictNe => !(strict_type(a) == strict_type(b) && self.values_equal(a, b)?),
            CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => {
                if let (Value::Tensor(_), _) | (_, Value::Tensor(_)) = (a, b) {
                    let x = self.to_tensor(a, span)?;
                    let y = self.to_tensor(b, span)?;
                    let r = x.compare(&y, move |p, q| match op {
                        CmpOp::Lt => p < q,
                        CmpOp::Le => p <= q,
                        CmpOp::Gt => p > q,
                        _ => p >= q,
                    });
                    return r
                        .map(Value::Tensor)
                        .map_err(|e| self.error("ValueError", e, span));
                }
                if let (Some(x), Some(y)) = (a.as_f64(), b.as_f64())
                    && (x.is_nan() || y.is_nan())
                {
                    return Ok(Value::Bool(false));
                }
                let ord = self.compare_values(a, b, span)?;
                match op {
                    CmpOp::Lt => ord == Ordering::Less,
                    CmpOp::Le => ord != Ordering::Greater,
                    CmpOp::Gt => ord == Ordering::Greater,
                    _ => ord != Ordering::Less,
                }
            }
            CmpOp::In => self.contains(b, a, span)?,
            CmpOp::NotIn => !self.contains(b, a, span)?,
            CmpOp::Is | CmpOp::IsNot => {
                let r = match b {
                    Value::Class(_) | Value::Type(_) | Value::Enum(_) => self.isinstance(a, b),
                    _ => identical(a, b),
                };
                if op == CmpOp::Is { r } else { !r }
            }
            CmpOp::Approx => match (a.as_f64(), b.as_f64()) {
                (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0),
                _ => self.values_equal(a, b)?,
            },
            CmpOp::Spaceship => {
                let ord = self.compare_values(a, b, span)?;
                return Ok(Value::Int(match ord {
                    Ordering::Less => -1,
                    Ordering::Equal => 0,
                    Ordering::Greater => 1,
                }));
            }
        }))
    }

    /// `isinstance(v, class_or_type)`.
    pub fn isinstance(&self, v: &Value, ty: &Value) -> bool {
        match ty {
            Value::Class(c) => match v {
                Value::Instance(i) => i.class.is_subclass_of(c),
                _ => false,
            },
            Value::Type(t) => self.type_tag_matches(v, *t),
            Value::Enum(e) => matches!(v, Value::Variant(var) if Rc::ptr_eq(&var.ty, e)),
            _ => false,
        }
    }

    /// Structural equality (calls `__eq__`/`equals` on instances).
    pub fn values_equal(&mut self, a: &Value, b: &Value) -> R<bool> {
        Ok(match (a, b) {
            (Value::Nil, Value::Nil) => true,
            (Value::Str(x), Value::Str(y)) => x == y,
            (Value::List(x), Value::List(y)) => {
                if Rc::ptr_eq(x, y) {
                    return Ok(true);
                }
                let (x, y) = (x.borrow().clone(), y.borrow().clone());
                self.seq_equal(&x, &y)?
            }
            (Value::Tuple(x), Value::Tuple(y)) => self.seq_equal(x, y)?,
            (Value::Map(x), Value::Map(y)) | (Value::Set(x), Value::Set(y)) => {
                let (x, y) = (x.borrow().clone(), y.borrow().clone());
                if x.len() != y.len() {
                    return Ok(false);
                }
                for (k, v) in x.iter() {
                    match y.get(k) {
                        Some(w) => {
                            if !self.values_equal(v, &w)? {
                                return Ok(false);
                            }
                        }
                        None => return Ok(false),
                    }
                }
                true
            }
            (Value::Range(x), Value::Range(y)) => x == y,
            (Value::Variant(x), Value::Variant(y)) => {
                Rc::ptr_eq(&x.ty, &y.ty)
                    && x.index == y.index
                    && self.seq_equal(&x.fields, &y.fields)?
            }
            (Value::Variant(v), Value::Str(s)) | (Value::Str(s), Value::Variant(v)) => {
                v.name().is(s)
            }
            (Value::Instance(x), Value::Instance(y)) => {
                if Rc::ptr_eq(x, y) {
                    return Ok(true);
                }
                if let Some(r) = self.call_overload(
                    a,
                    &["__eq__", "equals", "==", "eq"],
                    vec![b.clone()],
                    Span::default(),
                )? {
                    return Ok(r.truthy());
                }
                // Data classes / structs compare by value.
                let auto = x.class.def.as_ref().is_some_and(|d| d.auto_init);
                if auto && Rc::ptr_eq(&x.class, &y.class) {
                    let (fx, fy) = (x.fields.borrow().clone(), y.fields.borrow().clone());
                    if fx.len() != fy.len() {
                        return Ok(false);
                    }
                    for ((nx, vx), (ny, vy)) in fx.iter().zip(fy.iter()) {
                        if nx != ny || !self.values_equal(vx, vy)? {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }
                false
            }
            (Value::Tensor(x), Value::Tensor(y)) => x.shape() == y.shape() && x.data() == y.data(),
            (Value::Dual(d), other) | (other, Value::Dual(d)) => self.values_equal(&d.re, other)?,
            (x, y) => match (num(x), num(y)) {
                (Some(p), Some(q)) => match (p, q) {
                    (Num::I(a), Num::I(b)) => a == b,
                    (Num::F(a), b) | (b, Num::F(a)) => a == float(&b),
                    (p, q) => big(&p) == big(&q),
                },
                _ => match (x, y) {
                    (Value::Bool(p), Value::Bool(q)) => p == q,
                    _ => identical(x, y),
                },
            },
        })
    }

    fn seq_equal(&mut self, x: &[Value], y: &[Value]) -> R<bool> {
        if x.len() != y.len() {
            return Ok(false);
        }
        for (a, b) in x.iter().zip(y) {
            if !self.values_equal(a, b)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Total ordering used by `<`, `sort` and `max`.
    pub fn compare_values(&mut self, a: &Value, b: &Value, span: Span) -> R<Ordering> {
        if let (Some(x), Some(y)) = (num(a), num(b)) {
            return Ok(match (x, y) {
                (Num::I(p), Num::I(q)) => p.cmp(&q),
                (Num::F(p), q) => p.partial_cmp(&float(&q)).unwrap_or(Ordering::Equal),
                (p, Num::F(q)) => float(&p).partial_cmp(&q).unwrap_or(Ordering::Equal),
                (p, q) => big(&p).cmp(&big(&q)),
            });
        }
        match (a, b) {
            (Value::Str(x), Value::Str(y)) => Ok(x.cmp(y)),
            (Value::List(_) | Value::Tuple(_), Value::List(_) | Value::Tuple(_)) => {
                let x = seq_items(a);
                let y = seq_items(b);
                for (p, q) in x.iter().zip(y.iter()) {
                    let o = self.compare_values(p, q, span)?;
                    if o != Ordering::Equal {
                        return Ok(o);
                    }
                }
                Ok(x.len().cmp(&y.len()))
            }
            (Value::Variant(x), Value::Variant(y)) if Rc::ptr_eq(&x.ty, &y.ty) => {
                if x.index != y.index {
                    return Ok(x.index.cmp(&y.index));
                }
                for (p, q) in x.fields.iter().zip(y.fields.iter()) {
                    let o = self.compare_values(p, q, span)?;
                    if o != Ordering::Equal {
                        return Ok(o);
                    }
                }
                Ok(Ordering::Equal)
            }
            (Value::Dual(d), other) => self.compare_values(&d.re.clone(), other, span),
            (other, Value::Dual(d)) => self.compare_values(other, &d.re.clone(), span),
            (Value::Instance(_), _) => {
                if let Some(r) = self.call_overload(
                    a,
                    &["__cmp__", "compareTo", "<=>", "compare", "cmp"],
                    vec![b.clone()],
                    span,
                )? {
                    let n = r.as_int().unwrap_or(0);
                    return Ok(n.cmp(&0));
                }
                if let Some(r) = self.call_overload(a, &["__lt__", "<"], vec![b.clone()], span)? {
                    if r.truthy() {
                        return Ok(Ordering::Less);
                    }
                    let eq = self.values_equal(a, b)?;
                    return Ok(if eq {
                        Ordering::Equal
                    } else {
                        Ordering::Greater
                    });
                }
                let (ta, tb) = (a.type_name(), b.type_name());
                self.type_err(
                    format!("cannot compare {ta} with {tb} (define `__lt__` or `compareTo`)"),
                    span,
                )
            }
            (Value::Nil, Value::Nil) => Ok(Ordering::Equal),
            _ => {
                let (ta, tb) = (a.type_name(), b.type_name());
                self.type_err(format!("cannot compare {ta} with {tb}"), span)
            }
        }
    }

    /// Membership test: `item in container`.
    pub fn contains(&mut self, container: &Value, item: &Value, span: Span) -> R<bool> {
        match container {
            Value::List(l) => {
                let items = l.borrow().clone();
                for x in &items {
                    if self.values_equal(x, item)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Value::Tuple(t) => {
                for x in t.iter() {
                    if self.values_equal(x, item)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Value::Str(s) => match item {
                Value::Str(sub) => Ok(s.contains(&**sub)),
                other => {
                    let t = other.type_name();
                    self.type_err(
                        format!("`in <string>` needs a string on the left, not {t}"),
                        span,
                    )
                }
            },
            Value::Map(m) | Value::Set(m) => Ok(m.borrow().contains(item)),
            Value::Range(r) => Ok(match item {
                Value::Int(i) => r.contains(*i),
                Value::Float(f) if f.fract() == 0.0 => r.contains(*f as i64),
                _ => false,
            }),
            Value::Tensor(t) => {
                let Some(x) = item.as_f64() else {
                    return Ok(false);
                };
                Ok(t.with_data(|d| d.contains(&x)))
            }
            Value::Instance(_) => match self.call_overload(
                container,
                &["__contains__", "contains", "includes", "include?"],
                vec![item.clone()],
                span,
            )? {
                Some(v) => Ok(v.truthy()),
                None => self.type_err("this object does not support `in`", span),
            },
            Value::Enum(e) => Ok(matches!(item, Value::Variant(v) if Rc::ptr_eq(&v.ty, e))),
            other => {
                let t = other.type_name();
                self.type_err(format!("`in` needs a collection, not {t}"), span)
            }
        }
    }
}

impl Tensor {
    /// Elementwise op without autograd (for `%` and floor division).
    pub fn binary_map(
        &self,
        other: &Tensor,
        f: impl Fn(f64, f64) -> f64,
    ) -> Result<Tensor, String> {
        let shape = crate::ml::tensor::broadcast_shape(self.shape(), other.shape())?;
        let a = self.broadcast_data(&shape);
        let b = other.broadcast_data(&shape);
        Tensor::new(a.iter().zip(&b).map(|(&x, &y)| f(x, y)).collect(), shape)
    }
}

fn seq_items(v: &Value) -> Vec<Value> {
    match v {
        Value::List(l) => l.borrow().clone(),
        Value::Tuple(t) => t.to_vec(),
        _ => Vec::new(),
    }
}

/// Identity (`is`): same object for references, same value for immediates.
pub fn identical(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Nil, Value::Nil) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Type(x), Value::Type(y)) => x == y,
        (Value::Nil, _) | (_, Value::Nil) => false,
        _ => {
            std::mem::discriminant(a) == std::mem::discriminant(b)
                && ptr_of(a) == ptr_of(b)
                && ptr_of(a) != 0
        }
    }
}

fn strict_type(v: &Value) -> u8 {
    match v {
        Value::Int(_) | Value::Big(_) => 1,
        Value::Float(_) => 2,
        Value::Bool(_) => 3,
        Value::Str(_) => 4,
        Value::Nil => 5,
        _ => 6,
    }
}

/// JavaScript's `typeof` (with Chimera's extra types spelled plainly).
fn js_typeof(v: &Value) -> String {
    match v {
        Value::Nil => "undefined".into(),
        Value::Bool(_) => "boolean".into(),
        Value::Int(_) | Value::Float(_) | Value::Big(_) => "number".into(),
        Value::Str(_) => "string".into(),
        Value::Func(_) | Value::Native(_) | Value::Bound(_) | Value::Class(_) | Value::Type(_) => {
            "function".into()
        }
        _ => "object".into(),
    }
}
