//! The standard library.
//!
//! Built-ins are registered under every name programmers reach for —
//! `print`/`puts`/`console.log`/`System.out.println`/`fmt.Println`, `len`/
//! `length`/`size`, `map`/`collect`, … — so code written in any style finds
//! what it expects.
//!
//! Layout: [`core`] (global functions), [`methods`] (methods of built-in
//! types), [`math`] (math and random), [`objects`] (stream objects such as
//! `std::cout` and `System.out`), [`json`] and [`modules`] (importable
//! modules). The Chimera-language part of the library lives in
//! `prelude.chi`.

pub mod core;
pub mod json;
pub mod math;
pub mod methods;
pub mod modules;
pub mod objects;

use std::cell::RefCell;
use std::rc::Rc;

use chimera_core::Span;

use crate::bigint::BigInt;
use crate::interp::{Args, Interp, sym};
use crate::value::*;

/// The Chimera-language prelude (Result, helpers).
pub const PRELUDE: &str = include_str!("prelude.chi");

/// Installs every built-in into a fresh interpreter.
pub fn install(interp: &mut Interp) {
    install_errors(interp);
    install_types(interp);
    core::install(interp);
    math::install(interp);
    objects::install(interp);
    crate::ml::install(interp);
    crate::ai::install(interp);
    run_prelude(interp);
}

fn run_prelude(interp: &mut Interp) {
    let file = interp.add_file("<prelude>", PRELUDE);
    let (stmts, errors) = crate::parser::parse(PRELUDE, file);
    if !errors.is_empty() {
        let src = chimera_core::SourceFile::new("<prelude>", PRELUDE);
        for e in &errors {
            eprintln!("{}", e.render(&src, false));
        }
        panic!("the prelude failed to parse");
    }
    let scope = crate::interp::Env::global();
    if interp.run_in(&stmts, file, &scope).is_ok() {
        for (k, v) in scope.local_bindings() {
            interp.builtins.insert(k, v);
        }
    }
}

/// Registers `f` under each of `names`.
pub fn def(interp: &mut Interp, names: &[&str], f: impl Fn(&mut Interp, Args) -> R + 'static) {
    let value = Value::native(names[0], f);
    for n in names {
        interp.builtins.insert(sym(n), value.clone());
    }
}

/// Registers a constant under each of `names`.
pub fn constant(interp: &mut Interp, names: &[&str], v: Value) {
    for n in names {
        interp.builtins.insert(sym(n), v.clone());
    }
}

// ----- argument helpers -------------------------------------------------------

/// A required positional argument.
pub fn arg(interp: &mut Interp, args: &Args, i: usize, fname: &str) -> R {
    match args.get(i) {
        Some(v) => Ok(v.clone()),
        None => interp.type_err(
            format!("`{fname}` is missing argument {}", i + 1),
            args.span,
        ),
    }
}

/// A required integer argument.
pub fn int_arg(interp: &mut Interp, args: &Args, i: usize, fname: &str) -> R<i64> {
    let v = arg(interp, args, i, fname)?;
    match v.as_int() {
        Some(n) => Ok(n),
        None => {
            let t = v.type_name();
            interp.type_err(format!("`{fname}` needs an integer, got {t}"), args.span)
        }
    }
}

/// A required numeric argument.
pub fn float_arg(interp: &mut Interp, args: &Args, i: usize, fname: &str) -> R<f64> {
    let v = arg(interp, args, i, fname)?;
    match v.as_f64() {
        Some(n) => Ok(n),
        None => {
            let t = v.type_name();
            interp.type_err(format!("`{fname}` needs a number, got {t}"), args.span)
        }
    }
}

/// A required string argument.
pub fn str_arg(interp: &mut Interp, args: &Args, i: usize, fname: &str) -> R<Rc<str>> {
    let v = arg(interp, args, i, fname)?;
    match v {
        Value::Str(s) => Ok(s),
        other => {
            let t = other.type_name();
            interp.type_err(format!("`{fname}` needs a string, got {t}"), args.span)
        }
    }
}

/// Calls a function with positional arguments.
pub fn call(interp: &mut Interp, f: &Value, args: Vec<Value>, span: Span) -> R {
    interp.call_value(f.clone(), Args::new(args, span))
}

/// Calls a callback with as many of `args` as it accepts (JS `map` passes
/// `(item, index)`, but `x => x * 2` only wants the first).
pub fn call_flex(interp: &mut Interp, f: &Value, args: Vec<Value>, span: Span) -> R {
    let wanted = match f {
        Value::Func(c) if !c.def.implicit_params => {
            if c.def
                .params
                .iter()
                .any(|p| p.kind != crate::ast::ParamKind::Normal)
            {
                args.len()
            } else {
                c.def.params.len().max(c.required_params())
            }
        }
        Value::Bound(b) => match &b.func {
            Value::Func(c) => c.def.params.len(),
            _ => args.len(),
        },
        _ => args.len(),
    };
    let mut args = args;
    if wanted < args.len() {
        args.truncate(wanted.max(1).min(args.len()));
    }
    call(interp, f, args, span)
}

// ----- errors -----------------------------------------------------------------

fn install_errors(interp: &mut Interp) {
    let base = Rc::new(Class::new("Error", None));
    let init = Value::native("Error.init", |interp, args| {
        let this = args.pos.first().cloned().unwrap_or(Value::Nil);
        if let Value::Instance(inst) = &this {
            let msg = match args
                .pos
                .get(1)
                .or_else(|| args.named.first().map(|(_, v)| v))
            {
                Some(Value::Str(s)) => Value::Str(s.clone()),
                Some(other) => {
                    let s = interp.to_str(other)?;
                    Value::str(&s)
                }
                None => Value::str(""),
            };
            inst.set(sym("message"), msg);
            if args.pos.len() > 2 {
                inst.set(sym("args"), Value::tuple(args.pos[1..].to_vec()));
            }
        }
        Ok(Value::Nil)
    });
    base.methods.borrow_mut().insert(sym("init"), init);
    let get_message = Value::native("getMessage", |_interp, args| {
        Ok(match args.pos.first() {
            Some(Value::Instance(i)) => i.get(sym("message")).unwrap_or(Value::Nil),
            _ => Value::Nil,
        })
    });
    for n in ["getMessage", "what", "Error", "localizedDescription"] {
        base.methods
            .borrow_mut()
            .insert(sym(n), get_message.clone());
    }
    let register = |interp: &mut Interp, class: &Rc<Class>, names: &[&'static str]| {
        for n in names {
            interp.error_classes.insert(n, class.clone());
            interp.builtins.insert(sym(n), Value::Class(class.clone()));
        }
    };
    register(
        interp,
        &base,
        &[
            "Error",
            "Exception",
            "StandardError",
            "BaseException",
            "Throwable",
            "RuntimeException",
            "RuntimeError",
            "panic_error",
        ],
    );
    let kinds: &[&[&'static str]] = &[
        &["TypeError", "ClassCastException"],
        &[
            "ValueError",
            "IllegalArgumentException",
            "ArgumentError",
            "ArgumentException",
        ],
        &["NameError", "ReferenceError", "NoMethodError"],
        &[
            "IndexError",
            "IndexOutOfBoundsException",
            "ArrayIndexOutOfBoundsException",
            "RangeError",
        ],
        &["KeyError", "NoSuchElementException", "KeyNotFoundException"],
        &[
            "ZeroDivisionError",
            "ArithmeticException",
            "DivideByZeroException",
            "ArithmeticError",
            "FloatingPointError",
        ],
        &[
            "AttributeError",
            "NullPointerException",
            "NullReferenceException",
        ],
        &["RecursionError", "StackOverflowError"],
        &["AssertionError", "AssertionFailedError"],
        &["ImportError", "ModuleNotFoundError", "LoadError"],
        &["IOError", "FileNotFoundError", "IOException", "OSError"],
        &["NotImplementedError", "UnsupportedOperationException"],
        &["StopIteration"],
        &["AiError"],
        &["OverflowError"],
        &["IllegalStateException", "InvalidOperationException"],
        &["SyntaxError"],
        &["TimeoutError"],
        &["PermissionError"],
    ];
    for names in kinds {
        let c = Rc::new(Class::new(names[0], Some(base.clone())));
        register(interp, &c, names);
    }
}

// ----- types -----------------------------------------------------------------------

fn install_types(interp: &mut Interp) {
    let types: &[(&[&str], TypeTag)] = &[
        (
            &[
                "int", "Int", "Integer", "integer", "long", "Long", "i64", "i32", "BigInt",
            ],
            TypeTag::Int,
        ),
        (
            &["float", "Float", "Double", "double", "f64", "f32", "real"],
            TypeTag::Float,
        ),
        (&["str", "String", "string", "Symbol"], TypeTag::Str),
        (&["bool", "Bool", "Boolean", "boolean"], TypeTag::Bool),
        (
            &[
                "list",
                "List",
                "Array",
                "array",
                "Vec",
                "ArrayList",
                "vector",
            ],
            TypeTag::List,
        ),
        (&["tuple", "Tuple"], TypeTag::Tuple),
        (
            &[
                "dict",
                "Dict",
                "Map",
                "HashMap",
                "Hash",
                "Object",
                "object",
                "Dictionary",
            ],
            TypeTag::Map,
        ),
        (&["set", "Set", "HashSet", "frozenset"], TypeTag::Set),
        (&["range", "Range"], TypeTag::Range),
        (&["number", "Number", "Numeric"], TypeTag::Number),
        (&["function", "Function", "callable"], TypeTag::Func),
        (&["type", "Type"], TypeTag::Type),
        (&["NoneType", "NilClass", "Unit"], TypeTag::Nil),
    ];
    for (names, tag) in types {
        for n in *names {
            interp.builtins.insert(sym(n), Value::Type(*tag));
        }
    }
}

/// Calling a built-in type: `int("42")`, `list(range(3))`, `dict(a=1)`, `type(x)`.
pub fn convert(interp: &mut Interp, tag: TypeTag, args: Args) -> R {
    let span = args.span;
    let first = args.pos.first().cloned();
    match tag {
        TypeTag::Int => {
            let Some(v) = first else {
                return Ok(Value::Int(0));
            };
            let base = match args.arg(1, "base") {
                Some(b) => b.as_int().unwrap_or(10) as u32,
                None => 10,
            };
            to_int(interp, &v, base, span)
        }
        TypeTag::Float => {
            let Some(v) = first else {
                return Ok(Value::Float(0.0));
            };
            match &v {
                Value::Str(s) => parse_float(s).map(Value::Float).map_or_else(
                    || {
                        interp.value_err(
                            format!(
                                "could not convert {} to float",
                                chimera_core::text::escape_debug(s)
                            ),
                            span,
                        )
                    },
                    Ok,
                ),
                other => match other.as_f64() {
                    Some(f) => Ok(Value::Float(f)),
                    None => {
                        let t = other.type_name();
                        interp.type_err(format!("float() cannot convert a {t}"), span)
                    }
                },
            }
        }
        TypeTag::Number => {
            let Some(v) = first else {
                return Ok(Value::Int(0));
            };
            match &v {
                Value::Str(s) => {
                    let t = s.trim();
                    if let Ok(i) = t.parse::<i64>() {
                        Ok(Value::Int(i))
                    } else {
                        Ok(Value::Float(parse_float(t).unwrap_or(f64::NAN)))
                    }
                }
                Value::Bool(b) => Ok(Value::Int(i64::from(*b))),
                Value::Nil => Ok(Value::Int(0)),
                other if other.is_number() => Ok(other.clone()),
                _ => Ok(Value::Float(f64::NAN)),
            }
        }
        TypeTag::Str => match first {
            None => Ok(Value::str("")),
            Some(v) => {
                let s = interp.to_str(&v)?;
                Ok(Value::str(&s))
            }
        },
        TypeTag::Bool => Ok(Value::Bool(first.is_some_and(|v| v.truthy()))),
        TypeTag::List => match first {
            None => Ok(Value::list(Vec::new())),
            Some(Value::Int(n)) if args.pos.len() == 1 && false => {
                Ok(Value::list(vec![Value::Nil; n.max(0) as usize]))
            }
            Some(v) => Ok(Value::list(interp.collect(&v, span)?)),
        },
        TypeTag::Tuple => match first {
            None => Ok(Value::tuple(Vec::new())),
            Some(v) => Ok(Value::tuple(interp.collect(&v, span)?)),
        },
        TypeTag::Set => {
            let mut d = Dict::new();
            if let Some(v) = first {
                for x in interp.collect(&v, span)? {
                    if !d.insert(x.clone(), Value::Nil) {
                        let t = x.type_name();
                        return interp.type_err(format!("unhashable type: {t}"), span);
                    }
                }
            }
            Ok(Value::Set(Rc::new(RefCell::new(d))))
        }
        TypeTag::Map | TypeTag::Any => {
            let mut d = Dict::new();
            if let Some(v) = first {
                match &v {
                    Value::Map(m) => d = m.borrow().clone(),
                    Value::Instance(i) => {
                        for (k, x) in i.fields.borrow().iter() {
                            d.insert(Value::Str(k.as_str()), x.clone());
                        }
                    }
                    other => {
                        for pair in interp.collect(other, span)? {
                            let kv = interp.collect(&pair, span)?;
                            if kv.len() != 2 {
                                return interp.value_err("dict() needs (key, value) pairs", span);
                            }
                            d.insert(kv[0].clone(), kv[1].clone());
                        }
                    }
                }
            }
            for (k, v) in args.named {
                d.insert(Value::Str(k.as_str()), v);
            }
            Ok(Value::Map(Rc::new(RefCell::new(d))))
        }
        TypeTag::Range => {
            let ints: Vec<i64> = args.pos.iter().map(|v| v.as_int().unwrap_or(0)).collect();
            let (start, end, step) = match ints.as_slice() {
                [] => (0, 0, 1),
                [n] => (0, *n, 1),
                [a, b] => (*a, *b, 1),
                [a, b, s, ..] => (*a, *b, *s),
            };
            if step == 0 {
                return interp.value_err("range() step cannot be zero", span);
            }
            Ok(Value::Range(Rc::new(RangeVal { start, end, step })))
        }
        TypeTag::Tensor => crate::ml::make_tensor(interp, args),
        TypeTag::Type => Ok(match first {
            None => Value::Nil,
            Some(v) => type_of(&v),
        }),
        TypeTag::Nil => Ok(Value::Nil),
        TypeTag::Func => interp.type_err("cannot construct a function this way (use `fn`)", span),
    }
}

/// The type of a value, as a value.
pub fn type_of(v: &Value) -> Value {
    match v {
        Value::Instance(i) => Value::Class(i.class.clone()),
        Value::Variant(var) => Value::Enum(var.ty.clone()),
        Value::Nil => Value::Type(TypeTag::Nil),
        Value::Bool(_) => Value::Type(TypeTag::Bool),
        Value::Int(_) | Value::Big(_) => Value::Type(TypeTag::Int),
        Value::Float(_) => Value::Type(TypeTag::Float),
        Value::Str(_) => Value::Type(TypeTag::Str),
        Value::List(_) => Value::Type(TypeTag::List),
        Value::Tuple(_) => Value::Type(TypeTag::Tuple),
        Value::Map(_) => Value::Type(TypeTag::Map),
        Value::Set(_) => Value::Type(TypeTag::Set),
        Value::Range(_) => Value::Type(TypeTag::Range),
        Value::Tensor(_) => Value::Type(TypeTag::Tensor),
        Value::Class(_) | Value::Type(_) | Value::Enum(_) => Value::Type(TypeTag::Type),
        Value::Func(_) | Value::Native(_) | Value::Bound(_) => Value::Type(TypeTag::Func),
        _ => Value::Type(TypeTag::Any),
    }
}

/// Parses a float accepting `inf`, `nan`, underscores and surrounding space.
pub fn parse_float(s: &str) -> Option<f64> {
    let t: String = s.trim().chars().filter(|&c| c != '_').collect();
    match t.to_ascii_lowercase().as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        "nan" => Some(f64::NAN),
        _ => t.parse().ok(),
    }
}

/// `int(v, base)`.
pub fn to_int(interp: &mut Interp, v: &Value, base: u32, span: Span) -> R {
    match v {
        Value::Int(_) | Value::Big(_) => Ok(v.clone()),
        Value::Bool(b) => Ok(Value::Int(i64::from(*b))),
        Value::Float(f) => {
            if !f.is_finite() {
                return interp.value_err(
                    format!(
                        "cannot convert {} to an integer",
                        super::interp::format::fmt_float(*f)
                    ),
                    span,
                );
            }
            match BigInt::from_f64(*f) {
                Some(b) => Ok(Value::from_big(b)),
                None => interp.value_err("cannot convert this float to an integer", span),
            }
        }
        Value::Str(s) => {
            let t: String = s.trim().chars().filter(|&c| c != '_').collect();
            let (neg, digits) = match t.strip_prefix('-') {
                Some(d) => (true, d.to_owned()),
                None => (false, t.strip_prefix('+').unwrap_or(&t).to_owned()),
            };
            let (base, digits) = match (
                base,
                digits.get(..2).map(str::to_ascii_lowercase).as_deref(),
            ) {
                (16 | 10, Some("0x")) if base == 16 || base == 10 && false => {
                    (16, digits[2..].to_owned())
                }
                (16, Some("0x")) => (16, digits[2..].to_owned()),
                (2, Some("0b")) => (2, digits[2..].to_owned()),
                (8, Some("0o")) => (8, digits[2..].to_owned()),
                _ => (base, digits),
            };
            match BigInt::parse_radix(&digits, base) {
                Some(b) => Ok(Value::from_big(if neg { b.neg() } else { b })),
                None => {
                    // `int("3.0")` is forgiving when the value is integral.
                    if base == 10
                        && let Some(f) = parse_float(&t)
                        && f.fract() == 0.0
                        && f.is_finite()
                    {
                        return Ok(Value::Int(f as i64));
                    }
                    interp.value_err(
                        format!(
                            "invalid integer literal {}",
                            chimera_core::text::escape_debug(s)
                        ),
                        span,
                    )
                }
            }
        }
        Value::Dual(d) => to_int(interp, &d.re.clone(), base, span),
        Value::Tensor(t) if t.numel() == 1 => Ok(Value::Int(t.item() as i64)),
        Value::Nil => Ok(Value::Int(0)),
        other => {
            let t = other.type_name();
            interp.type_err(format!("int() cannot convert a {t}"), span)
        }
    }
}

/// Static attributes of built-in types: `int.MAX`, `Integer.parseInt`,
/// `String.valueOf`, `Object.keys`, `Number.isInteger`, `float.inf`, …
pub fn type_attr(interp: &mut Interp, tag: TypeTag, name: &str, span: Span) -> R<Option<Value>> {
    let _ = span;
    let native = |n: &str, f: fn(&mut Interp, Args) -> R| Some(Value::native(n, f));
    Ok(match (tag, name) {
        (TypeTag::Int | TypeTag::Number, "MAX" | "MAX_VALUE" | "max" | "MaxValue") => {
            Some(Value::Int(i64::MAX))
        }
        (TypeTag::Int | TypeTag::Number, "MIN" | "MIN_VALUE" | "min" | "MinValue") => {
            Some(Value::Int(i64::MIN))
        }
        (TypeTag::Number, "MAX_SAFE_INTEGER") => Some(Value::Int((1 << 53) - 1)),
        (TypeTag::Number, "EPSILON") | (TypeTag::Float, "EPSILON" | "epsilon") => {
            Some(Value::Float(f64::EPSILON))
        }
        (TypeTag::Float, "inf" | "INFINITY" | "POSITIVE_INFINITY" | "infinity") => {
            Some(Value::Float(f64::INFINITY))
        }
        (TypeTag::Number, "POSITIVE_INFINITY") => Some(Value::Float(f64::INFINITY)),
        (TypeTag::Float | TypeTag::Number, "nan" | "NaN" | "NAN") => Some(Value::Float(f64::NAN)),
        (TypeTag::Float, "MAX" | "MAX_VALUE" | "max") => Some(Value::Float(f64::MAX)),
        (
            TypeTag::Int | TypeTag::Number,
            "parseInt" | "parse" | "valueOf" | "from_str" | "fromString",
        ) => native("parseInt", |interp, args| {
            let v = arg(interp, &args, 0, "parseInt")?;
            let base = args.get(1).and_then(Value::as_int).unwrap_or(10) as u32;
            let span = args.span;
            to_int(interp, &v, base, span)
        }),
        (TypeTag::Float | TypeTag::Number, "parseFloat" | "parseDouble" | "parse") => {
            native("parseFloat", |interp, args| {
                let s = str_arg(interp, &args, 0, "parseFloat")?;
                Ok(Value::Float(parse_float(&s).unwrap_or(f64::NAN)))
            })
        }
        (TypeTag::Number, "isInteger") => native("isInteger", |_interp, args| {
            Ok(Value::Bool(match args.get(0) {
                Some(Value::Int(_) | Value::Big(_)) => true,
                Some(Value::Float(f)) => f.fract() == 0.0,
                _ => false,
            }))
        }),
        (TypeTag::Number, "isNaN") | (TypeTag::Float, "isNaN" | "is_nan") => {
            native("isNaN", |_interp, args| {
                Ok(Value::Bool(
                    args.get(0).and_then(Value::as_f64).is_none_or(f64::is_nan),
                ))
            })
        }
        (TypeTag::Str, "valueOf" | "from" | "toString") => native("valueOf", |interp, args| {
            let v = args.get(0).cloned().unwrap_or(Value::Nil);
            let s = interp.to_str(&v)?;
            Ok(Value::str(&s))
        }),
        (TypeTag::Str, "format") => native("String.format", |interp, args| {
            let fmt = str_arg(interp, &args, 0, "String.format")?;
            let s = interp.printf(&fmt, &args.pos[1..], args.span)?;
            Ok(Value::str(&s))
        }),
        (TypeTag::Str, "join") => native("String.join", |interp, args| {
            let sep = str_arg(interp, &args, 0, "String.join")?;
            let rest: Vec<Value> = if args.pos.len() == 2 {
                interp.collect(&args.pos[1].clone(), args.span)?
            } else {
                args.pos[1..].to_vec()
            };
            let mut parts = Vec::with_capacity(rest.len());
            for v in &rest {
                parts.push(interp.to_str(v)?);
            }
            Ok(Value::str(&parts.join(&sep)))
        }),
        (TypeTag::Str, "fromCharCode") => native("fromCharCode", |_interp, args| {
            let s: String = args
                .pos
                .iter()
                .filter_map(|v| v.as_int())
                .filter_map(|i| char::from_u32(i as u32))
                .collect();
            Ok(Value::str(&s))
        }),
        (TypeTag::List, "of" | "from" | "asList" | "new") => {
            native("Array.from", |interp, args| {
                let from_like =
                    args.pos.len() == 1 || (args.pos.len() == 2 && args.pos[1].is_callable());
                if from_like && !args.pos[0].is_number() {
                    let v = args.pos[0].clone();
                    // `Array.from({length: n}, f)`.
                    let items = match &v {
                        Value::Map(m) if m.borrow().get_str("length").is_some() => {
                            let n = m
                                .borrow()
                                .get_str("length")
                                .and_then(|n| n.as_int())
                                .unwrap_or(0)
                                .max(0);
                            vec![Value::Nil; n as usize]
                        }
                        other => interp.collect(other, args.span)?,
                    };
                    if let Some(f) = args.get(1) {
                        let mut out = Vec::with_capacity(items.len());
                        for (i, x) in items.into_iter().enumerate() {
                            out.push(call_flex(
                                interp,
                                &f.clone(),
                                vec![x, Value::Int(i as i64)],
                                args.span,
                            )?);
                        }
                        return Ok(Value::list(out));
                    }
                    return Ok(Value::list(items));
                }
                Ok(Value::list(args.pos))
            })
        }
        (TypeTag::List, "isArray") => native("isArray", |_interp, args| {
            Ok(Value::Bool(matches!(args.get(0), Some(Value::List(_)))))
        }),
        (TypeTag::Map, "keys") => native("Object.keys", |interp, args| {
            let v = arg(interp, &args, 0, "Object.keys")?;
            let span = args.span;
            interp.call_method(&v, sym("keys"), Args::new(Vec::new(), span), span)
        }),
        (TypeTag::Map, "values") => native("Object.values", |interp, args| {
            let v = arg(interp, &args, 0, "Object.values")?;
            let span = args.span;
            interp.call_method(&v, sym("values"), Args::new(Vec::new(), span), span)
        }),
        (TypeTag::Map, "entries" | "items") => native("Object.entries", |interp, args| {
            let v = arg(interp, &args, 0, "Object.entries")?;
            let span = args.span;
            let pairs =
                interp.call_method(&v, sym("entries"), Args::new(Vec::new(), span), span)?;
            // JavaScript entries are two-element arrays.
            let mut out = Vec::new();
            for p in interp.collect(&pairs, span)? {
                out.push(Value::list(interp.collect(&p, span)?));
            }
            Ok(Value::list(out))
        }),
        (TypeTag::Map, "assign") => native("Object.assign", |interp, args| {
            let target = arg(interp, &args, 0, "Object.assign")?;
            let Value::Map(t) = &target else {
                return interp.type_err("Object.assign needs a map target", args.span);
            };
            for src in &args.pos[1..] {
                if let Value::Map(s) = src {
                    let entries: Vec<(Value, Value)> = s
                        .borrow()
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    for (k, v) in entries {
                        t.borrow_mut().insert(k, v);
                    }
                }
            }
            Ok(target)
        }),
        (TypeTag::Map, "fromkeys" | "fromEntries") => native("fromkeys", |interp, args| {
            let src = arg(interp, &args, 0, "fromkeys")?;
            let default = args.get(1).cloned();
            let mut d = Dict::new();
            for item in interp.collect(&src, args.span)? {
                match &default {
                    Some(v) => {
                        d.insert(item, v.clone());
                    }
                    None => {
                        let kv = interp.collect(&item, args.span)?;
                        if kv.len() == 2 {
                            d.insert(kv[0].clone(), kv[1].clone());
                        } else {
                            d.insert(item, Value::Nil);
                        }
                    }
                }
            }
            Ok(Value::Map(Rc::new(RefCell::new(d))))
        }),
        _ => None,
    })
}
