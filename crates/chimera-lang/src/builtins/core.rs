//! Global functions.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;

use chimera_core::Span;

use super::{arg, call, call_flex, def, int_arg, str_arg};
use crate::interp::{Args, Flow, Interp, sym};
use crate::value::*;

/// Stable merge sort that propagates comparison errors. `cmp` is either a
/// one-argument key function or a two-argument comparator (JS/Ruby style).
pub fn sort_values(
    interp: &mut Interp,
    items: Vec<Value>,
    func: Option<&Value>,
    reverse: bool,
    span: Span,
) -> R<Vec<Value>> {
    let comparator = match func {
        Some(Value::Func(c)) => c.def.params.len() >= 2,
        Some(Value::Native(_)) => false,
        Some(Value::Bound(b)) => matches!(&b.func, Value::Func(c) if c.def.params.len() >= 2),
        _ => false,
    };
    // Decorate with keys up front for key functions.
    let keyed: Vec<(Value, Value)> = match (func, comparator) {
        (Some(f), false) => {
            let mut out = Vec::with_capacity(items.len());
            for x in items {
                let k = call(interp, f, vec![x.clone()], span)?;
                out.push((k, x));
            }
            out
        }
        _ => items.into_iter().map(|x| (x.clone(), x)).collect(),
    };
    let cmp_fn = if comparator { func.cloned() } else { None };
    let compare = |interp: &mut Interp, a: &Value, b: &Value| -> R<Ordering> {
        let o = match &cmp_fn {
            Some(f) => {
                let r = call(interp, f, vec![a.clone(), b.clone()], span)?;
                match r {
                    Value::Bool(true) => Ordering::Less,
                    Value::Bool(false) => Ordering::Greater,
                    other => {
                        let n = other.as_f64().unwrap_or(0.0);
                        if n < 0.0 {
                            Ordering::Less
                        } else if n > 0.0 {
                            Ordering::Greater
                        } else {
                            Ordering::Equal
                        }
                    }
                }
            }
            None => interp.compare_values(a, b, span)?,
        };
        Ok(if reverse { o.reverse() } else { o })
    };
    let mut v = keyed;
    // Bottom-up merge sort.
    let n = v.len();
    let mut width = 1;
    while width < n {
        let mut merged: Vec<(Value, Value)> = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let mid = (i + width).min(n);
            let end = (i + 2 * width).min(n);
            let (mut a, mut b) = (i, mid);
            while a < mid && b < end {
                if compare(interp, &v[b].0, &v[a].0)? == Ordering::Less {
                    merged.push(v[b].clone());
                    b += 1;
                } else {
                    merged.push(v[a].clone());
                    a += 1;
                }
            }
            merged.extend_from_slice(&v[a..mid]);
            merged.extend_from_slice(&v[b..end]);
            i = end;
        }
        v = merged;
        width *= 2;
    }
    Ok(v.into_iter().map(|(_, x)| x).collect())
}

/// Splits `(f, xs)` / `(xs, f)` argument orders.
fn fn_and_seq(interp: &mut Interp, args: &Args, name: &str) -> R<(Value, Value)> {
    let a = arg(interp, args, 0, name)?;
    let b = arg(interp, args, 1, name)?;
    // Either order works: `filter(f, xs)` (Python/Haskell) or `filter(xs, f)`.
    let is_fn = |v: &Value| matches!(v, Value::Func(_) | Value::Native(_) | Value::Bound(_));
    if is_fn(&a) && !is_fn(&b) || a.is_callable() && !b.is_callable() {
        Ok((a, b))
    } else {
        Ok((b, a))
    }
}

fn join_display(interp: &mut Interp, args: &[Value], sep: &str) -> R<String> {
    let mut parts = Vec::with_capacity(args.len());
    for a in args {
        parts.push(interp.to_str(a)?);
    }
    Ok(parts.join(sep))
}

fn write_line(interp: &mut Interp, s: &str) {
    interp.write_out(s);
    interp.write_out("\n");
}

/// Rust `format!`-family: first argument is a `{}` template.
fn rust_format(interp: &mut Interp, args: &Args, name: &str) -> R<String> {
    if args.pos.is_empty() {
        return Ok(String::new());
    }
    let template = match &args.pos[0] {
        Value::Str(s) => s.clone(),
        other => {
            let s = interp.to_str(other)?;
            return Ok(s);
        }
    };
    let _ = name;
    let env = interp.call_env.clone();
    interp.brace_format(
        &template,
        &args.pos[1..],
        &args.named,
        env.as_ref(),
        args.span,
    )
}

fn read_line() -> Option<String> {
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line.trim_end_matches(['\n', '\r']).to_owned()),
    }
}

/// Throws a `SystemExit`-style error carrying an exit code.
pub fn exit_flow(interp: &mut Interp, code: i64, span: Span) -> Flow {
    let flow = interp.error("RuntimeError", format!("exit({code})"), span);
    if let Flow::Throw(t) = &flow
        && let Value::Instance(i) = &t.value
    {
        i.set(sym("__exit_code__"), Value::Int(code));
    }
    flow
}

/// Installs the global functions.
pub fn install(interp: &mut Interp) {
    interp
        .builtins
        .insert(sym("__name__"), Value::str("__main__"));

    // ----- iteration ---------------------------------------------------------
    def(interp, &["next"], |interp, args| {
        let it = arg(interp, &args, 0, "next")?;
        crate::interp::generator::next_value(interp, &it, args.get(1).cloned(), args.span)
    });
    def(interp, &["iter", "iterator"], |interp, args| {
        let v = arg(interp, &args, 0, "iter")?;
        if let Value::Object(o) = &v
            && (o.as_any().is::<crate::interp::generator::Generator>()
                || o.as_any().is::<crate::interp::generator::IterObject>())
        {
            return Ok(v);
        }
        let it = interp.iterate(&v, args.span)?;
        Ok(Value::Object(Rc::new(
            crate::interp::generator::IterObject::new(it),
        )))
    });

    // ----- output ----------------------------------------------------------
    def(interp, &["print"], |interp, args| {
        let sep = match args.named("sep").or_else(|| args.named("separator")) {
            Some(v) => interp.to_str(&v)?,
            None => " ".into(),
        };
        // Python's `end=`, Swift's `terminator:`.
        let end = match args.named("end").or_else(|| args.named("terminator")) {
            Some(v) => interp.to_str(&v)?,
            None => "\n".into(),
        };
        let s = join_display(interp, &args.pos, &sep)?;
        interp.write_out(&s);
        interp.write_out(&end);
        Ok(Value::Nil)
    });
    def(
        interp,
        &[
            "println",
            "say",
            "echo",
            "display",
            "show_line",
            "putStrLn",
            "print_endline",
            "disp",
            "alert",
            "writeln",
            "Println",
            "print_line",
            "printLn",
            "NSLog",
            "log",
        ],
        |interp, args| {
            let s = join_display(interp, &args.pos, " ")?;
            write_line(interp, &s);
            Ok(Value::Nil)
        },
    );
    def(interp, &["puts"], |interp, args| {
        if args.pos.is_empty() {
            interp.write_out("\n");
        }
        for a in &args.pos {
            match a {
                Value::List(l) => {
                    let items = l.borrow().clone();
                    for x in items {
                        let s = interp.to_str(&x)?;
                        write_line(interp, &s);
                    }
                }
                other => {
                    let s = interp.to_str(other)?;
                    write_line(interp, &s);
                }
            }
        }
        Ok(Value::Nil)
    });
    def(
        interp,
        &["print_string", "write", "putStr", "print_no_newline"],
        |interp, args| {
            let s = join_display(interp, &args.pos, "")?;
            interp.write_out(&s);
            Ok(Value::Nil)
        },
    );
    def(
        interp,
        &[
            "p", "pp", "inspect", "dbg!", "dbg", "var_dump", "print_r", "debug",
        ],
        |interp, args| {
            for a in &args.pos {
                let s = interp.repr(a)?;
                write_line(interp, &s);
            }
            Ok(match args.pos.len() {
                0 => Value::Nil,
                1 => args.pos[0].clone(),
                _ => Value::tuple(args.pos.clone()),
            })
        },
    );
    def(interp, &["printf"], |interp, args| {
        let fmt = str_arg(interp, &args, 0, "printf")?;
        let s = interp.printf(&fmt, &args.pos[1..], args.span)?;
        interp.write_out(&s);
        Ok(Value::Nil)
    });
    def(interp, &["sprintf", "Sprintf"], |interp, args| {
        let fmt = str_arg(interp, &args, 0, "sprintf")?;
        let s = interp.printf(&fmt, &args.pos[1..], args.span)?;
        Ok(Value::str(&s))
    });
    def(interp, &["print!"], |interp, args| {
        let s = rust_format(interp, &args, "print!")?;
        interp.write_out(&s);
        Ok(Value::Nil)
    });
    def(interp, &["println!"], |interp, args| {
        let s = rust_format(interp, &args, "println!")?;
        write_line(interp, &s);
        Ok(Value::Nil)
    });
    def(
        interp,
        &["eprint!", "eprintln!", "eprintln", "eprint", "warn"],
        |interp, args| {
            let s = if matches!(args.get(0), Some(Value::Str(t)) if t.contains('{')) {
                rust_format(interp, &args, "eprintln!")?
            } else {
                join_display(interp, &args.pos, " ")?
            };
            if matches!(interp.output, crate::interp::Output::Stdout) {
                eprintln!("{s}");
            } else {
                write_line(interp, &s);
            }
            Ok(Value::Nil)
        },
    );
    def(interp, &["format!"], |interp, args| {
        let s = rust_format(interp, &args, "format!")?;
        Ok(Value::str(&s))
    });
    def(interp, &["format"], |interp, args| {
        // format(value, spec) | format("{} {}", a, b) | format("%d", n)
        let first = arg(interp, &args, 0, "format")?;
        match (&first, args.get(1)) {
            (Value::Str(t), _) if t.contains('{') => {
                let s = rust_format(interp, &args, "format")?;
                Ok(Value::str(&s))
            }
            (Value::Str(t), _) if t.contains('%') => {
                let s = interp.printf(t, &args.pos[1..], args.span)?;
                Ok(Value::str(&s))
            }
            (_, Some(Value::Str(spec))) => {
                let spec = spec.clone();
                let s = interp.format_spec(&first, &spec, args.span)?;
                Ok(Value::str(&s))
            }
            _ => {
                let s = interp.to_str(&first)?;
                Ok(Value::str(&s))
            }
        }
    });
    def(interp, &["write!", "writeln!"], |interp, args| {
        // `write!(f, "…", args)` inside a Rust `fmt` method.
        let target = arg(interp, &args, 0, "write!")?;
        let rest = Args {
            pos: args.pos[1..].to_vec(),
            named: args.named.clone(),
            span: args.span,
        };
        let s = rust_format(interp, &rest, "write!")?;
        let span = args.span;
        interp.call_method(
            &target,
            sym("write_str"),
            Args::new(vec![Value::str(&s)], span),
            span,
        )?;
        Ok(Value::Nil)
    });
    def(interp, &["vec!"], |_interp, args| Ok(Value::list(args.pos)));
    def(
        interp,
        &[
            "input",
            "gets",
            "readline",
            "readLine",
            "prompt",
            "ReadLine",
            "read_line",
            "raw_input",
        ],
        |interp, args| {
            if let Some(p) = args.get(0) {
                let s = interp.to_str(p)?;
                interp.write_out(&s);
            }
            Ok(read_line().map_or(Value::Nil, |s| Value::str(&s)))
        },
    );

    // ----- assertions and control ------------------------------------------
    def(
        interp,
        &["assert", "assert!", "require_that"],
        |interp, args| {
            let cond = arg(interp, &args, 0, "assert")?;
            if cond.truthy() {
                return Ok(Value::Nil);
            }
            let msg = match args.get(1) {
                Some(m) => interp.to_str(m)?,
                None => "assertion failed".into(),
            };
            Err(interp.error("AssertionError", msg, args.span))
        },
    );
    def(
        interp,
        &[
            "assert_eq",
            "assert_eq!",
            "assertEqual",
            "assertEquals",
            "assert_equal",
        ],
        |interp, args| {
            let a = arg(interp, &args, 0, "assert_eq")?;
            let b = arg(interp, &args, 1, "assert_eq")?;
            if interp.values_equal(&a, &b)? {
                return Ok(Value::Nil);
            }
            let (ra, rb) = (interp.repr(&a)?, interp.repr(&b)?);
            Err(interp.error(
                "AssertionError",
                format!("assertion failed: {ra} != {rb}"),
                args.span,
            ))
        },
    );
    def(
        interp,
        &["assert_ne", "assert_ne!", "assertNotEqual"],
        |interp, args| {
            let a = arg(interp, &args, 0, "assert_ne")?;
            let b = arg(interp, &args, 1, "assert_ne")?;
            if !interp.values_equal(&a, &b)? {
                return Ok(Value::Nil);
            }
            let ra = interp.repr(&a)?;
            Err(interp.error(
                "AssertionError",
                format!("assertion failed: both values are {ra}"),
                args.span,
            ))
        },
    );
    def(
        interp,
        &[
            "panic",
            "panic!",
            "error",
            "fail",
            "die",
            "abort",
            "unreachable!",
            "todo!",
            "unimplemented!",
        ],
        |interp, args| {
            let msg = if matches!(args.get(0), Some(Value::Str(t)) if t.contains('{')) {
                rust_format(interp, &args, "panic!")?
            } else {
                join_display(interp, &args.pos, " ")?
            };
            let msg = if msg.is_empty() {
                "explicit panic".into()
            } else {
                msg
            };
            Err(interp.error("RuntimeError", msg, args.span))
        },
    );
    def(
        interp,
        &["exit", "quit", "os_exit", "Exit"],
        |interp, args| {
            let code = args.get(0).and_then(Value::as_int).unwrap_or(0);
            Err(exit_flow(interp, code, args.span))
        },
    );
    def(
        interp,
        &["eval", "exec", "evaluate", "loadstring"],
        |interp, args| {
            let code = str_arg(interp, &args, 0, "eval")?;
            let file = interp.add_file("<eval>", &code);
            let (stmts, errors) = crate::parser::parse(&code, file);
            if let Some(e) = errors.first() {
                let rendered = e.render(&interp.files[file as usize], false);
                return Err(interp.error("SyntaxError", rendered, args.span));
            }
            let env = interp
                .call_env
                .clone()
                .unwrap_or_else(|| interp.globals.clone());
            interp.run_in(&stmts, file, &env)
        },
    );
    def(interp, &["sleep", "Sleep", "delay"], |interp, args| {
        let secs = super::float_arg(interp, &args, 0, "sleep")?;
        std::thread::sleep(std::time::Duration::from_secs_f64(secs.clamp(0.0, 3600.0)));
        Ok(Value::Nil)
    });
    def(
        interp,
        &["clock", "now", "time", "perf_counter", "process_time"],
        |_interp, _args| {
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            Ok(Value::Float(t))
        },
    );

    // ----- introspection -----------------------------------------------------
    def(
        interp,
        &["len", "length", "size", "count", "strlen", "len!"],
        |interp, args| {
            let v = arg(interp, &args, 0, "len")?;
            if let Some(item) = args.get(1) {
                // count(xs, item)
                let span = args.span;
                return interp.call_method(
                    &v,
                    sym("count"),
                    Args::new(vec![item.clone()], span),
                    span,
                );
            }
            length_of(interp, &v, args.span).map(|n| Value::Int(n as i64))
        },
    );
    def(
        interp,
        &["repr", "show", "inspect_str", "tostring_repr"],
        |interp, args| {
            let v = arg(interp, &args, 0, "repr")?;
            let s = interp.repr(&v)?;
            Ok(Value::str(&s))
        },
    );
    def(
        interp,
        &[
            "tostring",
            "to_string",
            "string_of_int",
            "string_of_float",
            "Itoa",
            "toString",
        ],
        |interp, args| {
            let v = arg(interp, &args, 0, "tostring")?;
            let s = interp.to_str(&v)?;
            Ok(Value::str(&s))
        },
    );
    def(interp, &["parseInt"], |interp, args| {
        // JavaScript: leading whitespace, optional sign, then as many digits
        // as parse; NaN when there are none.
        let v = arg(interp, &args, 0, "parseInt")?;
        let s = interp.to_str(&v)?;
        let radix = args
            .get(1)
            .and_then(Value::as_int)
            .unwrap_or(10)
            .clamp(2, 36) as u32;
        let t = s.trim_start();
        let (neg, t) = match t.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, t.strip_prefix('+').unwrap_or(t)),
        };
        let t = if radix == 16 {
            t.strip_prefix("0x")
                .or_else(|| t.strip_prefix("0X"))
                .unwrap_or(t)
        } else {
            t
        };
        let digits: String = t.chars().take_while(|c| c.is_digit(radix)).collect();
        if digits.is_empty() {
            return Ok(Value::Float(f64::NAN));
        }
        let span = args.span;
        let n = super::to_int(interp, &Value::str(&digits), radix, span)?;
        if neg {
            interp.unary_op(crate::ast::UnOp::Neg, n, span)
        } else {
            Ok(n)
        }
    });
    def(
        interp,
        &[
            "tonumber",
            "int_of_string",
            "float_of_string",
            "Atoi",
            "parse_int",
            "read",
        ],
        |interp, args| {
            let v = arg(interp, &args, 0, "tonumber")?;
            match &v {
                Value::Str(s) => {
                    let t = s.trim();
                    if let Some(base) = args.get(1).and_then(Value::as_int) {
                        let span = args.span;
                        return super::to_int(interp, &v, base as u32, span);
                    }
                    if let Ok(i) = t.parse::<i64>() {
                        return Ok(Value::Int(i));
                    }
                    Ok(super::parse_float(t).map_or(Value::Nil, Value::Float))
                }
                other => Ok(other.clone()),
            }
        },
    );
    def(
        interp,
        &["parseFloat", "float_of_int", "to_float", "parse_float"],
        |interp, args| {
            let v = arg(interp, &args, 0, "parseFloat")?;
            Ok(match &v {
                Value::Str(s) => super::parse_float(s).map_or(Value::Float(f64::NAN), Value::Float),
                other => other.as_f64().map_or(Value::Float(f64::NAN), Value::Float),
            })
        },
    );
    def(interp, &["isNaN", "is_nan"], |_interp, args| {
        Ok(Value::Bool(
            args.get(0).and_then(Value::as_f64).is_none_or(f64::is_nan),
        ))
    });
    def(
        interp,
        &["isinstance", "instanceof", "is_a", "is_instance"],
        |interp, args| {
            let v = arg(interp, &args, 0, "isinstance")?;
            let t = arg(interp, &args, 1, "isinstance")?;
            Ok(Value::Bool(match &t {
                Value::Tuple(ts) => ts.iter().any(|ty| interp.isinstance(&v, ty)),
                other => interp.isinstance(&v, other),
            }))
        },
    );
    def(interp, &["issubclass"], |interp, args| {
        let a = arg(interp, &args, 0, "issubclass")?;
        let b = arg(interp, &args, 1, "issubclass")?;
        Ok(Value::Bool(
            matches!((&a, &b), (Value::Class(x), Value::Class(y)) if x.is_subclass_of(y)),
        ))
    });
    def(
        interp,
        &["hasattr", "respond_to?", "has_key?"],
        |interp, args| {
            let v = arg(interp, &args, 0, "hasattr")?;
            let name = str_arg(interp, &args, 1, "hasattr")?;
            let span = args.span;
            Ok(Value::Bool(interp.get_attr(&v, sym(&name), span).is_ok()))
        },
    );
    def(interp, &["getattr"], |interp, args| {
        let v = arg(interp, &args, 0, "getattr")?;
        let name = str_arg(interp, &args, 1, "getattr")?;
        let span = args.span;
        match interp.get_attr(&v, sym(&name), span) {
            Ok(x) => Ok(x),
            Err(e) => match args.get(2) {
                Some(d) => Ok(d.clone()),
                None => Err(e),
            },
        }
    });
    def(interp, &["setattr"], |interp, args| {
        let v = arg(interp, &args, 0, "setattr")?;
        let name = str_arg(interp, &args, 1, "setattr")?;
        let x = arg(interp, &args, 2, "setattr")?;
        let span = args.span;
        interp.set_attr(&v, sym(&name), x, span)?;
        Ok(Value::Nil)
    });
    def(interp, &["callable", "is_function"], |_interp, args| {
        Ok(Value::Bool(args.get(0).is_some_and(Value::is_callable)))
    });
    def(
        interp,
        &["id", "object_id", "addressof"],
        |_interp, args| Ok(Value::Int(args.get(0).map_or(0, |v| ptr_of(v) as i64))),
    );
    def(interp, &["hash", "hashCode"], |interp, args| {
        let v = arg(interp, &args, 0, "hash")?;
        match key_of(&v) {
            Some(k) => {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                k.hash(&mut h);
                Ok(Value::Int(h.finish() as i64))
            }
            None => {
                let t = v.type_name();
                interp.type_err(format!("unhashable type: {t}"), args.span)
            }
        }
    });
    def(
        interp,
        &["vars", "locals", "globals", "dir"],
        |interp, _args| {
            let env = interp
                .call_env
                .clone()
                .unwrap_or_else(|| interp.globals.clone());
            let mut d = Dict::new();
            for (k, v) in env.local_bindings() {
                d.insert(Value::Str(k.as_str()), v);
            }
            Ok(Value::Map(Rc::new(RefCell::new(d))))
        },
    );

    // ----- numbers -------------------------------------------------------------
    def(
        interp,
        &["chr", "char", "fromCharCode", "Char"],
        |interp, args| {
            let n = int_arg(interp, &args, 0, "chr")?;
            match char::from_u32(n as u32) {
                Some(c) => Ok(Value::str(&c.to_string())),
                None => interp.value_err(format!("{n} is not a valid character code"), args.span),
            }
        },
    );
    def(
        interp,
        &["ord", "charCodeAt", "codepoint", "Char.code"],
        |interp, args| {
            let s = str_arg(interp, &args, 0, "ord")?;
            match s.chars().next() {
                Some(c) => Ok(Value::Int(c as i64)),
                None => interp.value_err("ord() of an empty string", args.span),
            }
        },
    );
    for (name, radix, prefix) in [("hex", 16u32, "0x"), ("bin", 2, "0b"), ("oct", 8, "0o")] {
        def(interp, &[name], move |interp, args| {
            let v = arg(interp, &args, 0, name)?;
            let s = match &v {
                Value::Int(i) => {
                    let mag = crate::bigint::BigInt::from_i64(*i)
                        .abs()
                        .to_string_radix(radix);
                    format!("{}{prefix}{mag}", if *i < 0 { "-" } else { "" })
                }
                Value::Big(b) => {
                    let mag = b.abs().to_string_radix(radix);
                    format!("{}{prefix}{mag}", if b.is_negative() { "-" } else { "" })
                }
                other => {
                    let t = other.type_name();
                    return interp
                        .type_err(format!("{name}() needs an integer, got {t}"), args.span);
                }
            };
            Ok(Value::str(&s))
        });
    }
    def(interp, &["divmod"], |interp, args| {
        let a = arg(interp, &args, 0, "divmod")?;
        let b = arg(interp, &args, 1, "divmod")?;
        let span = args.span;
        let q = interp.binary_op(crate::ast::BinOp::FloorDiv, a.clone(), b.clone(), span)?;
        let r = interp.binary_op(crate::ast::BinOp::Mod, a, b, span)?;
        Ok(Value::tuple(vec![q, r]))
    });
    def(interp, &["succ", "inc", "next_int"], |interp, args| {
        let a = arg(interp, &args, 0, "succ")?;
        interp.binary_op(crate::ast::BinOp::Add, a, Value::Int(1), args.span)
    });
    def(interp, &["pred", "dec"], |interp, args| {
        let a = arg(interp, &args, 0, "pred")?;
        interp.binary_op(crate::ast::BinOp::Sub, a, Value::Int(1), args.span)
    });
    for (names, pred) in [
        (&["is_even", "even", "even?"][..], 0u8),
        (&["is_odd", "odd", "odd?"], 1),
        (&["is_positive", "positive?"], 2),
        (&["is_negative", "negative?"], 3),
        (&["is_zero", "zero?"], 4),
        (&["is_prime", "prime?", "isPrime"], 5),
    ] {
        def(interp, names, move |interp, args| {
            let v = arg(interp, &args, 0, "predicate")?;
            let Some(x) = v.as_f64() else {
                let t = v.type_name();
                return interp.type_err(format!("expected a number, got {t}"), args.span);
            };
            Ok(Value::Bool(match pred {
                0 => x.rem_euclid(2.0) == 0.0,
                1 => x.rem_euclid(2.0) == 1.0,
                2 => x > 0.0,
                3 => x < 0.0,
                4 => x == 0.0,
                _ => {
                    let n = x as i64;
                    if n < 2 || x.fract() != 0.0 {
                        false
                    } else {
                        let mut i = 2i64;
                        let mut prime = true;
                        while i * i <= n {
                            if n % i == 0 {
                                prime = false;
                                break;
                            }
                            i += 1;
                        }
                        prime
                    }
                }
            }))
        });
    }
    def(
        interp,
        &["is_empty", "empty?", "is_blank", "isEmpty"],
        |interp, args| {
            let v = arg(interp, &args, 0, "is_empty")?;
            Ok(Value::Bool(match &v {
                Value::Str(s) => s.trim().is_empty(),
                Value::Nil => true,
                other => length_of(interp, other, args.span)
                    .map(|n| n == 0)
                    .unwrap_or(false),
            }))
        },
    );

    // ----- sequences -------------------------------------------------------------
    def(
        interp,
        &["enumerate", "each_with_index", "with_index", "ipairs_list"],
        |interp, args| {
            let v = arg(interp, &args, 0, "enumerate")?;
            let start = args.arg(1, "start").and_then(|s| s.as_int()).unwrap_or(0);
            let items = interp.collect(&v, args.span)?;
            Ok(Value::list(
                items
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| Value::tuple(vec![Value::Int(i as i64 + start), x]))
                    .collect(),
            ))
        },
    );
    def(interp, &["ipairs"], |interp, args| {
        let v = arg(interp, &args, 0, "ipairs")?;
        let items = interp.collect(&v, args.span)?;
        Ok(Value::list(
            items
                .into_iter()
                .enumerate()
                .map(|(i, x)| Value::tuple(vec![Value::Int(i as i64 + 1), x]))
                .collect(),
        ))
    });
    def(
        interp,
        &["pairs", "items", "entries", "__items"],
        |interp, args| {
            let v = arg(interp, &args, 0, "pairs")?;
            match &v {
                Value::Map(m) => Ok(Value::list(
                    m.borrow()
                        .iter()
                        .map(|(k, v)| Value::tuple(vec![k.clone(), v.clone()]))
                        .collect(),
                )),
                Value::Instance(i) => Ok(Value::list(
                    i.fields
                        .borrow()
                        .iter()
                        .map(|(k, v)| Value::tuple(vec![Value::Str(k.as_str()), v.clone()]))
                        .collect(),
                )),
                other => {
                    let items = interp.collect(other, args.span)?;
                    Ok(Value::list(
                        items
                            .into_iter()
                            .enumerate()
                            .map(|(i, x)| Value::tuple(vec![Value::Int(i as i64), x]))
                            .collect(),
                    ))
                }
            }
        },
    );
    def(interp, &["__go_range_keys"], |interp, args| {
        let v = arg(interp, &args, 0, "range")?;
        match &v {
            Value::Map(m) => Ok(Value::list(m.borrow().keys())),
            Value::Int(n) => Ok(Value::Range(Rc::new(RangeVal {
                start: 0,
                end: *n,
                step: 1,
            }))),
            other => {
                let n = length_of(interp, other, args.span)?;
                Ok(Value::Range(Rc::new(RangeVal {
                    start: 0,
                    end: n as i64,
                    step: 1,
                })))
            }
        }
    });
    def(interp, &["__go_range_pairs"], |interp, args| {
        let v = arg(interp, &args, 0, "range")?;
        let f = interp
            .builtins
            .get(&sym("pairs"))
            .cloned()
            .unwrap_or(Value::Nil);
        interp.call_value(f, Args::new(vec![v], args.span))
    });
    def(interp, &["keys"], |interp, args| {
        let v = arg(interp, &args, 0, "keys")?;
        let span = args.span;
        interp.call_method(&v, sym("keys"), Args::new(Vec::new(), span), span)
    });
    def(interp, &["values"], |interp, args| {
        let v = arg(interp, &args, 0, "values")?;
        let span = args.span;
        interp.call_method(&v, sym("values"), Args::new(Vec::new(), span), span)
    });
    def(interp, &["zip", "zipWith"], |interp, args| {
        // zipWith(f, xs, ys) when the first argument is a function.
        let (f, lists) = match args.pos.first() {
            Some(v) if v.is_callable() => (Some(v.clone()), &args.pos[1..]),
            _ => (None, &args.pos[..]),
        };
        let mut seqs = Vec::with_capacity(lists.len());
        for l in lists {
            seqs.push(interp.collect(l, args.span)?);
        }
        let n = seqs.iter().map(Vec::len).min().unwrap_or(0);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let row: Vec<Value> = seqs.iter().map(|s| s[i].clone()).collect();
            out.push(match &f {
                Some(f) => call(interp, f, row, args.span)?,
                None => Value::tuple(row),
            });
        }
        Ok(Value::list(out))
    });
    def(
        interp,
        &["map", "array_map", "fmap", "collect_map", "mapcar"],
        |interp, args| {
            let (f, xs) = fn_and_seq(interp, &args, "map")?;
            let items = interp.collect(&xs, args.span)?;
            let mut out = Vec::with_capacity(items.len());
            for (i, x) in items.into_iter().enumerate() {
                out.push(call_flex(
                    interp,
                    &f,
                    vec![x, Value::Int(i as i64)],
                    args.span,
                )?);
            }
            Ok(Value::list(out))
        },
    );
    def(
        interp,
        &["filter", "array_filter", "select_where", "keep"],
        |interp, args| {
            let (f, xs) = fn_and_seq(interp, &args, "filter")?;
            let items = interp.collect(&xs, args.span)?;
            let mut out = Vec::new();
            for (i, x) in items.into_iter().enumerate() {
                if call_flex(interp, &f, vec![x.clone(), Value::Int(i as i64)], args.span)?.truthy()
                {
                    out.push(x);
                }
            }
            Ok(Value::list(out))
        },
    );
    def(
        interp,
        &["reject", "remove_if", "filterfalse"],
        |interp, args| {
            let (f, xs) = fn_and_seq(interp, &args, "reject")?;
            let items = interp.collect(&xs, args.span)?;
            let mut out = Vec::new();
            for x in items {
                if !call(interp, &f, vec![x.clone()], args.span)?.truthy() {
                    out.push(x);
                }
            }
            Ok(Value::list(out))
        },
    );
    def(
        interp,
        &[
            "reduce",
            "fold",
            "foldl",
            "inject",
            "fold_left",
            "aggregate",
        ],
        |interp, args| {
            // reduce(f, xs[, init]) | reduce(xs, f[, init]) | fold(xs, init, f)
            let mut pos = args.pos.clone();
            let fi = pos.iter().position(Value::is_callable);
            let Some(fi) = fi else {
                return interp.type_err("reduce needs a function", args.span);
            };
            let f = pos.remove(fi);
            let xs = if pos.is_empty() {
                Value::list(Vec::new())
            } else {
                pos.remove(0)
            };
            let mut items = interp.collect(&xs, args.span)?.into_iter();
            let mut acc = match pos.pop().or_else(|| args.named("initial")) {
                Some(init) => init,
                None => match items.next() {
                    Some(x) => x,
                    None => {
                        return interp.type_err(
                            "reduce of an empty sequence with no initial value",
                            args.span,
                        );
                    }
                },
            };
            for x in items {
                acc = call(interp, &f, vec![acc, x], args.span)?;
            }
            Ok(acc)
        },
    );
    def(
        interp,
        &["foldr", "fold_right", "reduceRight"],
        |interp, args| {
            let mut pos = args.pos.clone();
            let Some(fi) = pos.iter().position(Value::is_callable) else {
                return interp.type_err("foldr needs a function", args.span);
            };
            let f = pos.remove(fi);
            if pos.len() < 2 {
                return interp.type_err("foldr needs a list and an initial value", args.span);
            }
            // Accept both (f, init, xs) and (xs, init, f).
            let (init, xs) = if matches!(pos[0], Value::List(_) | Value::Tuple(_) | Value::Range(_))
            {
                (pos[1].clone(), pos[0].clone())
            } else {
                (pos[0].clone(), pos[1].clone())
            };
            let items = interp.collect(&xs, args.span)?;
            let mut acc = init;
            for x in items.into_iter().rev() {
                acc = call(interp, &f, vec![x, acc], args.span)?;
            }
            Ok(acc)
        },
    );
    def(interp, &["sum", "array_sum", "total"], |interp, args| {
        let xs = arg(interp, &args, 0, "sum")?;
        let items = if args.pos.len() > 1 && args.pos.iter().all(Value::is_number) {
            args.pos.clone()
        } else {
            interp.collect(&xs, args.span)?
        };
        let mut acc = args
            .arg(1, "start")
            .filter(|_| args.pos.len() <= 2 && !args.pos.iter().all(Value::is_number))
            .unwrap_or(Value::Int(0));
        for x in items {
            acc = interp.binary_op(crate::ast::BinOp::Add, acc, x, args.span)?;
        }
        Ok(acc)
    });
    def(interp, &["product", "prod"], |interp, args| {
        let xs = arg(interp, &args, 0, "product")?;
        let items = interp.collect(&xs, args.span)?;
        let mut acc = Value::Int(1);
        for x in items {
            acc = interp.binary_op(crate::ast::BinOp::Mul, acc, x, args.span)?;
        }
        Ok(acc)
    });
    for (names, want_max) in [
        (&["max", "maximum", "Max"][..], true),
        (&["min", "minimum", "Min"], false),
    ] {
        def(interp, names, move |interp, args| {
            let items = if args.pos.len() == 1 {
                interp.collect(&args.pos[0].clone(), args.span)?
            } else {
                args.pos.clone()
            };
            let key = args.named("key");
            if items.is_empty() {
                return match args.named("default") {
                    Some(d) => Ok(d),
                    None => interp.value_err(
                        format!(
                            "{}() of an empty sequence",
                            if want_max { "max" } else { "min" }
                        ),
                        args.span,
                    ),
                };
            }
            let mut best = items[0].clone();
            let mut best_key = match &key {
                Some(f) => call(interp, f, vec![best.clone()], args.span)?,
                None => best.clone(),
            };
            for x in items.into_iter().skip(1) {
                let k = match &key {
                    Some(f) => call(interp, f, vec![x.clone()], args.span)?,
                    None => x.clone(),
                };
                let o = interp.compare_values(&k, &best_key, args.span)?;
                if (want_max && o == Ordering::Greater) || (!want_max && o == Ordering::Less) {
                    best = x;
                    best_key = k;
                }
            }
            Ok(best)
        });
    }
    def(
        interp,
        &["sorted", "sort", "sort_by", "sortBy", "sortWith"],
        |interp, args| {
            let xs = arg(interp, &args, 0, "sorted")?;
            let items = interp.collect(&xs, args.span)?;
            let func = args
                .named("key")
                .or_else(|| args.named("cmp"))
                .or_else(|| args.get(1).cloned());
            let reverse = args.named("reverse").is_some_and(|r| r.truthy());
            let sorted = sort_values(interp, items, func.as_ref(), reverse, args.span)?;
            Ok(Value::list(sorted))
        },
    );
    def(interp, &["reversed", "reverse", "rev"], |interp, args| {
        let xs = arg(interp, &args, 0, "reversed")?;
        if let Value::Str(s) = &xs {
            return Ok(Value::str(&s.chars().rev().collect::<String>()));
        }
        let mut items = interp.collect(&xs, args.span)?;
        items.reverse();
        Ok(Value::list(items))
    });
    def(interp, &["any", "some", "exists"], |interp, args| {
        let xs = arg(interp, &args, 0, "any")?;
        let items = interp.collect(&xs, args.span)?;
        for x in items {
            let ok = match args.get(1) {
                Some(f) => call(interp, f, vec![x], args.span)?.truthy(),
                None => x.truthy(),
            };
            if ok {
                return Ok(Value::Bool(true));
            }
        }
        Ok(Value::Bool(false))
    });
    def(interp, &["all", "every", "forall"], |interp, args| {
        let xs = arg(interp, &args, 0, "all")?;
        let items = interp.collect(&xs, args.span)?;
        for x in items {
            let ok = match args.get(1) {
                Some(f) => call(interp, f, vec![x], args.span)?.truthy(),
                None => x.truthy(),
            };
            if !ok {
                return Ok(Value::Bool(false));
            }
        }
        Ok(Value::Bool(true))
    });
    def(interp, &["head", "first", "car", "fst"], |interp, args| {
        let xs = arg(interp, &args, 0, "head")?;
        if let Value::Tuple(t) = &xs {
            return Ok(t.first().cloned().unwrap_or(Value::Nil));
        }
        let items = interp.collect(&xs, args.span)?;
        Ok(items.into_iter().next().unwrap_or(Value::Nil))
    });
    def(interp, &["snd", "second"], |interp, args| {
        let xs = arg(interp, &args, 0, "snd")?;
        let items = interp.collect(&xs, args.span)?;
        Ok(items.into_iter().nth(1).unwrap_or(Value::Nil))
    });
    def(interp, &["tail", "cdr", "rest"], |interp, args| {
        let xs = arg(interp, &args, 0, "tail")?;
        let items = interp.collect(&xs, args.span)?;
        Ok(Value::list(items.into_iter().skip(1).collect()))
    });
    def(interp, &["last", "end_of"], |interp, args| {
        let xs = arg(interp, &args, 0, "last")?;
        let items = interp.collect(&xs, args.span)?;
        Ok(items.into_iter().last().unwrap_or(Value::Nil))
    });
    def(interp, &["init", "butlast"], |interp, args| {
        let xs = arg(interp, &args, 0, "init")?;
        let mut items = interp.collect(&xs, args.span)?;
        items.pop();
        Ok(Value::list(items))
    });
    def(interp, &["take", "take_first"], |interp, args| {
        let (a, b) = (
            arg(interp, &args, 0, "take")?,
            arg(interp, &args, 1, "take")?,
        );
        let (n, xs) = if a.as_int().is_some() && !matches!(a, Value::List(_)) {
            (a, b)
        } else {
            (b, a)
        };
        let n = n.as_int().unwrap_or(0).max(0) as usize;
        // Lazy, so `take(5, iterate(f, x))` works on infinite generators.
        let mut it = interp.iterate(&xs, args.span)?;
        let mut items = Vec::with_capacity(n.min(1 << 16));
        while items.len() < n {
            match it.next(interp)? {
                Some(x) => items.push(x),
                None => break,
            }
        }
        Ok(Value::list(items))
    });
    def(interp, &["drop", "skip"], |interp, args| {
        let (a, b) = (
            arg(interp, &args, 0, "drop")?,
            arg(interp, &args, 1, "drop")?,
        );
        let (n, xs) = if a.as_int().is_some() && !matches!(a, Value::List(_)) {
            (a, b)
        } else {
            (b, a)
        };
        let n = n.as_int().unwrap_or(0).max(0) as usize;
        let items = interp.collect(&xs, args.span)?;
        Ok(Value::list(items.into_iter().skip(n).collect()))
    });
    def(interp, &["takeWhile", "take_while"], |interp, args| {
        let (f, xs) = fn_and_seq(interp, &args, "takeWhile")?;
        let mut out = Vec::new();
        let mut it = interp.iterate(&xs, args.span)?;
        while let Some(x) = it.next(interp)? {
            if !call(interp, &f, vec![x.clone()], args.span)?.truthy() {
                break;
            }
            out.push(x);
        }
        Ok(Value::list(out))
    });
    def(interp, &["dropWhile", "drop_while"], |interp, args| {
        let (f, xs) = fn_and_seq(interp, &args, "dropWhile")?;
        let items = interp.collect(&xs, args.span)?;
        let mut i = 0;
        while i < items.len() && call(interp, &f, vec![items[i].clone()], args.span)?.truthy() {
            i += 1;
        }
        Ok(Value::list(items[i..].to_vec()))
    });
    def(
        interp,
        &["concat", "flatten", "concatMap_identity", "flat"],
        |interp, args| {
            let xs = arg(interp, &args, 0, "flatten")?;
            if args.pos.len() > 1 && args.pos.iter().all(|v| matches!(v, Value::List(_))) {
                let mut out = Vec::new();
                for l in &args.pos {
                    out.extend(interp.collect(l, args.span)?);
                }
                return Ok(Value::list(out));
            }
            let mut out = Vec::new();
            for item in interp.collect(&xs, args.span)? {
                match item {
                    Value::List(_) | Value::Tuple(_) => {
                        out.extend(interp.collect(&item, args.span)?)
                    }
                    other => out.push(other),
                }
            }
            Ok(Value::list(out))
        },
    );
    def(
        interp,
        &["concatMap", "flat_map", "flatMap", "mapcat"],
        |interp, args| {
            let (f, xs) = fn_and_seq(interp, &args, "flatMap")?;
            let mut out = Vec::new();
            for x in interp.collect(&xs, args.span)? {
                let r = call(interp, &f, vec![x], args.span)?;
                match r {
                    Value::List(_) | Value::Tuple(_) => out.extend(interp.collect(&r, args.span)?),
                    other => out.push(other),
                }
            }
            Ok(Value::list(out))
        },
    );
    def(
        interp,
        &["replicate", "repeat_value", "array_fill"],
        |interp, args| {
            let (a, b) = (
                arg(interp, &args, 0, "replicate")?,
                arg(interp, &args, 1, "replicate")?,
            );
            let (n, v) = if a.as_int().is_some() { (a, b) } else { (b, a) };
            let n = n.as_int().unwrap_or(0).clamp(0, 50_000_000) as usize;
            Ok(Value::list(vec![v; n]))
        },
    );
    def(
        interp,
        &["elem", "contains", "in_array", "includes", "member"],
        |interp, args| {
            let (a, b) = (
                arg(interp, &args, 0, "elem")?,
                arg(interp, &args, 1, "elem")?,
            );
            // elem(x, xs) (Haskell/PHP) or contains(xs, x).
            let (item, coll) = if matches!(
                b,
                Value::List(_) | Value::Tuple(_) | Value::Set(_) | Value::Map(_) | Value::Range(_)
            ) && !matches!(a, Value::List(_) | Value::Map(_) | Value::Set(_))
            {
                (a, b)
            } else {
                (b, a)
            };
            Ok(Value::Bool(interp.contains(&coll, &item, args.span)?))
        },
    );
    def(
        interp,
        &["unique", "uniq", "distinct", "nub", "array_unique", "dedup"],
        |interp, args| {
            let xs = arg(interp, &args, 0, "unique")?;
            let items = interp.collect(&xs, args.span)?;
            let mut out: Vec<Value> = Vec::new();
            for x in items {
                let mut seen = false;
                for o in &out {
                    if interp.values_equal(o, &x)? {
                        seen = true;
                        break;
                    }
                }
                if !seen {
                    out.push(x);
                }
            }
            Ok(Value::list(out))
        },
    );
    def(
        interp,
        &["group_by", "groupBy", "groupingBy"],
        |interp, args| {
            let (f, xs) = fn_and_seq(interp, &args, "group_by")?;
            let mut d = Dict::new();
            for x in interp.collect(&xs, args.span)? {
                let k = call(interp, &f, vec![x.clone()], args.span)?;
                match d.get(&k) {
                    Some(Value::List(l)) => l.borrow_mut().push(x),
                    _ => {
                        d.insert(k, Value::list(vec![x]));
                    }
                }
            }
            Ok(Value::Map(Rc::new(RefCell::new(d))))
        },
    );
    def(interp, &["partition"], |interp, args| {
        let (f, xs) = fn_and_seq(interp, &args, "partition")?;
        let (mut yes, mut no) = (Vec::new(), Vec::new());
        for x in interp.collect(&xs, args.span)? {
            if call(interp, &f, vec![x.clone()], args.span)?.truthy() {
                yes.push(x);
            } else {
                no.push(x);
            }
        }
        Ok(Value::tuple(vec![Value::list(yes), Value::list(no)]))
    });
    def(
        interp,
        &["chunk", "chunks", "each_slice", "chunked", "batch"],
        |interp, args| {
            let xs = arg(interp, &args, 0, "chunks")?;
            let n = int_arg(interp, &args, 1, "chunks")?.max(1) as usize;
            let items = interp.collect(&xs, args.span)?;
            Ok(Value::list(
                items.chunks(n).map(|c| Value::list(c.to_vec())).collect(),
            ))
        },
    );
    def(
        interp,
        &["windows", "sliding", "each_cons", "windowed"],
        |interp, args| {
            let xs = arg(interp, &args, 0, "windows")?;
            let n = int_arg(interp, &args, 1, "windows")?.max(1) as usize;
            let items = interp.collect(&xs, args.span)?;
            Ok(Value::list(
                items.windows(n).map(|c| Value::list(c.to_vec())).collect(),
            ))
        },
    );
    def(
        interp,
        &[
            "frequencies",
            "counter",
            "Counter",
            "tally",
            "count_by",
            "value_counts",
        ],
        |interp, args| {
            let xs = arg(interp, &args, 0, "frequencies")?;
            let mut d = Dict::new();
            for x in interp.collect(&xs, args.span)? {
                let n = d.get(&x).and_then(|v| v.as_int()).unwrap_or(0);
                d.insert(x, Value::Int(n + 1));
            }
            Ok(Value::Map(Rc::new(RefCell::new(d))))
        },
    );
    def(interp, &["words"], |interp, args| {
        let s = str_arg(interp, &args, 0, "words")?;
        Ok(Value::list(s.split_whitespace().map(Value::str).collect()))
    });
    def(interp, &["unwords"], |interp, args| {
        let xs = arg(interp, &args, 0, "unwords")?;
        let items = interp.collect(&xs, args.span)?;
        let s = join_display(interp, &items, " ")?;
        Ok(Value::str(&s))
    });
    def(interp, &["lines"], |interp, args| {
        let s = str_arg(interp, &args, 0, "lines")?;
        Ok(Value::list(s.lines().map(Value::str).collect()))
    });
    def(interp, &["unlines"], |interp, args| {
        let xs = arg(interp, &args, 0, "unlines")?;
        let items = interp.collect(&xs, args.span)?;
        let s = join_display(interp, &items, "\n")?;
        Ok(Value::str(&format!("{s}\n")))
    });
    def(
        interp,
        &["implode", "join_list", "str_join"],
        |interp, args| {
            // implode(glue, pieces) (PHP) or join(list, sep).
            let (a, b) = (
                arg(interp, &args, 0, "implode")?,
                arg(interp, &args, 1, "implode")?,
            );
            let (sep, xs) = if matches!(a, Value::Str(_)) {
                (a, b)
            } else {
                (b, a)
            };
            let sep = interp.to_str(&sep)?;
            let items = interp.collect(&xs, args.span)?;
            let s = join_display(interp, &items, &sep)?;
            Ok(Value::str(&s))
        },
    );
    def(interp, &["explode"], |interp, args| {
        let sep = str_arg(interp, &args, 0, "explode")?;
        let s = str_arg(interp, &args, 1, "explode")?;
        Ok(Value::list(s.split(&*sep).map(Value::str).collect()))
    });
    def(
        interp,
        &["append", "array_push", "conj", "push"],
        |interp, args| {
            // Go `append(xs, a, b)` returns a new slice; PHP array_push mutates.
            let xs = arg(interp, &args, 0, "append")?;
            let Value::List(l) = &xs else {
                return interp.type_err("append needs a list", args.span);
            };
            let mut items = l.borrow().clone();
            items.extend(args.pos[1..].iter().cloned());
            Ok(Value::list(items))
        },
    );
    def(interp, &["make"], |interp, args| {
        // Go `make([]int, n)` / `make(map[string]int)`.
        let n = args
            .get(1)
            .and_then(Value::as_int)
            .unwrap_or(0)
            .clamp(0, 50_000_000) as usize;
        match args.get(0) {
            Some(Value::Type(TypeTag::Map)) => Ok(Value::Map(Rc::new(RefCell::new(Dict::new())))),
            Some(Value::Type(TypeTag::Int)) => Ok(Value::list(vec![Value::Int(0); n])),
            _ => {
                let _ = interp;
                Ok(Value::list(vec![Value::Nil; n]))
            }
        }
    });
    def(interp, &["__array_of"], |_interp, args| {
        let n = args
            .get(0)
            .and_then(Value::as_int)
            .unwrap_or(0)
            .clamp(0, 50_000_000) as usize;
        let fill = args.get(1).cloned().unwrap_or(Value::Int(0));
        Ok(Value::list(vec![fill; n]))
    });
    // Ruby's parenthesis-free calls: `puts greet` calls `greet` when it is
    // a named function needing no arguments.
    def(interp, &["__autocall"], |interp, args| {
        let v = args.get(0).cloned().unwrap_or(Value::Nil);
        match &v {
            Value::Func(c)
                if c.def.name.is_some() && c.required_params() == 0 && !c.def.is_generator() =>
            {
                let span = args.span;
                interp.call_value(v.clone(), Args::new(Vec::new(), span))
            }
            _ => Ok(v),
        }
    });
    // Haskell sections: `(* 3)` and `(2 ^)`.
    for (name, right) in [("__section_right", true), ("__section_left", false)] {
        def(interp, &[name], move |interp, args| {
            let op = arg(interp, &args, 0, "section")?;
            let operand = arg(interp, &args, 1, "section")?;
            let op = match &op {
                Value::Str(s) if &**s == "^" => Value::str("**"),
                other => other.clone(),
            };
            Ok(Value::native("section", move |interp, args| {
                let x = args.get(0).cloned().unwrap_or(Value::Nil);
                let pair = if right {
                    vec![x, operand.clone()]
                } else {
                    vec![operand.clone(), x]
                };
                interp.call_value(op.clone(), Args::new(pair, args.span))
            }))
        });
    }
    // Go's comma-ok: `v, ok := m[key]`.
    def(interp, &["__comma_ok"], |interp, args| {
        let obj = arg(interp, &args, 0, "comma-ok")?;
        let key = arg(interp, &args, 1, "comma-ok")?;
        Ok(match &obj {
            Value::Map(m) => {
                let v = m.borrow().get(&key);
                let found = v.is_some();
                Value::tuple(vec![v.unwrap_or(Value::Nil), Value::Bool(found)])
            }
            _ => {
                let span = args.span;
                match interp.index_value(&obj, &key, span) {
                    Ok(v) => Value::tuple(vec![v, Value::Bool(true)]),
                    Err(Flow::Throw(_)) => Value::tuple(vec![Value::Nil, Value::Bool(false)]),
                    Err(e) => return Err(e),
                }
            }
        })
    });

    // ----- strings --------------------------------------------------------------
    def(
        interp,
        &["strtoupper", "toupper", "uppercase", "upcase_str"],
        |interp, args| {
            let s = str_arg(interp, &args, 0, "upper")?;
            Ok(Value::str(&s.to_uppercase()))
        },
    );
    def(
        interp,
        &["strtolower", "tolower", "lowercase"],
        |interp, args| {
            let s = str_arg(interp, &args, 0, "lower")?;
            Ok(Value::str(&s.to_lowercase()))
        },
    );
    def(interp, &["str_replace"], |interp, args| {
        let from = str_arg(interp, &args, 0, "str_replace")?;
        let to = str_arg(interp, &args, 1, "str_replace")?;
        let s = str_arg(interp, &args, 2, "str_replace")?;
        Ok(Value::str(&s.replace(&*from, &to)))
    });
    def(interp, &["str_repeat"], |interp, args| {
        let s = str_arg(interp, &args, 0, "str_repeat")?;
        let n = int_arg(interp, &args, 1, "str_repeat")?.clamp(0, 10_000_000) as usize;
        Ok(Value::str(&s.repeat(n)))
    });
    def(interp, &["ucfirst", "capitalize_str"], |interp, args| {
        let s = str_arg(interp, &args, 0, "ucfirst")?;
        let mut c = s.chars();
        Ok(Value::str(&match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => String::new(),
        }))
    });

    // ----- files -------------------------------------------------------------------
    def(
        interp,
        &[
            "read_file",
            "readFile",
            "file_get_contents",
            "slurp",
            "File.read",
        ],
        |interp, args| {
            let path = str_arg(interp, &args, 0, "read_file")?;
            let full = interp.base_dir.join(&*path);
            match std::fs::read_to_string(&full) {
                Ok(s) => Ok(Value::str(&s)),
                Err(e) => {
                    Err(interp.error("IOError", format!("cannot read `{path}`: {e}"), args.span))
                }
            }
        },
    );
    def(
        interp,
        &["write_file", "writeFile", "file_put_contents", "spit"],
        |interp, args| {
            let path = str_arg(interp, &args, 0, "write_file")?;
            let content = arg(interp, &args, 1, "write_file")?;
            let s = interp.to_str(&content)?;
            let full = interp.base_dir.join(&*path);
            match std::fs::write(&full, s) {
                Ok(()) => Ok(Value::Nil),
                Err(e) => {
                    Err(interp.error("IOError", format!("cannot write `{path}`: {e}"), args.span))
                }
            }
        },
    );
    def(
        interp,
        &["file_exists", "exists_file", "path_exists"],
        |interp, args| {
            let path = str_arg(interp, &args, 0, "file_exists")?;
            Ok(Value::Bool(interp.base_dir.join(&*path).exists()))
        },
    );
    def(interp, &["open", "fopen", "File.open"], |interp, args| {
        let path = str_arg(interp, &args, 0, "open")?;
        let mode = match args.arg(1, "mode") {
            Some(Value::Str(m)) => m.to_string(),
            _ => "r".into(),
        };
        super::objects::open_file(interp, &path, &mode, args.span)
    });
    def(interp, &["__close"], |interp, args| {
        let v = arg(interp, &args, 0, "__close")?;
        let span = args.span;
        for name in ["__exit__", "close", "dispose", "Dispose"] {
            if interp
                .try_call_method(&v, sym(name), Args::new(Vec::new(), span), span)?
                .is_some()
            {
                break;
            }
        }
        Ok(Value::Nil)
    });

    // ----- glue used by desugarings ------------------------------------------------------
    def(interp, &["__english_add"], |interp, args| {
        let target = arg(interp, &args, 0, "add")?;
        let value = arg(interp, &args, 1, "add")?;
        if let Value::List(l) = &target {
            l.borrow_mut().push(value);
            return Ok(target);
        }
        interp.binary_op(crate::ast::BinOp::Add, target, value, args.span)
    });
    def(interp, &["__variant_named"], |interp, args| {
        let name = str_arg(interp, &args, 0, "variant")?;
        Ok(Value::Str(name))
    });
    def(interp, &["__like"], |interp, args| {
        let s = arg(interp, &args, 0, "like")?;
        let pat = str_arg(interp, &args, 1, "like")?;
        let s = interp.to_str(&s)?;
        Ok(Value::Bool(sql_like(&s, &pat)))
    });
    def(interp, &["__identity", "identity_fn"], |_interp, args| {
        Ok(args.pos.first().cloned().unwrap_or(Value::Nil))
    });
}

/// `len()` of anything sized.
pub fn length_of(interp: &mut Interp, v: &Value, span: Span) -> R<usize> {
    Ok(match v {
        Value::Str(s) => s.chars().count(),
        Value::List(l) => l.borrow().len(),
        Value::Tuple(t) => t.len(),
        Value::Map(m) | Value::Set(m) => m.borrow().len(),
        Value::Range(r) => r.len(),
        Value::Tensor(t) => t.shape().first().copied().unwrap_or(1),
        Value::Instance(_) | Value::Object(_) => {
            for name in ["__len__", "len", "length", "size", "count"] {
                if let Some(r) =
                    interp.try_call_method(v, sym(name), Args::new(Vec::new(), span), span)?
                {
                    return Ok(r.as_int().unwrap_or(0).max(0) as usize);
                }
            }
            return interp.type_err("this object has no length (define `__len__`)", span);
        }
        Value::Nil => 0,
        other => {
            let t = other.type_name();
            return interp.type_err(format!("`{t}` has no length"), span);
        }
    })
}

/// SQL `LIKE` with `%` (any run) and `_` (any one character).
pub fn sql_like(s: &str, pat: &str) -> bool {
    let s: Vec<char> = s.chars().collect();
    let p: Vec<char> = pat.chars().collect();
    let (mut i, mut j) = (0usize, 0usize);
    let (mut star, mut mark) = (None, 0usize);
    while i < s.len() {
        if j < p.len() && (p[j] == '_' || p[j].eq_ignore_ascii_case(&s[i])) {
            i += 1;
            j += 1;
        } else if j < p.len() && p[j] == '%' {
            star = Some(j);
            mark = i;
            j += 1;
        } else if let Some(st) = star {
            j = st + 1;
            mark += 1;
            i = mark;
        } else {
            return false;
        }
    }
    while j < p.len() && p[j] == '%' {
        j += 1;
    }
    j == p.len()
}

#[cfg(test)]
mod tests {
    use super::sql_like;

    #[test]
    fn like_patterns() {
        assert!(sql_like("Alice", "A%"));
        assert!(sql_like("Alice", "%ice"));
        assert!(sql_like("Alice", "_lice"));
        assert!(sql_like("alice", "AL%"));
        assert!(!sql_like("Bob", "A%"));
        assert!(sql_like("", "%"));
        assert!(!sql_like("abc", "a_"));
    }
}
