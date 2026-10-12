//! Importable built-in modules (`import math`, `from itertools import
//! permutations`, `import "fmt"`, …).

use std::cell::RefCell;
use std::rc::Rc;

use super::objects::{builtin, module};
use super::{arg, call, int_arg, str_arg};
use crate::interp::{Args, Interp, sym};
use crate::symbol::Sym;
use crate::value::*;

fn aliases(interp: &Interp, pairs: &[(&'static str, &str)]) -> Vec<(&'static str, Value)> {
    pairs
        .iter()
        .map(|(k, t)| (*k, builtin(interp, t)))
        .collect()
}

/// Returns the built-in module called `name`, if there is one.
pub fn builtin_module(interp: &mut Interp, name: &str) -> Option<Value> {
    if let Some(m) = crate::ml::module(interp, name) {
        return Some(m);
    }
    if let Some(m) = crate::ai::module(interp, name) {
        return Some(m);
    }
    Some(match name {
        "math" | "cmath" | "Math" => {
            let mut members = aliases(
                interp,
                &[
                    ("sqrt", "sqrt"),
                    ("isqrt", "isqrt"),
                    ("exp", "exp"),
                    ("log", "log"),
                    ("log2", "log2"),
                    ("log10", "log10"),
                    ("log1p", "log1p"),
                    ("sin", "sin"),
                    ("cos", "cos"),
                    ("tan", "tan"),
                    ("asin", "asin"),
                    ("acos", "acos"),
                    ("atan", "atan"),
                    ("atan2", "atan2"),
                    ("sinh", "sinh"),
                    ("cosh", "cosh"),
                    ("tanh", "tanh"),
                    ("floor", "floor"),
                    ("ceil", "ceil"),
                    ("trunc", "trunc"),
                    ("fabs", "abs"),
                    ("pow", "pow"),
                    ("hypot", "hypot"),
                    ("gcd", "gcd"),
                    ("lcm", "lcm"),
                    ("factorial", "factorial"),
                    ("comb", "comb"),
                    ("perm", "perm"),
                    ("isclose", "isclose"),
                    ("radians", "radians"),
                    ("degrees", "degrees"),
                    ("pi", "pi"),
                    ("e", "e"),
                    ("tau", "tau"),
                    ("inf", "inf"),
                    ("nan", "nan"),
                    ("isnan", "isNaN"),
                    ("prod", "product"),
                    ("fsum", "sum"),
                    ("copysign", "sign"),
                    ("cbrt", "cbrt"),
                ],
            );
            members.push((
                "isqrt",
                Value::native("isqrt", |interp, args| {
                    let n = int_arg(interp, &args, 0, "isqrt")?;
                    if n < 0 {
                        return interp.value_err("isqrt of a negative number", args.span);
                    }
                    let mut x = (n as f64).sqrt() as i64;
                    while x * x > n {
                        x -= 1;
                    }
                    while (x + 1) * (x + 1) <= n {
                        x += 1;
                    }
                    Ok(Value::Int(x))
                }),
            ));
            members.push((
                "isinf",
                Value::native("isinf", |_i, args| {
                    Ok(Value::Bool(
                        args.get(0)
                            .and_then(Value::as_f64)
                            .is_some_and(f64::is_infinite),
                    ))
                }),
            ));
            module("math", members)
        }
        "random" | "rand" => module(
            "random",
            aliases(
                interp,
                &[
                    ("random", "random"),
                    ("randint", "randint"),
                    ("randrange", "randint"),
                    ("uniform", "uniform"),
                    ("gauss", "gauss"),
                    ("normalvariate", "gauss"),
                    ("choice", "choice"),
                    ("shuffle", "shuffle"),
                    ("sample", "sample"),
                    ("seed", "seed"),
                    ("Intn", "randint"),
                    ("Float64", "random"),
                ],
            ),
        ),
        "json" | "JSON" | "encoding/json" | "simplejson" => super::json::module(interp),
        "time" | "datetime" | "chrono" => module(
            "time",
            aliases(
                interp,
                &[
                    ("time", "clock"),
                    ("sleep", "sleep"),
                    ("perf_counter", "clock"),
                    ("monotonic", "clock"),
                    ("now", "clock"),
                    ("Now", "clock"),
                    ("Sleep", "sleep"),
                ],
            ),
        ),
        "os" | "os.path" | "path" => {
            let environ = {
                let mut d = Dict::new();
                for (k, v) in std::env::vars() {
                    d.insert(Value::str(&k), Value::str(&v));
                }
                Value::Map(Rc::new(RefCell::new(d)))
            };
            let path = module(
                "path",
                vec![
                    (
                        "exists",
                        Value::native("exists", |interp, args| {
                            let p = str_arg(interp, &args, 0, "exists")?;
                            Ok(Value::Bool(interp.base_dir.join(&*p).exists()))
                        }),
                    ),
                    (
                        "join",
                        Value::native("join", |interp, args| {
                            let mut p = std::path::PathBuf::new();
                            for a in &args.pos {
                                p.push(interp.to_str(a)?);
                            }
                            Ok(Value::str(&p.to_string_lossy()))
                        }),
                    ),
                    (
                        "basename",
                        Value::native("basename", |interp, args| {
                            let p = str_arg(interp, &args, 0, "basename")?;
                            let b = std::path::Path::new(&*p)
                                .file_name()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_default();
                            Ok(Value::str(&b))
                        }),
                    ),
                    (
                        "dirname",
                        Value::native("dirname", |interp, args| {
                            let p = str_arg(interp, &args, 0, "dirname")?;
                            let d = std::path::Path::new(&*p)
                                .parent()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_default();
                            Ok(Value::str(&d))
                        }),
                    ),
                ],
            );
            if name != "os" {
                return Some(path);
            }
            module(
                "os",
                vec![
                    ("environ", environ),
                    ("path", path),
                    (
                        "getenv",
                        Value::native("getenv", |interp, args| {
                            let k = str_arg(interp, &args, 0, "getenv")?;
                            Ok(std::env::var(&*k).map_or_else(
                                |_| args.get(1).cloned().unwrap_or(Value::Nil),
                                |v| Value::str(&v),
                            ))
                        }),
                    ),
                    (
                        "getcwd",
                        Value::native("getcwd", |interp, _args| {
                            Ok(Value::str(&interp.base_dir.to_string_lossy()))
                        }),
                    ),
                    (
                        "listdir",
                        Value::native("listdir", |interp, args| {
                            let p = match args.get(0) {
                                Some(v) => interp.to_str(v)?,
                                None => ".".into(),
                            };
                            let dir = interp.base_dir.join(p);
                            let mut names: Vec<String> = match std::fs::read_dir(&dir) {
                                Ok(rd) => rd
                                    .filter_map(|e| {
                                        e.ok().map(|e| e.file_name().to_string_lossy().to_string())
                                    })
                                    .collect(),
                                Err(e) => {
                                    return Err(interp.error(
                                        "IOError",
                                        format!("cannot list `{}`: {e}", dir.display()),
                                        args.span,
                                    ));
                                }
                            };
                            names.sort();
                            Ok(Value::list(names.iter().map(|n| Value::str(n)).collect()))
                        }),
                    ),
                    ("exit", builtin(interp, "exit")),
                    ("name", Value::str(std::env::consts::FAMILY)),
                ],
            )
        }
        "sys" => module(
            "sys",
            vec![
                (
                    "argv",
                    interp
                        .builtins
                        .get(&sym("__argv"))
                        .cloned()
                        .unwrap_or_else(|| Value::list(Vec::new())),
                ),
                ("exit", builtin(interp, "exit")),
                (
                    "version",
                    Value::str(concat!("Chimera ", env!("CARGO_PKG_VERSION"))),
                ),
                ("platform", Value::str(std::env::consts::OS)),
                ("maxsize", Value::Int(i64::MAX)),
                ("stdout", builtin(interp, "cout")),
                ("stderr", builtin(interp, "cerr")),
                (
                    "setrecursionlimit",
                    Value::native("setrecursionlimit", |interp, args| {
                        let n = int_arg(interp, &args, 0, "setrecursionlimit")?;
                        interp.max_depth = n.clamp(10, 1_000_000) as usize;
                        Ok(Value::Nil)
                    }),
                ),
            ],
        ),
        "string" => module(
            "string",
            vec![
                (
                    "ascii_letters",
                    Value::str("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                ),
                ("ascii_lowercase", Value::str("abcdefghijklmnopqrstuvwxyz")),
                ("ascii_uppercase", Value::str("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
                ("digits", Value::str("0123456789")),
                ("hexdigits", Value::str("0123456789abcdefABCDEF")),
                (
                    "punctuation",
                    Value::str("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"),
                ),
                ("whitespace", Value::str(" \t\n\r\x0b\x0c")),
            ],
        ),
        "statistics" | "stats" => module(
            "statistics",
            aliases(
                interp,
                &[
                    ("mean", "mean"),
                    ("median", "median"),
                    ("mode", "mode"),
                    ("stdev", "stdev"),
                    ("variance", "variance"),
                    ("pstdev", "stdev"),
                    ("pvariance", "variance"),
                    ("fmean", "mean"),
                ],
            ),
        ),
        "itertools" => {
            let mut members = aliases(
                interp,
                &[
                    ("chain", "concat"),
                    ("accumulate_list", "reduce"),
                    ("zip_longest", "zip"),
                ],
            );
            members.push((
                "permutations",
                Value::native("permutations", |interp, args| {
                    let xs = arg(interp, &args, 0, "permutations")?;
                    let items = interp.collect(&xs, args.span)?;
                    let r = args
                        .get(1)
                        .and_then(Value::as_int)
                        .map_or(items.len(), |r| r.max(0) as usize);
                    let mut out = Vec::new();
                    permute(
                        &items,
                        r,
                        &mut Vec::new(),
                        &mut vec![false; items.len()],
                        &mut out,
                    );
                    Ok(Value::list(out))
                }),
            ));
            members.push((
                "combinations",
                Value::native("combinations", |interp, args| {
                    let xs = arg(interp, &args, 0, "combinations")?;
                    let items = interp.collect(&xs, args.span)?;
                    let r = int_arg(interp, &args, 1, "combinations")?.max(0) as usize;
                    let mut out = Vec::new();
                    combine(&items, r, 0, &mut Vec::new(), &mut out);
                    Ok(Value::list(out))
                }),
            ));
            members.push((
                "product",
                Value::native("product", |interp, args| {
                    let repeat = args
                        .named("repeat")
                        .and_then(|v| v.as_int())
                        .unwrap_or(1)
                        .max(1) as usize;
                    let mut pools = Vec::new();
                    for _ in 0..repeat {
                        for p in &args.pos {
                            pools.push(interp.collect(p, args.span)?);
                        }
                    }
                    let mut out: Vec<Vec<Value>> = vec![Vec::new()];
                    for pool in pools {
                        let mut next = Vec::new();
                        for prefix in &out {
                            for x in &pool {
                                let mut v = prefix.clone();
                                v.push(x.clone());
                                next.push(v);
                            }
                        }
                        out = next;
                        if out.len() > 10_000_000 {
                            return interp.value_err("product is too large", args.span);
                        }
                    }
                    Ok(Value::list(out.into_iter().map(Value::tuple).collect()))
                }),
            ));
            members.push((
                "accumulate",
                Value::native("accumulate", |interp, args| {
                    let xs = arg(interp, &args, 0, "accumulate")?;
                    let f = args.get(1).cloned();
                    let mut acc: Option<Value> = None;
                    let mut out = Vec::new();
                    for x in interp.collect(&xs, args.span)? {
                        let next = match (&acc, &f) {
                            (None, _) => x,
                            (Some(a), Some(f)) => call(interp, f, vec![a.clone(), x], args.span)?,
                            (Some(a), None) => {
                                interp.binary_op(crate::ast::BinOp::Add, a.clone(), x, args.span)?
                            }
                        };
                        out.push(next.clone());
                        acc = Some(next);
                    }
                    Ok(Value::list(out))
                }),
            ));
            members.push((
                "islice",
                Value::native("islice", |interp, args| {
                    let xs = arg(interp, &args, 0, "islice")?;
                    let items = interp.collect(&xs, args.span)?;
                    let (start, stop) = match (
                        args.get(1).and_then(Value::as_int),
                        args.get(2).and_then(Value::as_int),
                    ) {
                        (Some(a), Some(b)) => (a, b),
                        (Some(b), None) => (0, b),
                        _ => (0, items.len() as i64),
                    };
                    Ok(Value::list(
                        items
                            .into_iter()
                            .skip(start.max(0) as usize)
                            .take((stop - start).max(0) as usize)
                            .collect(),
                    ))
                }),
            ));
            module("itertools", members)
        }
        "functools" => {
            let memo = builtin(interp, "memoize");
            let mut members = aliases(interp, &[("reduce", "reduce"), ("partial", "partial")]);
            members.push((
                "lru_cache",
                Value::native("lru_cache", move |interp, args| {
                    // Both `@lru_cache` and `@lru_cache(maxsize=…)`.
                    match args.get(0) {
                        Some(f) if f.is_callable() => {
                            call(interp, &memo.clone(), vec![f.clone()], args.span)
                        }
                        _ => Ok(memo.clone()),
                    }
                }),
            ));
            members.push(("cache", builtin(interp, "memoize")));
            members.push((
                "wraps",
                Value::native("wraps", |_i, _a| {
                    Ok(Value::native("wraps", |_i, a| {
                        Ok(a.pos.first().cloned().unwrap_or(Value::Nil))
                    }))
                }),
            ));
            module("functools", members)
        }
        "collections" => {
            let namedtuple = Value::native("namedtuple", |interp, args| {
                let name = str_arg(interp, &args, 0, "namedtuple")?;
                let fields = arg(interp, &args, 1, "namedtuple")?;
                let names: Vec<Sym> = match &fields {
                    Value::Str(s) => s
                        .split([',', ' '])
                        .filter(|f| !f.is_empty())
                        .map(sym)
                        .collect(),
                    other => {
                        let mut v = Vec::new();
                        for f in interp.collect(other, args.span)? {
                            v.push(sym(&interp.to_str(&f)?));
                        }
                        v
                    }
                };
                let mut class = Class::new(&name, None);
                class.field_names = names;
                Ok(Value::Class(Rc::new(class)))
            });
            module(
                "collections",
                vec![
                    ("Counter", builtin(interp, "frequencies")),
                    ("OrderedDict", Value::Type(TypeTag::Map)),
                    ("deque", Value::Type(TypeTag::List)),
                    ("namedtuple", namedtuple),
                    ("defaultdict", builtin(interp, "defaultdict")),
                ],
            )
        }
        "operator" => {
            let bin = |op: crate::ast::BinOp, n: &str| {
                Value::native(n, move |interp, args| {
                    let a = arg(interp, &args, 0, "operator")?;
                    let b = arg(interp, &args, 1, "operator")?;
                    interp.binary_op(op, a, b, args.span)
                })
            };
            use crate::ast::BinOp as B;
            module(
                "operator",
                vec![
                    ("add", bin(B::Add, "add")),
                    ("sub", bin(B::Sub, "sub")),
                    ("mul", bin(B::Mul, "mul")),
                    ("truediv", bin(B::Div, "truediv")),
                    ("floordiv", bin(B::FloorDiv, "floordiv")),
                    ("mod", bin(B::Mod, "mod")),
                    ("pow", bin(B::Pow, "pow")),
                    (
                        "itemgetter",
                        Value::native("itemgetter", |_i, args| {
                            let keys = args.pos.clone();
                            Ok(Value::native("itemgetter", move |interp, a| {
                                let obj = arg(interp, &a, 0, "itemgetter")?;
                                if keys.len() == 1 {
                                    return interp.index_value(&obj, &keys[0], a.span);
                                }
                                let mut out = Vec::new();
                                for k in &keys {
                                    out.push(interp.index_value(&obj, k, a.span)?);
                                }
                                Ok(Value::tuple(out))
                            }))
                        }),
                    ),
                    (
                        "attrgetter",
                        Value::native("attrgetter", |interp, args| {
                            let name = str_arg(interp, &args, 0, "attrgetter")?;
                            Ok(Value::native("attrgetter", move |interp, a| {
                                let obj = arg(interp, &a, 0, "attrgetter")?;
                                interp.get_attr(&obj, sym(&name), a.span)
                            }))
                        }),
                    ),
                ],
            )
        }
        "fmt" => builtin(interp, "fmt"),
        "strings" => module(
            "strings",
            vec![
                ("ToUpper", builtin(interp, "strtoupper")),
                ("ToLower", builtin(interp, "strtolower")),
                ("Repeat", builtin(interp, "str_repeat")),
                ("Fields", builtin(interp, "words")),
                (
                    "Join",
                    Value::native("strings.Join", |interp, args| {
                        let xs = arg(interp, &args, 0, "Join")?;
                        let sep = str_arg(interp, &args, 1, "Join")?;
                        let items = interp.collect(&xs, args.span)?;
                        let mut parts = Vec::new();
                        for x in &items {
                            parts.push(interp.to_str(x)?);
                        }
                        Ok(Value::str(&parts.join(&sep)))
                    }),
                ),
                ("Split", method_fn("split")),
                ("Contains", method_fn("contains")),
                ("HasPrefix", method_fn("startswith")),
                ("HasSuffix", method_fn("endswith")),
                ("TrimSpace", method_fn("strip")),
                ("Replace", method_fn("replace")),
                ("ReplaceAll", method_fn("replace")),
                ("Index", method_fn("find")),
            ],
        ),
        "strconv" => module(
            "strconv",
            aliases(
                interp,
                &[
                    ("Itoa", "tostring"),
                    ("Atoi", "tonumber"),
                    ("ParseFloat", "parseFloat"),
                    ("FormatInt", "tostring"),
                ],
            ),
        ),
        "typing" | "dataclasses" | "abc" | "enum" | "__future__" => module(name, Vec::new()),
        _ => return None,
    })
}

/// A function `f(recv, args…)` that calls `recv.method(args…)`.
fn method_fn(method: &'static str) -> Value {
    Value::native(method, move |interp, mut args: Args| {
        if args.pos.is_empty() {
            return interp.type_err(format!("`{method}` needs a receiver"), args.span);
        }
        let recv = args.pos.remove(0);
        let span = args.span;
        interp.call_method(&recv, sym(method), args, span)
    })
}

fn permute(
    items: &[Value],
    r: usize,
    cur: &mut Vec<Value>,
    used: &mut Vec<bool>,
    out: &mut Vec<Value>,
) {
    if cur.len() == r {
        out.push(Value::tuple(cur.clone()));
        return;
    }
    for i in 0..items.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        cur.push(items[i].clone());
        permute(items, r, cur, used, out);
        cur.pop();
        used[i] = false;
        if out.len() > 5_000_000 {
            return;
        }
    }
}

fn combine(items: &[Value], r: usize, start: usize, cur: &mut Vec<Value>, out: &mut Vec<Value>) {
    if cur.len() == r {
        out.push(Value::tuple(cur.clone()));
        return;
    }
    for i in start..items.len() {
        cur.push(items[i].clone());
        combine(items, r, i + 1, cur, out);
        cur.pop();
        if out.len() > 5_000_000 {
            return;
        }
    }
}
