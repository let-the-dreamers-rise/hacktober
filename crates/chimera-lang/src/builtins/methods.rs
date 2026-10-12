//! Methods of built-in types, under the names Python, JavaScript, Ruby,
//! Java, Kotlin, Swift and Rust programmers expect.

use std::cell::RefCell;
use std::rc::Rc;

use chimera_core::Span;

use super::core::{length_of, sort_values};
use super::{call, call_flex};
use crate::interp::{Args, Interp, sym};
use crate::value::*;

/// The extension-method key for a value's type.
pub fn tag_of(v: &Value) -> Option<TypeTag> {
    Some(match v {
        Value::Str(_) => TypeTag::Str,
        Value::Int(_) | Value::Big(_) => TypeTag::Int,
        Value::Float(_) => TypeTag::Float,
        Value::Bool(_) => TypeTag::Bool,
        Value::List(_) => TypeTag::List,
        Value::Tuple(_) => TypeTag::Tuple,
        Value::Map(_) => TypeTag::Map,
        Value::Set(_) => TypeTag::Set,
        Value::Range(_) => TypeTag::Range,
        Value::Tensor(_) => TypeTag::Tensor,
        Value::Nil => TypeTag::Nil,
        _ => return None,
    })
}

const STR_METHODS: &[&str] = &[
    "upper",
    "lower",
    "toUpperCase",
    "toLowerCase",
    "upcase",
    "downcase",
    "capitalize",
    "title",
    "strip",
    "trim",
    "lstrip",
    "rstrip",
    "trimStart",
    "trimEnd",
    "split",
    "join",
    "replace",
    "replaceAll",
    "gsub",
    "sub",
    "startswith",
    "endswith",
    "startsWith",
    "endsWith",
    "start_with?",
    "end_with?",
    "starts_with",
    "ends_with",
    "find",
    "indexOf",
    "index",
    "rfind",
    "lastIndexOf",
    "contains",
    "includes",
    "include?",
    "count",
    "len",
    "length",
    "size",
    "chars",
    "bytes",
    "lines",
    "words",
    "reverse",
    "reversed",
    "repeat",
    "center",
    "ljust",
    "rjust",
    "padStart",
    "padEnd",
    "zfill",
    "isdigit",
    "isalpha",
    "isalnum",
    "isspace",
    "isupper",
    "islower",
    "is_empty",
    "isEmpty",
    "empty?",
    "format",
    "to_i",
    "to_int",
    "to_f",
    "to_float",
    "to_s",
    "toString",
    "to_string",
    "to_upper",
    "to_lower",
    "substring",
    "substr",
    "slice",
    "charAt",
    "char_at",
    "at",
    "first",
    "last",
    "split_whitespace",
    "partition",
    "encode",
    "swapcase",
    "casefold",
    "removeprefix",
    "removesuffix",
    "matches",
    "equals",
    "equalsIgnoreCase",
    "compareTo",
    "hash",
    "ord",
    "parse",
    "to_list",
    "concat",
    "splitlines",
    "expandtabs",
    "strip_prefix",
    "strip_suffix",
    "sorted",
    "unique",
];

const LIST_METHODS: &[&str] = &[
    "append",
    "push",
    "add",
    "push_back",
    "pop",
    "pop_back",
    "shift",
    "unshift",
    "prepend",
    "insert",
    "remove",
    "extend",
    "concat",
    "len",
    "length",
    "size",
    "count",
    "index",
    "indexOf",
    "find_index",
    "findIndex",
    "contains",
    "includes",
    "include?",
    "map",
    "collect",
    "filter",
    "select",
    "where",
    "reject",
    "reduce",
    "fold",
    "inject",
    "forEach",
    "each",
    "for_each",
    "each_with_index",
    "sort",
    "sorted",
    "sort_by",
    "sortBy",
    "reverse",
    "reversed",
    "join",
    "slice",
    "first",
    "last",
    "min",
    "max",
    "sum",
    "product",
    "any",
    "some",
    "all",
    "every",
    "find",
    "detect",
    "flat",
    "flatten",
    "flatMap",
    "flat_map",
    "unique",
    "uniq",
    "distinct",
    "group_by",
    "groupBy",
    "partition",
    "zip",
    "enumerate",
    "take",
    "drop",
    "chunk",
    "chunks",
    "copy",
    "clone",
    "clear",
    "is_empty",
    "isEmpty",
    "empty?",
    "to_list",
    "to_a",
    "toList",
    "tolist",
    "to_set",
    "iter",
    "into_iter",
    "average",
    "mean",
    "max_by",
    "min_by",
    "tally",
    "count_by",
    "each_slice",
    "shuffle",
    "sample",
    "to_s",
    "toString",
    "get",
    "set",
    "fill",
    "lastIndexOf",
    "splice",
    "remove_at",
    "removeAt",
    "delete",
    "delete_at",
    "compact",
    "rotate",
    "swap",
    "to_tensor",
    "len",
    "keys",
    "windows",
    "sum_by",
    "none",
    "cycle",
];

const MAP_METHODS: &[&str] = &[
    "keys",
    "values",
    "items",
    "entries",
    "get",
    "set",
    "put",
    "has",
    "has_key",
    "has_key?",
    "key?",
    "containsKey",
    "contains_key",
    "includes",
    "contains",
    "remove",
    "delete",
    "pop",
    "update",
    "merge",
    "len",
    "length",
    "size",
    "clear",
    "copy",
    "clone",
    "setdefault",
    "get_or_insert",
    "getOrDefault",
    "each",
    "forEach",
    "for_each",
    "map",
    "filter",
    "select",
    "to_list",
    "to_a",
    "is_empty",
    "isEmpty",
    "empty?",
    "invert",
    "count",
    "any",
    "all",
    "find",
    "sort_by",
    "sorted",
    "min_by",
    "max_by",
    "sum",
    "to_s",
    "toString",
    "fetch",
    "values_at",
    "dig",
    "transform_values",
    "map_values",
    "reduce",
];

const NUM_METHODS: &[&str] = &[
    "abs",
    "round",
    "floor",
    "ceil",
    "sqrt",
    "to_s",
    "toString",
    "to_string",
    "to_i",
    "to_f",
    "toFixed",
    "times",
    "upto",
    "downto",
    "pow",
    "is_integer",
    "is_even",
    "even?",
    "odd?",
    "is_odd",
    "zero?",
    "clamp",
    "min",
    "max",
    "sign",
    "signum",
    "bit_length",
    "digits",
    "gcd",
    "lcm",
    "real",
    "imag",
    "conjugate",
    "succ",
    "pred",
    "chr",
    "trunc",
    "is_nan",
    "is_finite",
    "toPrecision",
    "format",
    "between?",
    "hex",
];

/// Names of the methods a value supports (for "did you mean" hints).
pub fn method_names(v: &Value) -> &'static [&'static str] {
    match v {
        Value::Str(_) => STR_METHODS,
        Value::List(_) | Value::Tuple(_) | Value::Range(_) | Value::Set(_) => LIST_METHODS,
        Value::Map(_) => MAP_METHODS,
        Value::Int(_) | Value::Float(_) | Value::Big(_) => NUM_METHODS,
        _ => &[],
    }
}

/// Whether `v` has a built-in method called `name`.
pub fn has_method(v: &Value, name: &str) -> bool {
    method_names(v).contains(&name) || GENERIC.contains(&name)
}

const GENERIC: &[&str] = &[
    "unwrap",
    "unwrap_or",
    "expect",
    "is_some",
    "is_none",
    "is_nil",
    "nil?",
    "to_s",
    "inspect",
    "then",
    "also",
    "let",
    "apply",
    "tap",
    "class",
    "type",
    "equals",
    "hashCode",
    "is_a?",
    "instance_of?",
    "respond_to?",
    "dup",
];

/// Properties read without a call: `xs.length`, `s.size`, `t.shape`, …
pub fn property(interp: &mut Interp, v: &Value, name: &str, span: Span) -> R<Option<Value>> {
    if let Value::Tensor(t) = v {
        return crate::ml::tensor_property(interp, t, name, span);
    }
    let sized = matches!(
        v,
        Value::Str(_)
            | Value::List(_)
            | Value::Tuple(_)
            | Value::Map(_)
            | Value::Set(_)
            | Value::Range(_)
    );
    Ok(match name {
        "length" | "size" | "len" | "count" | "Count" | "Length" if sized => {
            Some(Value::Int(length_of(interp, v, span)? as i64))
        }
        "first" | "last" | "is_empty" | "isEmpty" | "empty?" | "head" | "tail" | "keys"
        | "values" | "items" | "entries" | "sum" | "min" | "max" | "reverse" | "reversed"
        | "sorted" | "upper" | "lower" | "upcase" | "downcase" | "strip" | "trim" | "chars"
        | "abs" | "floor" | "ceil" | "round" | "to_s" | "to_i" | "to_f" | "capitalize" | "uniq"
        | "flatten" | "compact" | "lines" | "words" | "average" | "mean" | "negate"
            if sized || v.is_number() =>
        {
            call_named(interp, v, name, Vec::new(), span)?
        }
        "real" if v.is_number() => Some(v.clone()),
        "imag" if v.is_number() => Some(Value::Int(0)),
        "start" | "end" | "step" => match v {
            Value::Range(r) => Some(Value::Int(match name {
                "start" => r.start,
                "end" => r.end,
                _ => r.step,
            })),
            _ => None,
        },
        _ => None,
    })
}

fn call_named(
    interp: &mut Interp,
    v: &Value,
    name: &str,
    args: Vec<Value>,
    span: Span,
) -> R<Option<Value>> {
    dispatch(interp, v, name, Args::new(args, span))
}

/// Calls a built-in method; `None` if the type has no such method.
pub fn dispatch(interp: &mut Interp, v: &Value, name: &str, args: Args) -> R<Option<Value>> {
    call_impl(interp, v, name, args)
}

fn call_impl(interp: &mut Interp, v: &Value, name: &str, args: Args) -> R<Option<Value>> {
    let r = match v {
        Value::Str(s) => str_method(interp, s, name, &args)?,
        Value::List(l) => list_method(interp, v, l, name, &args)?,
        Value::Tuple(t) => {
            let as_list = Value::list(t.to_vec());
            match name {
                "count" | "index" | "len" | "length" | "size" | "contains" | "includes"
                | "first" | "last" | "to_list" | "to_a" | "sum" | "min" | "max" | "map"
                | "filter" | "join" => {
                    let Value::List(l) = &as_list else {
                        unreachable!()
                    };
                    list_method(interp, &as_list, l, name, &args)?
                }
                _ => None,
            }
        }
        Value::Range(_) | Value::Set(_) => {
            let items = interp.collect(v, args.span)?;
            match (v, name) {
                (Value::Set(s), "add" | "insert") => {
                    for a in &args.pos {
                        s.borrow_mut().insert(a.clone(), Value::Nil);
                    }
                    Some(Value::Nil)
                }
                (Value::Set(s), "remove" | "discard" | "delete") => {
                    let a = args.get(0).cloned().unwrap_or(Value::Nil);
                    let had = s.borrow_mut().remove(&a).is_some();
                    if !had && name == "remove" {
                        let k = crate::interp::format::repr_plain(&a);
                        return Err(interp.error("KeyError", k, args.span));
                    }
                    Some(Value::Nil)
                }
                (
                    Value::Set(s),
                    "union"
                    | "intersection"
                    | "difference"
                    | "symmetric_difference"
                    | "issubset"
                    | "issuperset"
                    | "isdisjoint",
                ) => {
                    let other = args.get(0).cloned().unwrap_or(Value::Nil);
                    let other_items = interp.collect(&other, args.span)?;
                    let mut o = Dict::new();
                    for x in other_items {
                        o.insert(x, Value::Nil);
                    }
                    let mine = s.borrow().clone();
                    let mut d = Dict::new();
                    match name {
                        "union" => {
                            for k in mine.keys().into_iter().chain(o.keys()) {
                                d.insert(k, Value::Nil);
                            }
                        }
                        "intersection" => {
                            for k in mine.keys() {
                                if o.contains(&k) {
                                    d.insert(k, Value::Nil);
                                }
                            }
                        }
                        "difference" => {
                            for k in mine.keys() {
                                if !o.contains(&k) {
                                    d.insert(k, Value::Nil);
                                }
                            }
                        }
                        "symmetric_difference" => {
                            for k in mine.keys() {
                                if !o.contains(&k) {
                                    d.insert(k, Value::Nil);
                                }
                            }
                            for k in o.keys() {
                                if !mine.contains(&k) {
                                    d.insert(k, Value::Nil);
                                }
                            }
                        }
                        "issubset" => {
                            return Ok(Some(Value::Bool(mine.keys().iter().all(|k| o.contains(k)))));
                        }
                        "issuperset" => {
                            return Ok(Some(Value::Bool(o.keys().iter().all(|k| mine.contains(k)))));
                        }
                        _ => {
                            return Ok(Some(Value::Bool(
                                !mine.keys().iter().any(|k| o.contains(k)),
                            )));
                        }
                    }
                    Some(Value::Set(Rc::new(RefCell::new(d))))
                }
                _ => {
                    let list = Value::list(items);
                    let Value::List(l) = &list else {
                        unreachable!()
                    };
                    list_method(interp, &list, l, name, &args)?
                }
            }
        }
        Value::Map(m) => map_method(interp, v, m, name, &args)?,
        Value::Int(_) | Value::Float(_) | Value::Big(_) | Value::Bool(_) => {
            num_method(interp, v, name, &args)?
        }
        Value::Tensor(t) => crate::ml::tensor_method(interp, t, name, &args)?,
        _ => None,
    };
    if r.is_some() {
        return Ok(r);
    }
    generic_method(interp, v, name, &args)
}

/// Methods every value has (Option-style helpers, Kotlin scope functions…).
fn generic_method(interp: &mut Interp, v: &Value, name: &str, args: &Args) -> R<Option<Value>> {
    let span = args.span;
    Ok(Some(match name {
        "unwrap" | "expect" | "get_or_throw" => {
            if matches!(v, Value::Nil) {
                let msg = match args.get(0) {
                    Some(m) => interp.to_str(m)?,
                    None => "called unwrap() on nil".into(),
                };
                return Err(interp.error("ValueError", msg, span));
            }
            v.clone()
        }
        "unwrap_or" | "or_else" | "getOrElse" | "orElse" | "unwrap_or_default" => {
            if matches!(v, Value::Nil) {
                match args.get(0) {
                    Some(d) if d.is_callable() && name == "or_else" => {
                        super::call(interp, d, Vec::new(), span)?
                    }
                    Some(d) => d.clone(),
                    None => Value::Nil,
                }
            } else {
                v.clone()
            }
        }
        "is_some" | "isPresent" | "has_value" => Value::Bool(!matches!(v, Value::Nil)),
        "is_none" | "is_nil" | "nil?" | "isEmpty_opt" | "is_null" => {
            Value::Bool(matches!(v, Value::Nil))
        }
        "to_s" | "toString" | "to_string" | "description" | "str" => {
            let s = interp.to_str(v)?;
            Value::str(&s)
        }
        "inspect" | "repr" | "debug" => {
            let s = interp.repr(v)?;
            Value::str(&s)
        }
        "then" | "let" | "map_opt" | "pipe" | "into" => {
            if matches!(v, Value::Nil) && name == "let" {
                return Ok(Some(Value::Nil));
            }
            let f = super::arg(interp, args, 0, name)?;
            super::call(interp, &f, vec![v.clone()], span)?
        }
        "also" | "tap" | "apply" => {
            let f = super::arg(interp, args, 0, name)?;
            super::call(interp, &f, vec![v.clone()], span)?;
            v.clone()
        }
        "class" | "type" | "getClass" | "__class__" => super::type_of(v),
        "equals" | "eql?" | "equal?" => {
            let o = super::arg(interp, args, 0, name)?;
            Value::Bool(interp.values_equal(v, &o)?)
        }
        "is_a?" | "kind_of?" | "instance_of?" | "isinstance" => {
            let t = super::arg(interp, args, 0, name)?;
            Value::Bool(interp.isinstance(v, &t))
        }
        "respond_to?" => {
            let n = super::str_arg(interp, args, 0, name)?;
            Value::Bool(has_method(v, &n) || interp.get_attr(v, sym(&n), span).is_ok())
        }
        "dup" | "clone" | "copy" | "deepcopy" | "deep_copy" => deep_copy(v),
        "hashCode" | "hash" => {
            let f = interp
                .builtins
                .get(&sym("hash"))
                .cloned()
                .unwrap_or(Value::Nil);
            interp.call_value(f, Args::new(vec![v.clone()], span))?
        }
        _ => return Ok(None),
    }))
}

/// Methods on class instances that are not user-defined.
pub fn object_method(interp: &mut Interp, v: &Value, name: &str, args: Args) -> R<Option<Value>> {
    generic_method(interp, v, name, &args)
}

/// A deep copy of collections (instances are copied one level deep).
pub fn deep_copy(v: &Value) -> Value {
    match v {
        Value::List(l) => Value::list(l.borrow().iter().map(deep_copy).collect()),
        Value::Map(m) => {
            let mut d = Dict::new();
            for (k, x) in m.borrow().iter() {
                d.insert(k.clone(), deep_copy(x));
            }
            Value::Map(Rc::new(RefCell::new(d)))
        }
        Value::Set(m) => Value::Set(Rc::new(RefCell::new(m.borrow().clone()))),
        Value::Tuple(t) => Value::tuple(t.iter().map(deep_copy).collect()),
        Value::Instance(i) => Value::Instance(Rc::new(Instance {
            class: i.class.clone(),
            fields: RefCell::new(
                i.fields
                    .borrow()
                    .iter()
                    .map(|(k, x)| (*k, deep_copy(x)))
                    .collect(),
            ),
        })),
        Value::Tensor(t) => Value::Tensor(t.detach()),
        other => other.clone(),
    }
}

fn arg_str(interp: &mut Interp, args: &Args, i: usize, name: &str) -> R<String> {
    let v = super::arg(interp, args, i, name)?;
    interp.to_str(&v)
}

fn char_index_to_byte(s: &str, i: usize) -> usize {
    s.char_indices().nth(i).map_or(s.len(), |(b, _)| b)
}

fn str_method(interp: &mut Interp, s: &Rc<str>, name: &str, args: &Args) -> R<Option<Value>> {
    let span = args.span;
    let st: &str = s;
    let out = match name {
        "upper" | "toUpperCase" | "upcase" | "to_upper" | "uppercase" | "to_uppercase"
        | "ToUpper" => Value::str(&st.to_uppercase()),
        "lower" | "toLowerCase" | "downcase" | "to_lower" | "lowercase" | "to_lowercase"
        | "ToLower" | "casefold" => Value::str(&st.to_lowercase()),
        "swapcase" => Value::str(
            &st.chars()
                .map(|c| {
                    if c.is_uppercase() {
                        c.to_lowercase().collect::<String>()
                    } else {
                        c.to_uppercase().collect::<String>()
                    }
                })
                .collect::<String>(),
        ),
        "capitalize" => {
            let mut c = st.chars();
            Value::str(&match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                None => String::new(),
            })
        }
        "title" | "titlecase" => {
            let mut out = String::with_capacity(st.len());
            let mut start = true;
            for c in st.chars() {
                if c.is_alphanumeric() {
                    if start {
                        out.extend(c.to_uppercase());
                    } else {
                        out.extend(c.to_lowercase());
                    }
                    start = false;
                } else {
                    out.push(c);
                    start = true;
                }
            }
            Value::str(&out)
        }
        "strip" | "trim" | "Trim" => match args.get(0) {
            Some(Value::Str(chars)) => Value::str(st.trim_matches(|c| chars.contains(c))),
            _ => Value::str(st.trim()),
        },
        "lstrip" | "trimStart" | "trim_start" | "trimLeft" | "TrimStart" => {
            Value::str(st.trim_start())
        }
        "rstrip" | "trimEnd" | "trim_end" | "trimRight" | "chomp" | "TrimEnd" => {
            match args.get(0) {
                Some(Value::Str(chars)) => Value::str(st.trim_end_matches(|c| chars.contains(c))),
                _ => Value::str(st.trim_end()),
            }
        }
        "split" | "Split" => {
            let limit = args.arg(1, "maxsplit").and_then(|v| v.as_int());
            match args.get(0) {
                None | Some(Value::Nil) => {
                    Value::list(st.split_whitespace().map(Value::str).collect())
                }
                Some(sep) => {
                    let sep = interp.to_str(sep)?;
                    if sep.is_empty() {
                        Value::list(st.chars().map(|c| Value::str(&c.to_string())).collect())
                    } else {
                        let parts: Vec<Value> = match limit {
                            Some(n) if n >= 0 => {
                                st.splitn(n as usize + 1, &*sep).map(Value::str).collect()
                            }
                            _ => st.split(&*sep).map(Value::str).collect(),
                        };
                        Value::list(parts)
                    }
                }
            }
        }
        "split_whitespace" | "words" => {
            Value::list(st.split_whitespace().map(Value::str).collect())
        }
        "lines" | "splitlines" => Value::list(st.lines().map(Value::str).collect()),
        "join" => {
            // Python: ", ".join(xs)
            let xs = super::arg(interp, args, 0, "join")?;
            let items = interp.collect(&xs, span)?;
            let mut parts = Vec::with_capacity(items.len());
            for x in &items {
                parts.push(interp.to_str(x)?);
            }
            Value::str(&parts.join(st))
        }
        "replace" | "replaceAll" | "gsub" | "Replace" | "replace_all" => {
            let from = arg_str(interp, args, 0, name)?;
            let to = arg_str(interp, args, 1, name)?;
            match args.get(2).and_then(Value::as_int) {
                Some(n) => Value::str(&st.replacen(&from, &to, n.max(0) as usize)),
                None => Value::str(&st.replace(&from, &to)),
            }
        }
        "sub" | "replace_first" | "replaceFirst" => {
            let from = arg_str(interp, args, 0, name)?;
            let to = arg_str(interp, args, 1, name)?;
            Value::str(&st.replacen(&from, &to, 1))
        }
        "startswith" | "startsWith" | "start_with?" | "starts_with" | "StartsWith"
        | "hasPrefix" => {
            let p = arg_str(interp, args, 0, name)?;
            Value::Bool(st.starts_with(&p))
        }
        "endswith" | "endsWith" | "end_with?" | "ends_with" | "EndsWith" | "hasSuffix" => {
            let p = arg_str(interp, args, 0, name)?;
            Value::Bool(st.ends_with(&p))
        }
        "removeprefix" | "strip_prefix" | "delete_prefix" => {
            let p = arg_str(interp, args, 0, name)?;
            Value::str(st.strip_prefix(&*p).unwrap_or(st))
        }
        "removesuffix" | "strip_suffix" | "delete_suffix" => {
            let p = arg_str(interp, args, 0, name)?;
            Value::str(st.strip_suffix(&*p).unwrap_or(st))
        }
        "find" | "indexOf" | "index" | "IndexOf" | "find_index" => {
            let p = arg_str(interp, args, 0, name)?;
            match st.find(&p) {
                Some(b) => Value::Int(st[..b].chars().count() as i64),
                None if name == "index" => {
                    return Err(interp.error("ValueError", "substring not found", span));
                }
                None => Value::Int(-1),
            }
        }
        "rfind" | "lastIndexOf" | "rindex" => {
            let p = arg_str(interp, args, 0, name)?;
            match st.rfind(&p) {
                Some(b) => Value::Int(st[..b].chars().count() as i64),
                None => Value::Int(-1),
            }
        }
        "contains" | "includes" | "include?" | "Contains" => {
            let p = arg_str(interp, args, 0, name)?;
            Value::Bool(st.contains(&p))
        }
        "count" if !args.is_empty() => {
            let p = arg_str(interp, args, 0, name)?;
            Value::Int(if p.is_empty() {
                st.chars().count() as i64 + 1
            } else {
                st.matches(&p).count() as i64
            })
        }
        "len" | "length" | "size" | "count" | "Length" | "chars_count" => {
            Value::Int(st.chars().count() as i64)
        }
        "bytes" | "encode" | "as_bytes" | "getBytes" => {
            Value::list(st.bytes().map(|b| Value::Int(i64::from(b))).collect())
        }
        "chars" | "to_list" | "toCharArray" | "split_chars" => {
            Value::list(st.chars().map(|c| Value::str(&c.to_string())).collect())
        }
        "reverse" | "reversed" => Value::str(&st.chars().rev().collect::<String>()),
        "repeat" | "times_str" => {
            let n = super::int_arg(interp, args, 0, name)?.clamp(0, 10_000_000) as usize;
            Value::str(&st.repeat(n))
        }
        "center" | "ljust" | "rjust" | "padStart" | "padEnd" | "PadLeft" | "PadRight" => {
            let width = super::int_arg(interp, args, 0, name)?.max(0) as usize;
            let fill = match args.get(1) {
                Some(f) => interp.to_str(f)?.chars().next().unwrap_or(' '),
                None => ' ',
            };
            let len = st.chars().count();
            if len >= width {
                Value::Str(s.clone())
            } else {
                let n = width - len;
                let f = fill.to_string();
                Value::str(&match name {
                    "center" => format!("{}{st}{}", f.repeat(n / 2), f.repeat(n - n / 2)),
                    "ljust" | "padEnd" | "PadRight" => format!("{st}{}", f.repeat(n)),
                    _ => format!("{}{st}", f.repeat(n)),
                })
            }
        }
        "zfill" => {
            let width = super::int_arg(interp, args, 0, name)?.max(0) as usize;
            let (sign, digits) = if st.starts_with('-') || st.starts_with('+') {
                st.split_at(1)
            } else {
                ("", st)
            };
            let len = digits.chars().count() + sign.len();
            Value::str(&format!(
                "{sign}{}{digits}",
                "0".repeat(width.saturating_sub(len))
            ))
        }
        "isdigit" | "isnumeric" | "isdecimal" | "is_numeric" => {
            Value::Bool(!st.is_empty() && st.chars().all(|c| c.is_ascii_digit()))
        }
        "isalpha" | "is_alphabetic" => {
            Value::Bool(!st.is_empty() && st.chars().all(char::is_alphabetic))
        }
        "isalnum" | "is_alphanumeric" => {
            Value::Bool(!st.is_empty() && st.chars().all(char::is_alphanumeric))
        }
        "isspace" | "is_whitespace" | "isBlank" | "blank?" => {
            Value::Bool(st.chars().all(char::is_whitespace))
        }
        "isupper" => {
            Value::Bool(st.chars().any(char::is_alphabetic) && !st.chars().any(char::is_lowercase))
        }
        "islower" => {
            Value::Bool(st.chars().any(char::is_alphabetic) && !st.chars().any(char::is_uppercase))
        }
        "is_empty" | "isEmpty" | "empty?" | "IsNullOrEmpty" => Value::Bool(st.is_empty()),
        "format" => {
            let env = interp.call_env.clone();
            let r = interp.brace_format(st, &args.pos, &args.named, env.as_ref(), span)?;
            Value::str(&r)
        }
        "to_i" | "to_int" | "toInt" | "parseInt" | "parse_int" | "toInteger" => {
            let base = args.get(0).and_then(Value::as_int).unwrap_or(10) as u32;
            match super::to_int(interp, &Value::Str(s.clone()), base, span) {
                Ok(v) => v,
                Err(_) if name == "to_i" => {
                    // Ruby: leading digits, else 0.
                    let digits: String = st
                        .trim()
                        .chars()
                        .take_while(|c| c.is_ascii_digit() || *c == '-')
                        .collect();
                    Value::Int(digits.parse().unwrap_or(0))
                }
                Err(e) => return Err(e),
            }
        }
        "to_f" | "to_float" | "toDouble" | "toFloat" | "parseFloat" => {
            Value::Float(super::parse_float(st).unwrap_or(0.0))
        }
        "parse" => {
            // Rust: `s.parse::<i32>()` is a Result.
            let (ctor, payload) = if let Ok(i) = st.trim().parse::<i64>() {
                ("Ok", Value::Int(i))
            } else if let Some(f) = super::parse_float(st) {
                ("Ok", Value::Float(f))
            } else if let Ok(b) = st.trim().parse::<bool>() {
                ("Ok", Value::Bool(b))
            } else {
                (
                    "Err",
                    Value::str(&format!(
                        "invalid digit found in string {}",
                        chimera_core::text::escape_debug(st)
                    )),
                )
            };
            let f = interp
                .builtins
                .get(&sym(ctor))
                .cloned()
                .unwrap_or(Value::Nil);
            interp.call_value(f, Args::new(vec![payload], span))?
        }
        "substring" | "substr" | "slice" | "Substring" | "get" => {
            let chars: Vec<char> = st.chars().collect();
            let n = chars.len() as i64;
            let a = args.get(0).and_then(Value::as_int).unwrap_or(0);
            let a = if a < 0 { (n + a).max(0) } else { a.min(n) };
            let b = match (name, args.get(1).and_then(Value::as_int)) {
                // substr(start, length) vs substring(start, end)
                ("substr", Some(len)) | ("Substring", Some(len)) => (a + len).min(n),
                (_, Some(e)) => {
                    if e < 0 {
                        (n + e).max(0)
                    } else {
                        e.min(n)
                    }
                }
                (_, None) => n,
            };
            let (a, b) = if a > b && name == "substring" {
                (b, a)
            } else {
                (a, b.max(a))
            };
            Value::str(&chars[a as usize..b as usize].iter().collect::<String>())
        }
        "charAt" | "char_at" | "at" | "nth" => {
            let chars: Vec<char> = st.chars().collect();
            let i = args.get(0).and_then(Value::as_int).unwrap_or(0);
            let i = if i < 0 { i + chars.len() as i64 } else { i };
            match chars.get(i as usize) {
                Some(c) if i >= 0 => Value::str(&c.to_string()),
                _ => Value::str(""),
            }
        }
        "first" => Value::str(
            &st.chars()
                .take(args.get(0).and_then(Value::as_int).unwrap_or(1).max(0) as usize)
                .collect::<String>(),
        ),
        "last" => {
            let n = args.get(0).and_then(Value::as_int).unwrap_or(1).max(0) as usize;
            let chars: Vec<char> = st.chars().collect();
            Value::str(
                &chars[chars.len().saturating_sub(n)..]
                    .iter()
                    .collect::<String>(),
            )
        }
        "partition" => {
            let sep = arg_str(interp, args, 0, name)?;
            match st.split_once(&*sep) {
                Some((a, b)) => Value::tuple(vec![Value::str(a), Value::str(&sep), Value::str(b)]),
                None => Value::tuple(vec![Value::Str(s.clone()), Value::str(""), Value::str("")]),
            }
        }
        "equalsIgnoreCase" | "eq_ignore_case" => {
            let o = arg_str(interp, args, 0, name)?;
            Value::Bool(st.to_lowercase() == o.to_lowercase())
        }
        "compareTo" | "cmp" | "localeCompare" => {
            let o = arg_str(interp, args, 0, name)?;
            Value::Int(match st.cmp(o.as_str()) {
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            })
        }
        "ord" => Value::Int(st.chars().next().map_or(0, |c| c as i64)),
        "concat" | "plus" => {
            let mut out = st.to_owned();
            for a in &args.pos {
                out.push_str(&interp.to_str(a)?);
            }
            Value::str(&out)
        }
        "matches" | "like" => {
            let p = arg_str(interp, args, 0, name)?;
            Value::Bool(super::core::sql_like(st, &p))
        }
        "expandtabs" => Value::str(&st.replace('\t', "        ")),
        "sorted" => {
            let mut c: Vec<char> = st.chars().collect();
            c.sort_unstable();
            Value::list(c.into_iter().map(|c| Value::str(&c.to_string())).collect())
        }
        "unique" => {
            let mut seen = Vec::new();
            for c in st.chars() {
                if !seen.contains(&c) {
                    seen.push(c);
                }
            }
            Value::str(&seen.into_iter().collect::<String>())
        }
        "each_char" | "forEach" | "each" => {
            let f = super::arg(interp, args, 0, name)?;
            for c in st.chars() {
                call(interp, &f, vec![Value::str(&c.to_string())], span)?;
            }
            Value::Nil
        }
        "map" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut out = String::new();
            for c in st.chars() {
                let r = call(interp, &f, vec![Value::str(&c.to_string())], span)?;
                out.push_str(&interp.to_str(&r)?);
            }
            Value::str(&out)
        }
        _ => return Ok(None),
    };
    Ok(Some(out))
}

fn list_method(
    interp: &mut Interp,
    v: &Value,
    l: &Rc<RefCell<Vec<Value>>>,
    name: &str,
    args: &Args,
) -> R<Option<Value>> {
    let span = args.span;
    let items = || l.borrow().clone();
    let out = match name {
        "append" | "push" | "add" | "push_back" | "Add" | "addLast" | "pushLast" => {
            l.borrow_mut().extend(args.pos.iter().cloned());
            if name == "push" && args.pos.len() == 1 {
                // JS returns the new length; Ruby returns the array.
                Value::Int(l.borrow().len() as i64)
            } else {
                Value::Nil
            }
        }
        "extend" | "concat_in_place" | "addAll" | "append_all" | "AddRange" => {
            let other = super::arg(interp, args, 0, name)?;
            let more = interp.collect(&other, span)?;
            l.borrow_mut().extend(more);
            Value::Nil
        }
        "concat" | "plus" | "chain" => {
            let mut out = items();
            for a in &args.pos {
                match a {
                    Value::List(_) | Value::Tuple(_) => out.extend(interp.collect(a, span)?),
                    other => out.push(other.clone()),
                }
            }
            Value::list(out)
        }
        "pop" | "pop_back" | "removeLast" => {
            let len = l.borrow().len();
            if len == 0 {
                return Err(interp.error("IndexError", "pop from an empty list", span));
            }
            match args.get(0).and_then(Value::as_int) {
                Some(i) => {
                    let idx = interp.list_index(&Value::Int(i), len, span)?;
                    l.borrow_mut().remove(idx)
                }
                None => l.borrow_mut().pop().unwrap_or(Value::Nil),
            }
        }
        "shift" | "pop_front" | "removeFirst" | "popleft" => {
            if l.borrow().is_empty() {
                Value::Nil
            } else {
                l.borrow_mut().remove(0)
            }
        }
        "unshift" | "prepend" | "push_front" | "addFirst" | "appendleft" => {
            let mut b = l.borrow_mut();
            for (i, a) in args.pos.iter().enumerate() {
                b.insert(i, a.clone());
            }
            Value::Int(b.len() as i64)
        }
        "insert" | "Insert" | "add_at" => {
            let i = super::int_arg(interp, args, 0, name)?;
            let x = super::arg(interp, args, 1, name)?;
            let len = l.borrow().len() as i64;
            let i = if i < 0 { (len + i).max(0) } else { i.min(len) } as usize;
            l.borrow_mut().insert(i, x);
            Value::Nil
        }
        "remove" | "Remove" | "delete" => {
            let x = super::arg(interp, args, 0, name)?;
            let current = items();
            let mut found = None;
            for (i, y) in current.iter().enumerate() {
                if interp.values_equal(y, &x)? {
                    found = Some(i);
                    break;
                }
            }
            match found {
                Some(i) => {
                    l.borrow_mut().remove(i);
                    if name == "delete" { x } else { Value::Nil }
                }
                None if name == "remove" => {
                    let r = interp.repr(&x)?;
                    return Err(interp.error(
                        "ValueError",
                        format!("{r} is not in the list"),
                        span,
                    ));
                }
                None => Value::Nil,
            }
        }
        "remove_at" | "removeAt" | "delete_at" | "RemoveAt" => {
            let len = l.borrow().len();
            let i = super::arg(interp, args, 0, name)?;
            let i = interp.list_index(&i, len, span)?;
            l.borrow_mut().remove(i)
        }
        "splice" => {
            let len = l.borrow().len() as i64;
            let start = args.get(0).and_then(Value::as_int).unwrap_or(0);
            let start = if start < 0 {
                (len + start).max(0)
            } else {
                start.min(len)
            } as usize;
            let count = args.get(1).and_then(Value::as_int).unwrap_or(len).max(0) as usize;
            let end = (start + count).min(len as usize);
            let removed: Vec<Value> = l
                .borrow_mut()
                .splice(start..end, args.pos.iter().skip(2).cloned())
                .collect();
            Value::list(removed)
        }
        "clear" | "Clear" => {
            l.borrow_mut().clear();
            Value::Nil
        }
        "len" | "length" | "size" | "Count" => Value::Int(l.borrow().len() as i64),
        "count" => match args.get(0) {
            None => Value::Int(l.borrow().len() as i64),
            Some(f) if f.is_callable() => {
                let mut n = 0;
                for x in items() {
                    if call(interp, f, vec![x], span)?.truthy() {
                        n += 1;
                    }
                }
                Value::Int(n)
            }
            Some(x) => {
                let mut n = 0;
                for y in items() {
                    if interp.values_equal(&y, x)? {
                        n += 1;
                    }
                }
                Value::Int(n)
            }
        },
        "index" | "indexOf" | "find_index" | "findIndex" | "IndexOf" | "position"
        | "lastIndexOf" => {
            let target = super::arg(interp, args, 0, name)?;
            let all = items();
            let order: Box<dyn Iterator<Item = (usize, &Value)>> = if name == "lastIndexOf" {
                Box::new(all.iter().enumerate().rev())
            } else {
                Box::new(all.iter().enumerate())
            };
            let mut found = None;
            for (i, y) in order {
                let hit = if target.is_callable()
                    && matches!(name, "find_index" | "findIndex" | "position")
                {
                    call(interp, &target, vec![y.clone()], span)?.truthy()
                } else {
                    interp.values_equal(y, &target)?
                };
                if hit {
                    found = Some(i);
                    break;
                }
            }
            match found {
                Some(i) => Value::Int(i as i64),
                None if name == "index" => {
                    let r = interp.repr(&target)?;
                    return Err(interp.error(
                        "ValueError",
                        format!("{r} is not in the list"),
                        span,
                    ));
                }
                None => Value::Int(-1),
            }
        }
        "contains" | "includes" | "include?" | "Contains" | "member?" | "has" => {
            let x = super::arg(interp, args, 0, name)?;
            Value::Bool(interp.contains(v, &x, span)?)
        }
        // Rust's `iter.collect()` takes no function.
        "collect" | "collect_vec" | "to_vec" | "into_iter" | "iter" | "cloned" | "copied"
            if args.pos.is_empty() =>
        {
            Value::list(items())
        }
        "map" | "collect" | "Select" | "map_with_index" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut out = Vec::new();
            for (i, x) in items().into_iter().enumerate() {
                out.push(call_flex(interp, &f, vec![x, Value::Int(i as i64)], span)?);
            }
            Value::list(out)
        }
        "filter" | "select" | "where" | "Where" | "find_all" | "filter_map_bool" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut out = Vec::new();
            for (i, x) in items().into_iter().enumerate() {
                if call_flex(interp, &f, vec![x.clone(), Value::Int(i as i64)], span)?.truthy() {
                    out.push(x);
                }
            }
            Value::list(out)
        }
        "reject" | "filterNot" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut out = Vec::new();
            for x in items() {
                if !call(interp, &f, vec![x.clone()], span)?.truthy() {
                    out.push(x);
                }
            }
            Value::list(out)
        }
        "filter_map" | "mapNotNull" | "compact_map" | "compactMap" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut out = Vec::new();
            for x in items() {
                let r = call(interp, &f, vec![x], span)?;
                if !matches!(r, Value::Nil | Value::Bool(false)) {
                    out.push(r);
                }
            }
            Value::list(out)
        }
        "reduce" | "fold" | "inject" | "Aggregate" | "foldLeft" => {
            // JS: reduce(f, init); Ruby: inject(init) { }, Rust: fold(init, f)
            let (f, init) = match (args.get(0), args.get(1)) {
                (Some(a), Some(b)) if a.is_callable() => (a.clone(), Some(b.clone())),
                (Some(a), Some(b)) => (b.clone(), Some(a.clone())),
                (Some(a), None) => (a.clone(), None),
                _ => {
                    return interp
                        .type_err(format!("`{name}` needs a function"), span)
                        .map(Some);
                }
            };
            let mut it = items().into_iter();
            let mut acc = match init {
                Some(i) => i,
                None => match it.next() {
                    Some(x) => x,
                    None => {
                        return Err(interp.error(
                            "TypeError",
                            "reduce of an empty list with no initial value",
                            span,
                        ));
                    }
                },
            };
            for x in it {
                acc = call(interp, &f, vec![acc, x], span)?;
            }
            acc
        }
        "forEach" | "each" | "for_each" | "ForEach" | "iter_each" | "each_with_index"
        | "forEachIndexed" => {
            let f = super::arg(interp, args, 0, name)?;
            for (i, x) in items().into_iter().enumerate() {
                if name == "each_with_index" {
                    call_flex(interp, &f, vec![x, Value::Int(i as i64)], span)?;
                } else if name == "forEachIndexed" {
                    call(interp, &f, vec![Value::Int(i as i64), x], span)?;
                } else {
                    call_flex(interp, &f, vec![x, Value::Int(i as i64)], span)?;
                }
            }
            if name == "each" {
                v.clone()
            } else {
                Value::Nil
            }
        }
        "sort" | "sort!" | "Sort" | "sort_unstable" => {
            let f = args.named("key").or_else(|| args.get(0).cloned());
            let reverse = args.named("reverse").is_some_and(|r| r.truthy());
            let sorted = sort_values(interp, items(), f.as_ref(), reverse, span)?;
            *l.borrow_mut() = sorted;
            if name == "sort" && args.named("key").is_none() && args.get(0).is_none() {
                // Python's list.sort() returns None; JS/Ruby return the array.
                v.clone()
            } else {
                v.clone()
            }
        }
        "sorted" | "sort_by" | "sortBy" | "sortedBy" | "OrderBy" | "sorted_by" => {
            let f = args.named("key").or_else(|| args.get(0).cloned());
            let reverse = args.named("reverse").is_some_and(|r| r.truthy());
            Value::list(sort_values(interp, items(), f.as_ref(), reverse, span)?)
        }
        "sortedByDescending" | "OrderByDescending" | "sort_by_desc" => {
            let f = args.get(0).cloned();
            Value::list(sort_values(interp, items(), f.as_ref(), true, span)?)
        }
        "reverse" | "reverse!" | "Reverse" => {
            l.borrow_mut().reverse();
            v.clone()
        }
        // C++ iterators are `(container, index)` pairs.
        "begin" | "cbegin" => Value::tuple(vec![v.clone(), Value::Int(0)]),
        "end" | "cend" => Value::tuple(vec![v.clone(), Value::Int(items().len() as i64)]),
        "reversed" | "toReversed" => {
            let mut r = items();
            r.reverse();
            Value::list(r)
        }
        "join" | "mkString" | "joinToString" | "Join" => {
            let sep = match args.get(0) {
                Some(s) => interp.to_str(s)?,
                None => {
                    if matches!(name, "join") && false {
                        String::new()
                    } else {
                        ",".into()
                    }
                }
            };
            let sep = if args.get(0).is_none() && name != "join" {
                ", ".into()
            } else {
                sep
            };
            let mut parts = Vec::new();
            for x in items() {
                parts.push(interp.to_str(&x)?);
            }
            Value::str(&parts.join(&sep))
        }
        "slice" | "subList" | "sublist" | "GetRange" => {
            let all = items();
            let n = all.len() as i64;
            let a = args.get(0).and_then(Value::as_int).unwrap_or(0);
            let b = args.get(1).and_then(Value::as_int).unwrap_or(n);
            let norm = |x: i64| if x < 0 { (n + x).max(0) } else { x.min(n) };
            let (a, b) = (norm(a), norm(b));
            Value::list(if a < b {
                all[a as usize..b as usize].to_vec()
            } else {
                Vec::new()
            })
        }
        "first" | "head" | "First" | "front" => match args.get(0).and_then(Value::as_int) {
            Some(n) => Value::list(items().into_iter().take(n.max(0) as usize).collect()),
            None => match args.get(0) {
                Some(f) if f.is_callable() => {
                    let mut found = Value::Nil;
                    for x in items() {
                        if call(interp, f, vec![x.clone()], span)?.truthy() {
                            found = x;
                            break;
                        }
                    }
                    found
                }
                _ => l.borrow().first().cloned().unwrap_or(Value::Nil),
            },
        },
        "last" | "Last" | "back" => match args.get(0).and_then(Value::as_int) {
            Some(n) => {
                let all = items();
                Value::list(all[all.len().saturating_sub(n.max(0) as usize)..].to_vec())
            }
            None => l.borrow().last().cloned().unwrap_or(Value::Nil),
        },
        "tail" | "rest" => Value::list(items().into_iter().skip(1).collect()),
        "get" | "at" | "elementAt" | "fetch" | "nth" => {
            let i = super::arg(interp, args, 0, name)?;
            let len = l.borrow().len();
            let idx = i.as_int().unwrap_or(0);
            let j = if idx < 0 { idx + len as i64 } else { idx };
            if j < 0 || j >= len as i64 {
                args.get(1).cloned().unwrap_or(Value::Nil)
            } else {
                l.borrow()[j as usize].clone()
            }
        }
        "set" => {
            let i = super::arg(interp, args, 0, name)?;
            let x = super::arg(interp, args, 1, name)?;
            let len = l.borrow().len();
            let idx = interp.list_index(&i, len, span)?;
            l.borrow_mut()[idx] = x;
            Value::Nil
        }
        "min" | "max" | "Min" | "Max" | "minimum" | "maximum" | "min_by" | "max_by" | "minBy"
        | "maxBy" | "minOf" | "maxOf" => {
            let all = items();
            let want_max = name.to_ascii_lowercase().starts_with("max");
            if all.is_empty() {
                return Ok(Some(Value::Nil));
            }
            let key = args.get(0).cloned();
            let mut best = all[0].clone();
            let mut best_key = match &key {
                Some(f) => call(interp, f, vec![best.clone()], span)?,
                None => best.clone(),
            };
            for x in all.into_iter().skip(1) {
                let k = match &key {
                    Some(f) => call(interp, f, vec![x.clone()], span)?,
                    None => x.clone(),
                };
                let o = interp.compare_values(&k, &best_key, span)?;
                if (want_max && o.is_gt()) || (!want_max && o.is_lt()) {
                    best = x;
                    best_key = k;
                }
            }
            if matches!(name, "minOf" | "maxOf") {
                best_key
            } else {
                best
            }
        }
        "sum" | "Sum" | "sum_by" | "sumOf" | "sumBy" => {
            let mut acc = Value::Int(0);
            let f = args.get(0).cloned();
            for x in items() {
                let x = match &f {
                    Some(f) => call(interp, f, vec![x], span)?,
                    None => x,
                };
                acc = interp.binary_op(crate::ast::BinOp::Add, acc, x, span)?;
            }
            acc
        }
        "product" => {
            let mut acc = Value::Int(1);
            for x in items() {
                acc = interp.binary_op(crate::ast::BinOp::Mul, acc, x, span)?;
            }
            acc
        }
        "average" | "mean" | "Average" | "avg" => {
            let all = items();
            if all.is_empty() {
                return Ok(Some(Value::Nil));
            }
            let n = all.len();
            let mut acc = Value::Int(0);
            for x in all {
                acc = interp.binary_op(crate::ast::BinOp::Add, acc, x, span)?;
            }
            interp.binary_op(crate::ast::BinOp::Div, acc, Value::Int(n as i64), span)?
        }
        "any" | "some" | "Any" | "any?" => {
            let f = args.get(0).cloned();
            let mut r = false;
            for x in items() {
                let ok = match &f {
                    Some(f) => call(interp, f, vec![x], span)?.truthy(),
                    None => x.truthy(),
                };
                if ok {
                    r = true;
                    break;
                }
            }
            Value::Bool(r)
        }
        "all" | "every" | "All" | "all?" => {
            let f = args.get(0).cloned();
            let mut r = true;
            for x in items() {
                let ok = match &f {
                    Some(f) => call(interp, f, vec![x], span)?.truthy(),
                    None => x.truthy(),
                };
                if !ok {
                    r = false;
                    break;
                }
            }
            Value::Bool(r)
        }
        "none" | "none?" | "noneMatch" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut r = true;
            for x in items() {
                if call(interp, &f, vec![x], span)?.truthy() {
                    r = false;
                    break;
                }
            }
            Value::Bool(r)
        }
        "find" | "detect" | "Find" | "firstOrNull" | "find_first" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut found = Value::Nil;
            for x in items() {
                if call(interp, &f, vec![x.clone()], span)?.truthy() {
                    found = x;
                    break;
                }
            }
            found
        }
        "flat" | "flatten" | "flatten!" | "Flatten" => {
            let depth = match args.get(0) {
                Some(Value::Float(f)) if f.is_infinite() => i64::MAX,
                Some(d) => d.as_int().unwrap_or(1),
                None if name == "flat" => 1,
                None => i64::MAX,
            };
            Value::list(flatten(&items(), depth))
        }
        "flatMap" | "flat_map" | "SelectMany" | "concat_map" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut out = Vec::new();
            for x in items() {
                let r = call(interp, &f, vec![x], span)?;
                match r {
                    Value::List(_) | Value::Tuple(_) => out.extend(interp.collect(&r, span)?),
                    other => out.push(other),
                }
            }
            Value::list(out)
        }
        "unique" | "uniq" | "distinct" | "Distinct" | "dedup" | "toSet_list" => {
            let mut out: Vec<Value> = Vec::new();
            for x in items() {
                let mut seen = false;
                for y in &out {
                    if interp.values_equal(y, &x)? {
                        seen = true;
                        break;
                    }
                }
                if !seen {
                    out.push(x);
                }
            }
            Value::list(out)
        }
        "group_by" | "groupBy" | "GroupBy" | "chunk_by" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut d = Dict::new();
            for x in items() {
                let k = call(interp, &f, vec![x.clone()], span)?;
                match d.get(&k) {
                    Some(Value::List(g)) => g.borrow_mut().push(x),
                    _ => {
                        d.insert(k, Value::list(vec![x]));
                    }
                }
            }
            Value::Map(Rc::new(RefCell::new(d)))
        }
        "count_by" | "tally" | "countBy" | "frequencies" => {
            let f = args.get(0).cloned();
            let mut d = Dict::new();
            for x in items() {
                let k = match &f {
                    Some(f) => call(interp, f, vec![x], span)?,
                    None => x,
                };
                let n = d.get(&k).and_then(|v| v.as_int()).unwrap_or(0);
                d.insert(k, Value::Int(n + 1));
            }
            Value::Map(Rc::new(RefCell::new(d)))
        }
        "partition" => {
            let f = super::arg(interp, args, 0, name)?;
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for x in items() {
                if call(interp, &f, vec![x.clone()], span)?.truthy() {
                    a.push(x);
                } else {
                    b.push(x);
                }
            }
            Value::tuple(vec![Value::list(a), Value::list(b)])
        }
        "zip" | "Zip" => {
            let other = super::arg(interp, args, 0, name)?;
            let ys = interp.collect(&other, span)?;
            Value::list(
                items()
                    .into_iter()
                    .zip(ys)
                    .map(|(a, b)| Value::tuple(vec![a, b]))
                    .collect(),
            )
        }
        "enumerate" | "with_index" | "withIndex" | "indexed" => Value::list(
            items()
                .into_iter()
                .enumerate()
                .map(|(i, x)| Value::tuple(vec![Value::Int(i as i64), x]))
                .collect(),
        ),
        "take" | "Take" | "limit" => {
            let n = super::int_arg(interp, args, 0, name)?.max(0) as usize;
            Value::list(items().into_iter().take(n).collect())
        }
        "drop" | "skip" | "Skip" => {
            let n = super::int_arg(interp, args, 0, name)?.max(0) as usize;
            Value::list(items().into_iter().skip(n).collect())
        }
        "takeWhile" | "take_while" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut out = Vec::new();
            for x in items() {
                if !call(interp, &f, vec![x.clone()], span)?.truthy() {
                    break;
                }
                out.push(x);
            }
            Value::list(out)
        }
        "chunk" | "chunks" | "each_slice" | "chunked" => {
            let n = super::int_arg(interp, args, 0, name)?.max(1) as usize;
            Value::list(items().chunks(n).map(|c| Value::list(c.to_vec())).collect())
        }
        "windows" | "each_cons" | "windowed" => {
            let n = super::int_arg(interp, args, 0, name)?.max(1) as usize;
            Value::list(
                items()
                    .windows(n)
                    .map(|c| Value::list(c.to_vec()))
                    .collect(),
            )
        }
        "copy" | "clone" | "dup" | "to_vec" | "slice_all" | "toList" | "to_list" | "to_a"
        | "toArray" | "list" | "tolist" | "iter" | "into_iter" | "collect_list" | "ToList" => {
            Value::list(items())
        }
        "to_set" | "toSet" | "ToHashSet" => {
            let mut d = Dict::new();
            for x in items() {
                d.insert(x, Value::Nil);
            }
            Value::Set(Rc::new(RefCell::new(d)))
        }
        "to_tensor" | "to_numpy" => Value::Tensor(interp.to_tensor(v, span)?),
        "is_empty" | "isEmpty" | "empty?" | "IsEmpty" => Value::Bool(l.borrow().is_empty()),
        "compact" | "compact!" => Value::list(
            items()
                .into_iter()
                .filter(|x| !matches!(x, Value::Nil))
                .collect(),
        ),
        "fill" => {
            let x = super::arg(interp, args, 0, name)?;
            let n = l.borrow().len();
            *l.borrow_mut() = vec![x; n];
            v.clone()
        }
        "rotate" => {
            let mut all = items();
            if !all.is_empty() {
                let k = args
                    .get(0)
                    .and_then(Value::as_int)
                    .unwrap_or(1)
                    .rem_euclid(all.len() as i64) as usize;
                all.rotate_left(k);
            }
            Value::list(all)
        }
        "swap" => {
            let len = l.borrow().len();
            let a = super::arg(interp, args, 0, name)?;
            let b = super::arg(interp, args, 1, name)?;
            let i = interp.list_index(&a, len, span)?;
            let j = interp.list_index(&b, len, span)?;
            l.borrow_mut().swap(i, j);
            Value::Nil
        }
        "shuffle" | "shuffled" | "shuffle!" => {
            let mut all = items();
            interp.rng.shuffle(&mut all);
            if name == "shuffle!" {
                *l.borrow_mut() = all;
                v.clone()
            } else {
                Value::list(all)
            }
        }
        "sample" | "random" | "choice" => {
            let all = items();
            if all.is_empty() {
                Value::Nil
            } else {
                let i = interp.rng.index(all.len());
                all[i].clone()
            }
        }
        "cycle" => {
            let n = super::int_arg(interp, args, 0, name)?.clamp(0, 1_000_000) as usize;
            let all = items();
            let mut out = Vec::with_capacity(all.len() * n);
            for _ in 0..n {
                out.extend(all.iter().cloned());
            }
            Value::list(out)
        }
        "keys" => Value::list(
            (0..l.borrow().len())
                .map(|i| Value::Int(i as i64))
                .collect(),
        ),
        _ => return Ok(None),
    };
    Ok(Some(out))
}

fn flatten(items: &[Value], depth: i64) -> Vec<Value> {
    let mut out = Vec::new();
    for x in items {
        match x {
            Value::List(l) if depth > 0 => out.extend(flatten(&l.borrow(), depth - 1)),
            Value::Tuple(t) if depth > 0 => out.extend(flatten(t, depth - 1)),
            other => out.push(other.clone()),
        }
    }
    out
}

fn map_method(
    interp: &mut Interp,
    v: &Value,
    m: &Rc<RefCell<Dict>>,
    name: &str,
    args: &Args,
) -> R<Option<Value>> {
    let span = args.span;
    let entries = || {
        m.borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>()
    };
    let out = match name {
        "keys" | "keySet" | "Keys" | "keys_list" => Value::list(m.borrow().keys()),
        "values" | "Values" | "values_list" => Value::list(m.borrow().values()),
        "items" | "entries" | "entrySet" | "to_a" | "to_list" | "pairs" | "iter" => Value::list(
            entries()
                .into_iter()
                .map(|(k, v)| Value::tuple(vec![k, v]))
                .collect(),
        ),
        "get" | "fetch" | "getOrDefault" | "get_or" | "Get" | "TryGetValue" => {
            let k = super::arg(interp, args, 0, name)?;
            match m.borrow().get(&k) {
                Some(v) => v,
                None => match args.get(1) {
                    Some(d) => d.clone(),
                    None if name == "fetch" => {
                        let r = interp.repr(&k)?;
                        return Err(interp.error("KeyError", format!("key not found: {r}"), span));
                    }
                    None => Value::Nil,
                },
            }
        }
        "set" | "put" | "insert" | "store" | "Add" | "set!" => {
            let k = super::arg(interp, args, 0, name)?;
            let x = super::arg(interp, args, 1, name)?;
            let old = m.borrow().get(&k);
            if !m.borrow_mut().insert(k.clone(), x) {
                let t = k.type_name();
                return interp
                    .type_err(format!("unhashable key type: {t}"), span)
                    .map(Some);
            }
            if name == "set" {
                v.clone()
            } else {
                old.unwrap_or(Value::Nil)
            }
        }
        "has" | "has_key" | "has_key?" | "key?" | "containsKey" | "contains_key" | "includes"
        | "contains" | "ContainsKey" | "include?" | "member?" => {
            let k = super::arg(interp, args, 0, name)?;
            Value::Bool(m.borrow().contains(&k))
        }
        "containsValue" | "has_value" | "value?" => {
            let x = super::arg(interp, args, 0, name)?;
            let mut found = false;
            for val in m.borrow().values() {
                if interp.values_equal(&val, &x)? {
                    found = true;
                    break;
                }
            }
            Value::Bool(found)
        }
        "remove" | "delete" | "pop" | "Remove" => {
            let k = super::arg(interp, args, 0, name)?;
            let r = m.borrow_mut().remove(&k);
            match r {
                Some(x) => x,
                None => match args.get(1) {
                    Some(d) => d.clone(),
                    None if name == "pop" => {
                        let r = interp.repr(&k)?;
                        return Err(interp.error("KeyError", r, span));
                    }
                    None => Value::Nil,
                },
            }
        }
        "update" | "merge!" | "extend" | "putAll" => {
            for a in &args.pos {
                if let Value::Map(other) = a {
                    for (k, x) in other.borrow().iter() {
                        m.borrow_mut().insert(k.clone(), x.clone());
                    }
                }
            }
            for (k, x) in &args.named {
                m.borrow_mut().insert(Value::Str(k.as_str()), x.clone());
            }
            Value::Nil
        }
        "merge" | "plus" | "assign" => {
            let mut d = m.borrow().clone();
            for a in &args.pos {
                if let Value::Map(other) = a {
                    for (k, x) in other.borrow().iter() {
                        d.insert(k.clone(), x.clone());
                    }
                }
            }
            Value::Map(Rc::new(RefCell::new(d)))
        }
        "setdefault" | "get_or_insert" | "computeIfAbsent" | "or_insert" | "putIfAbsent" => {
            let k = super::arg(interp, args, 0, name)?;
            let existing = m.borrow().get(&k);
            match existing {
                Some(x) => x,
                None => {
                    let d = match args.get(1) {
                        Some(f) if f.is_callable() && name == "computeIfAbsent" => {
                            call(interp, f, vec![k.clone()], span)?
                        }
                        Some(d) => d.clone(),
                        None => Value::Nil,
                    };
                    m.borrow_mut().insert(k, d.clone());
                    d
                }
            }
        }
        "len" | "length" | "size" | "count" | "Count" => Value::Int(m.borrow().len() as i64),
        "clear" | "Clear" => {
            m.borrow_mut().clear();
            Value::Nil
        }
        "copy" | "clone" | "dup" | "to_h" | "toMap" | "ToDictionary" => {
            Value::Map(Rc::new(RefCell::new(m.borrow().clone())))
        }
        "is_empty" | "isEmpty" | "empty?" => Value::Bool(m.borrow().is_empty()),
        "each" | "forEach" | "for_each" | "each_pair" | "ForEach" => {
            let f = super::arg(interp, args, 0, name)?;
            for (k, x) in entries() {
                // JS forEach passes (value, key); Ruby each passes (key, value).
                let pair = if name == "forEach" {
                    vec![x, k]
                } else {
                    vec![k, x]
                };
                call_flex(interp, &f, pair, span)?;
            }
            Value::Nil
        }
        "map" | "map_pairs" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut out = Vec::new();
            for (k, x) in entries() {
                out.push(call_flex(interp, &f, vec![k, x], span)?);
            }
            Value::list(out)
        }
        "transform_values" | "map_values" | "mapValues" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut d = Dict::new();
            for (k, x) in entries() {
                let nx = call(interp, &f, vec![x], span)?;
                d.insert(k, nx);
            }
            Value::Map(Rc::new(RefCell::new(d)))
        }
        "filter" | "select" | "filter_pairs" => {
            let f = super::arg(interp, args, 0, name)?;
            let mut d = Dict::new();
            for (k, x) in entries() {
                if call_flex(interp, &f, vec![k.clone(), x.clone()], span)?.truthy() {
                    d.insert(k, x);
                }
            }
            Value::Map(Rc::new(RefCell::new(d)))
        }
        "invert" => {
            let mut d = Dict::new();
            for (k, x) in entries() {
                d.insert(x, k);
            }
            Value::Map(Rc::new(RefCell::new(d)))
        }
        "sort_by" | "sorted" | "sortedBy" => {
            let pairs: Vec<Value> = entries()
                .into_iter()
                .map(|(k, x)| Value::tuple(vec![k, x]))
                .collect();
            let f = args.get(0).cloned();
            Value::list(sort_values(interp, pairs, f.as_ref(), false, span)?)
        }
        "min_by" | "max_by" | "minBy" | "maxBy" => {
            let pairs = Value::list(
                entries()
                    .into_iter()
                    .map(|(k, x)| Value::tuple(vec![k, x]))
                    .collect(),
            );
            let Value::List(l) = &pairs else {
                unreachable!()
            };
            return list_method(interp, &pairs, l, name, args);
        }
        "values_at" => {
            let mut out = Vec::new();
            for k in &args.pos {
                out.push(m.borrow().get(k).unwrap_or(Value::Nil));
            }
            Value::list(out)
        }
        "dig" => {
            let mut cur = v.clone();
            for k in &args.pos {
                cur = match &cur {
                    Value::Map(mm) => mm.borrow().get(k).unwrap_or(Value::Nil),
                    Value::List(_) => interp.index_value(&cur, k, span).unwrap_or(Value::Nil),
                    _ => Value::Nil,
                };
            }
            cur
        }
        "any" | "all" | "count_if" | "find" | "sum" | "reduce" => {
            let pairs = Value::list(
                entries()
                    .into_iter()
                    .map(|(k, x)| Value::tuple(vec![k, x]))
                    .collect(),
            );
            let Value::List(l) = &pairs else {
                unreachable!()
            };
            return list_method(interp, &pairs, l, name, args);
        }
        _ => return Ok(None),
    };
    Ok(Some(out))
}

fn num_method(interp: &mut Interp, v: &Value, name: &str, args: &Args) -> R<Option<Value>> {
    let span = args.span;
    let math = |interp: &mut Interp, f: &str, extra: Vec<Value>| -> R<Value> {
        let func = interp.builtins.get(&sym(f)).cloned().unwrap_or(Value::Nil);
        let mut a = vec![v.clone()];
        a.extend(extra);
        interp.call_value(func, Args::new(a, span))
    };
    let out = match name {
        "abs" | "floor" | "ceil" | "sqrt" | "round" | "trunc" | "sign" | "signum" | "exp"
        | "log" | "sin" | "cos" | "tan" => {
            let f = if name == "signum" { "sign" } else { name };
            math(interp, f, args.pos.clone())?
        }
        "pow" | "powi" | "powf" => {
            let e = super::arg(interp, args, 0, name)?;
            interp.binary_op(crate::ast::BinOp::Pow, v.clone(), e, span)?
        }
        "to_s" | "toString" | "to_string" | "ToString" => match args.get(0).and_then(Value::as_int)
        {
            Some(radix) if (2..=36).contains(&radix) => match v {
                Value::Int(i) => {
                    let s = crate::bigint::BigInt::from_i64(*i).to_string_radix(radix as u32);
                    Value::str(&s)
                }
                _ => {
                    let s = interp.to_str(v)?;
                    Value::str(&s)
                }
            },
            _ => {
                let s = interp.to_str(v)?;
                Value::str(&s)
            }
        },
        "to_i" | "to_int" | "toInt" | "trunc_int" => super::to_int(interp, v, 10, span)?,
        "to_f" | "to_float" | "toDouble" | "toFloat" => Value::Float(v.as_f64().unwrap_or(0.0)),
        "toFixed" | "round_to" => {
            let d = args
                .get(0)
                .and_then(Value::as_int)
                .unwrap_or(0)
                .clamp(0, 50) as usize;
            Value::str(&format!("{:.d$}", v.as_f64().unwrap_or(0.0)))
        }
        "toPrecision" => {
            let p = args
                .get(0)
                .and_then(Value::as_int)
                .unwrap_or(6)
                .clamp(1, 50);
            let s = interp.format_spec(v, &format!(".{p}g"), span)?;
            Value::str(&s)
        }
        "format" => {
            let spec = match args.get(0) {
                Some(s) => interp.to_str(s)?,
                None => String::new(),
            };
            Value::str(&interp.format_spec(v, &spec, span)?)
        }
        "times" => {
            let n = v.as_int().unwrap_or(0);
            let f = super::arg(interp, args, 0, name)?;
            for i in 0..n.max(0) {
                call_flex(interp, &f, vec![Value::Int(i)], span)?;
            }
            v.clone()
        }
        "upto" | "downto" | "to" | "until" | "rangeTo" => {
            let a = v.as_int().unwrap_or(0);
            let b = super::int_arg(interp, args, 0, name)?;
            let range = match name {
                "downto" => RangeVal {
                    start: a,
                    end: b - 1,
                    step: -1,
                },
                "until" => RangeVal {
                    start: a,
                    end: b,
                    step: 1,
                },
                _ => RangeVal {
                    start: a,
                    end: b + 1,
                    step: 1,
                },
            };
            let r = Value::Range(Rc::new(range));
            if let Some(f) = args.get(1) {
                let f = f.clone();
                for x in interp.collect(&r, span)? {
                    call(interp, &f, vec![x], span)?;
                }
                v.clone()
            } else {
                r
            }
        }
        "is_integer" | "isInteger" => Value::Bool(match v {
            Value::Float(f) => f.fract() == 0.0,
            _ => true,
        }),
        "is_nan" | "isNaN" => Value::Bool(v.as_f64().is_some_and(f64::is_nan)),
        "is_finite" | "isFinite" => Value::Bool(v.as_f64().is_some_and(f64::is_finite)),
        "even?" | "is_even" => Value::Bool(v.as_int().is_some_and(|i| i % 2 == 0)),
        "odd?" | "is_odd" => Value::Bool(v.as_int().is_some_and(|i| i % 2 != 0)),
        "zero?" | "is_zero" => Value::Bool(v.as_f64() == Some(0.0)),
        "positive?" => Value::Bool(v.as_f64().is_some_and(|x| x > 0.0)),
        "negative?" => Value::Bool(v.as_f64().is_some_and(|x| x < 0.0)),
        "between?" => {
            let lo = super::arg(interp, args, 0, name)?;
            let hi = super::arg(interp, args, 1, name)?;
            Value::Bool(
                interp.compare_values(v, &lo, span)?.is_ge()
                    && interp.compare_values(v, &hi, span)?.is_le(),
            )
        }
        "clamp" | "coerceIn" => {
            let lo = super::arg(interp, args, 0, name)?;
            let hi = super::arg(interp, args, 1, name)?;
            if interp.compare_values(v, &lo, span)?.is_lt() {
                lo
            } else if interp.compare_values(v, &hi, span)?.is_gt() {
                hi
            } else {
                v.clone()
            }
        }
        "min" | "max" => {
            let o = super::arg(interp, args, 0, name)?;
            let ord = interp.compare_values(v, &o, span)?;
            if (name == "min") == ord.is_le() {
                v.clone()
            } else {
                o
            }
        }
        "succ" | "next" => {
            interp.binary_op(crate::ast::BinOp::Add, v.clone(), Value::Int(1), span)?
        }
        "pred" | "prev" => {
            interp.binary_op(crate::ast::BinOp::Sub, v.clone(), Value::Int(1), span)?
        }
        "chr" => {
            let c = v
                .as_int()
                .and_then(|i| char::from_u32(i as u32))
                .unwrap_or('\u{fffd}');
            Value::str(&c.to_string())
        }
        "bit_length" => Value::Int(
            v.as_int()
                .map_or(0, |i| 64 - i64::from(i.unsigned_abs().leading_zeros())),
        ),
        "digits" => {
            let s = interp.to_str(v)?;
            Value::list(
                s.chars()
                    .filter(char::is_ascii_digit)
                    .rev()
                    .map(|c| Value::Int(i64::from(c as u8 - b'0')))
                    .collect(),
            )
        }
        "gcd" | "lcm" => math(interp, name, args.pos.clone())?,
        "real" | "conjugate" => v.clone(),
        "imag" => Value::Int(0),
        "hex" => {
            let f = interp
                .builtins
                .get(&sym("hex"))
                .cloned()
                .unwrap_or(Value::Nil);
            interp.call_value(f, Args::new(vec![v.clone()], span))?
        }
        _ => return Ok(None),
    };
    Ok(Some(out))
}
