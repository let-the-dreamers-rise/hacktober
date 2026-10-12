//! Turning values into text: `str`/`repr`, Python format specs
//! (`{x:>10.2f}`), C/Go `printf` verbs and Rust/Python brace templates.

use std::rc::Rc;

use chimera_core::Span;

use super::{Interp, sym};
use crate::ast::TypeExpr;
use crate::value::*;

/// Formats a float the way Python's `repr` does.
pub fn fmt_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let a = f.abs();
    if a != 0.0 && !(1e-4..1e16).contains(&a) {
        let s = format!("{f:e}");
        // Rust: 1e21, 1.5e-7 → Python: 1e+21, 1.5e-07
        if let Some((mantissa, exp)) = s.split_once('e') {
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ("-", d),
                None => ("+", exp),
            };
            return format!("{mantissa}e{sign}{digits:0>2}");
        }
        return s;
    }
    if f.fract() == 0.0 {
        return format!("{f:.1}");
    }
    format!("{f}")
}

/// `repr` without calling user code (used for debugging and error text).
pub fn repr_plain(v: &Value) -> String {
    match v {
        Value::Str(s) => chimera_core::text::escape_debug(s),
        other => display_plain(other, true),
    }
}

/// Display without calling user code.
pub fn display_plain(v: &Value, nested: bool) -> String {
    match v {
        Value::Nil => "nil".into(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => fmt_float(*f),
        Value::Big(b) => b.to_string(),
        Value::Str(s) => {
            if nested {
                chimera_core::text::escape_debug(s)
            } else {
                s.to_string()
            }
        }
        Value::List(l) => {
            let items: Vec<String> = l.borrow().iter().map(|x| display_plain(x, true)).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Tuple(t) => {
            let items: Vec<String> = t.iter().map(|x| display_plain(x, true)).collect();
            if items.len() == 1 {
                format!("({},)", items[0])
            } else {
                format!("({})", items.join(", "))
            }
        }
        Value::Map(m) => {
            let items: Vec<String> = m
                .borrow()
                .iter()
                .map(|(k, v)| format!("{}: {}", display_plain(k, true), display_plain(v, true)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        Value::Set(m) => {
            let items: Vec<String> = m
                .borrow()
                .keys()
                .iter()
                .map(|k| display_plain(k, true))
                .collect();
            if items.is_empty() {
                "set()".into()
            } else {
                format!("{{{}}}", items.join(", "))
            }
        }
        Value::Range(r) => {
            if r.step == 1 {
                format!("{}..{}", r.start, r.end)
            } else {
                format!("{}..{} by {}", r.start, r.end, r.step)
            }
        }
        Value::Func(c) => format!("<fn {}>", c.name()),
        Value::Native(n) => format!("<builtin {}>", n.name),
        Value::Bound(b) => match &b.func {
            Value::Func(c) => format!("<method {}>", c.name()),
            Value::Native(n) => format!("<method {}>", n.name),
            _ => "<method>".into(),
        },
        Value::Class(c) => format!("<class {}>", c.name),
        Value::Instance(i) => {
            if i.class.has_ancestor_named("Error") {
                // `str(e)` is the message (Python, Ruby, Java's getMessage);
                // `repr(e)` names the class.
                let msg = i
                    .get(sym("message"))
                    .map(|m| display_plain(&m, false))
                    .unwrap_or_default();
                return if nested {
                    format!(
                        "{}({})",
                        i.class.name,
                        chimera_core::text::escape_debug(&msg)
                    )
                } else {
                    msg
                };
            }
            let fields: Vec<String> = i
                .fields
                .borrow()
                .iter()
                .map(|(k, v)| format!("{k}={}", display_plain(v, true)))
                .collect();
            format!("{}({})", i.class.name, fields.join(", "))
        }
        Value::Enum(e) => format!("<enum {}>", e.name),
        Value::Variant(v) => variant_text(v, |x| display_plain(x, true)),
        Value::Module(m) => format!("<module {}>", m.name),
        Value::Type(t) => format!("<type {}>", t.name()),
        Value::Tensor(t) => t.display(),
        Value::Dual(d) => format!(
            "{} + {}ε",
            display_plain(&d.re, true),
            display_plain(&d.du, true)
        ),
        Value::Object(o) => o.display(),
    }
}

fn variant_text(v: &Variant, mut show: impl FnMut(&Value) -> String) -> String {
    let info = v.info();
    if v.fields.is_empty() {
        return info.name.to_string();
    }
    if !info.field_names.is_empty() {
        let parts: Vec<String> = info
            .field_names
            .iter()
            .zip(&v.fields)
            .map(|(n, f)| format!("{n}: {}", show(f)))
            .collect();
        return format!("{} {{ {} }}", info.name, parts.join(", "));
    }
    let parts: Vec<String> = v.fields.iter().map(show).collect();
    format!("{}({})", info.name, parts.join(", "))
}

/// Human-readable name of a type annotation.
pub fn type_expr_name(t: &TypeExpr) -> String {
    match t {
        TypeExpr::Any => "any".into(),
        TypeExpr::Named(n, args) if args.is_empty() => n.to_string(),
        TypeExpr::Named(n, args) => format!(
            "{n}<{}>",
            args.iter()
                .map(type_expr_name)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TypeExpr::Optional(inner) => format!("{}?", type_expr_name(inner)),
        TypeExpr::Union(alts) => alts
            .iter()
            .map(type_expr_name)
            .collect::<Vec<_>>()
            .join(" | "),
        TypeExpr::List(inner) => format!("{}[]", type_expr_name(inner)),
        TypeExpr::Tuple(items) => format!(
            "({})",
            items
                .iter()
                .map(type_expr_name)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// A parsed Python format spec.
#[derive(Debug, Default)]
struct Spec {
    fill: char,
    align: Option<char>,
    sign: Option<char>,
    alt: bool,
    zero: bool,
    width: usize,
    grouping: Option<char>,
    precision: Option<usize>,
    kind: Option<char>,
}

fn parse_spec(spec: &str) -> Spec {
    let chars: Vec<char> = spec.chars().collect();
    let mut s = Spec {
        fill: ' ',
        ..Spec::default()
    };
    let mut i = 0;
    let is_align = |c: char| matches!(c, '<' | '>' | '^' | '=');
    if chars.len() >= 2 && is_align(chars[1]) {
        s.fill = chars[0];
        s.align = Some(chars[1]);
        i = 2;
    } else if !chars.is_empty() && is_align(chars[0]) {
        s.align = Some(chars[0]);
        i = 1;
    }
    if i < chars.len() && matches!(chars[i], '+' | '-' | ' ') {
        s.sign = Some(chars[i]);
        i += 1;
    }
    if i < chars.len() && chars[i] == '#' {
        s.alt = true;
        i += 1;
    }
    if i < chars.len() && chars[i] == '0' {
        s.zero = true;
        i += 1;
    }
    let mut width = String::new();
    while i < chars.len() && chars[i].is_ascii_digit() {
        width.push(chars[i]);
        i += 1;
    }
    s.width = width.parse().unwrap_or(0);
    if i < chars.len() && matches!(chars[i], ',' | '_') {
        s.grouping = Some(chars[i]);
        i += 1;
    }
    if i < chars.len() && chars[i] == '.' {
        i += 1;
        let mut p = String::new();
        while i < chars.len() && chars[i].is_ascii_digit() {
            p.push(chars[i]);
            i += 1;
        }
        s.precision = p.parse().ok();
    }
    if i < chars.len() {
        s.kind = Some(chars[i]);
    }
    s
}

fn group_digits(digits: &str, sep: char) -> String {
    let (int_part, rest) = match digits.find('.') {
        Some(p) => (&digits[..p], &digits[p..]),
        None => (digits, ""),
    };
    let (sign, int_part) = match int_part.strip_prefix('-') {
        Some(d) => ("-", d),
        None => ("", int_part),
    };
    let mut out = String::new();
    for (i, c) in int_part.chars().enumerate() {
        if i > 0 && (int_part.len() - i) % 3 == 0 {
            out.push(sep);
        }
        out.push(c);
    }
    format!("{sign}{out}{rest}")
}

fn pad(s: String, spec: &Spec, numeric: bool) -> String {
    let len = s.chars().count();
    if len >= spec.width {
        return s;
    }
    let n = spec.width - len;
    let align = spec.align.unwrap_or(if numeric { '>' } else { '<' });
    let fill = if spec.zero && spec.align.is_none() && numeric {
        '0'
    } else {
        spec.fill
    };
    if fill == '0' && numeric && spec.align.is_none() {
        // Zero padding goes after the sign.
        let (sign, digits) = if s.starts_with('-') || s.starts_with('+') {
            s.split_at(1)
        } else {
            ("", s.as_str())
        };
        return format!("{sign}{}{digits}", "0".repeat(n));
    }
    let f = fill.to_string();
    match align {
        '<' => format!("{s}{}", f.repeat(n)),
        '^' => format!("{}{s}{}", f.repeat(n / 2), f.repeat(n - n / 2)),
        _ => format!("{}{s}", f.repeat(n)),
    }
}

impl Interp {
    /// `str(v)`: user-defined `__str__`/`toString`/`to_s` are honoured.
    pub fn to_str(&mut self, v: &Value) -> R<String> {
        match v {
            Value::Str(s) => Ok(s.to_string()),
            Value::Instance(i) => {
                for name in [
                    "__str__",
                    "toString",
                    "to_s",
                    "description",
                    "__repr__",
                    "fmt",
                    "to_string",
                    "show",
                    "str",
                ] {
                    if let Some(m) = i.class.find_method(sym(name)) {
                        if name == "fmt" {
                            // Rust `impl Display`: `fn fmt(&self, f) { write!(f, …) }`.
                            return crate::builtins::objects::rust_display(self, v, m);
                        }
                        let bound = Value::Bound(Rc::new(Bound {
                            receiver: v.clone(),
                            func: m,
                        }));
                        let r = self.call_value(bound, super::Args::default())?;
                        return Ok(match r {
                            Value::Str(s) => s.to_string(),
                            other => display_plain(&other, false),
                        });
                    }
                }
                self.display_nested(v)
            }
            Value::List(_)
            | Value::Tuple(_)
            | Value::Map(_)
            | Value::Set(_)
            | Value::Variant(_) => self.display_nested(v),
            other => Ok(display_plain(other, false)),
        }
    }

    /// `repr(v)`: strings quoted; instances via `__repr__` when defined.
    pub fn repr(&mut self, v: &Value) -> R<String> {
        match v {
            Value::Str(s) => Ok(chimera_core::text::escape_debug(s)),
            Value::Instance(i) => {
                if let Some(m) = i.class.find_method(sym("__repr__")) {
                    let bound = Value::Bound(Rc::new(Bound {
                        receiver: v.clone(),
                        func: m,
                    }));
                    let r = self.call_value(bound, super::Args::default())?;
                    return Ok(display_plain(&r, false));
                }
                self.to_str(v)
            }
            other => self.to_str(other),
        }
    }

    /// Display of containers, using `repr` for elements (so user classes
    /// inside lists still print through their own methods).
    fn display_nested(&mut self, v: &Value) -> R<String> {
        Ok(match v {
            Value::List(l) => {
                let items = l.borrow().clone();
                let mut parts = Vec::with_capacity(items.len());
                for x in &items {
                    parts.push(self.repr(x)?);
                }
                format!("[{}]", parts.join(", "))
            }
            Value::Tuple(t) => {
                let mut parts = Vec::with_capacity(t.len());
                for x in t.iter() {
                    parts.push(self.repr(x)?);
                }
                if parts.len() == 1 {
                    format!("({},)", parts[0])
                } else {
                    format!("({})", parts.join(", "))
                }
            }
            Value::Map(m) => {
                let entries: Vec<(Value, Value)> = m
                    .borrow()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                let mut parts = Vec::with_capacity(entries.len());
                for (k, v) in &entries {
                    parts.push(format!("{}: {}", self.repr(k)?, self.repr(v)?));
                }
                format!("{{{}}}", parts.join(", "))
            }
            Value::Set(m) => {
                let keys = m.borrow().keys();
                if keys.is_empty() {
                    return Ok("set()".into());
                }
                let mut parts = Vec::with_capacity(keys.len());
                for k in &keys {
                    parts.push(self.repr(k)?);
                }
                format!("{{{}}}", parts.join(", "))
            }
            Value::Variant(var) => {
                let fields = var.fields.clone();
                let mut shown = Vec::with_capacity(fields.len());
                for f in &fields {
                    shown.push(self.repr(f)?);
                }
                let mut i = 0;
                variant_text(var, |_| {
                    let s = shown.get(i).cloned().unwrap_or_default();
                    i += 1;
                    s
                })
            }
            Value::Instance(inst) => {
                if inst.class.has_ancestor_named("Error") {
                    return Ok(display_plain(v, false));
                }
                let fields = inst.fields.borrow().clone();
                let mut parts = Vec::with_capacity(fields.len());
                for (k, x) in &fields {
                    parts.push(format!("{k}={}", self.repr(x)?));
                }
                format!("{}({})", inst.class.name, parts.join(", "))
            }
            other => display_plain(other, false),
        })
    }

    /// Applies a Python format spec (`>10`, `.2f`, `,`, `08.3f`, `x`, `%`).
    pub fn format_spec(&mut self, v: &Value, spec: &str, span: Span) -> R<String> {
        if let Some(rest) = spec.strip_prefix("!r") {
            let r = self.repr(v)?;
            return if rest.is_empty() {
                Ok(r)
            } else {
                self.format_spec(&Value::str(&r), rest, span)
            };
        }
        let s = parse_spec(spec);
        let numeric = v.is_number() || matches!(v, Value::Bool(_));
        let body = match s.kind {
            Some('d') | Some('n') => match v {
                Value::Int(_) | Value::Big(_) | Value::Bool(_) => display_plain(v, false),
                Value::Float(f) => format!("{}", f.round() as i64),
                other => {
                    let t = other.type_name();
                    return self
                        .value_err(format!("format code `d` needs an integer, got {t}"), span);
                }
            },
            Some(k @ ('f' | 'F' | 'e' | 'E' | 'g' | 'G' | '%')) => {
                let Some(f) = v.as_f64() else {
                    let t = v.type_name();
                    return self
                        .value_err(format!("format code `{k}` needs a number, got {t}"), span);
                };
                let p = s.precision.unwrap_or(6);
                match k {
                    'f' | 'F' => format!("{f:.p$}"),
                    'e' | 'E' => {
                        let r = format!("{f:.p$e}");
                        let r = match r.split_once('e') {
                            Some((m, e)) => {
                                let (sign, d) = match e.strip_prefix('-') {
                                    Some(d) => ("-", d),
                                    None => ("+", e),
                                };
                                format!("{m}e{sign}{d:0>2}")
                            }
                            None => r,
                        };
                        if k == 'E' { r.to_uppercase() } else { r }
                    }
                    '%' => format!("{:.p$}%", f * 100.0),
                    _ => {
                        // General format: precision significant digits.
                        let p = p.max(1);
                        if f != 0.0 && (f.abs() < 1e-4 || f.abs() >= 10f64.powi(p as i32)) {
                            format!("{:.*e}", p - 1, f)
                        } else {
                            let digits =
                                (p as i32 - 1 - f.abs().log10().floor() as i32).max(0) as usize;
                            let r = format!("{f:.digits$}");
                            if r.contains('.') {
                                r.trim_end_matches('0').trim_end_matches('.').to_owned()
                            } else {
                                r
                            }
                        }
                    }
                }
            }
            Some(k @ ('x' | 'X' | 'o' | 'b')) => {
                let Some(i) = v.as_int() else {
                    return self.value_err(format!("format code `{k}` needs an integer"), span);
                };
                let (prefix, digits) = match k {
                    'x' => ("0x", format!("{i:x}")),
                    'X' => ("0X", format!("{i:X}")),
                    'o' => ("0o", format!("{i:o}")),
                    _ => ("0b", format!("{i:b}")),
                };
                if s.alt {
                    format!("{prefix}{digits}")
                } else {
                    digits
                }
            }
            Some('c') => match v.as_int().and_then(|i| char::from_u32(i as u32)) {
                Some(c) => c.to_string(),
                None => self.to_str(v)?,
            },
            Some('s') | None => {
                if let (Some(p), Value::Float(f)) = (s.precision, v) {
                    format!("{f:.p$}")
                } else if let (Some(p), false) = (s.precision, numeric) {
                    self.to_str(v)?.chars().take(p).collect()
                } else {
                    self.to_str(v)?
                }
            }
            Some('?') => self.repr(v)?,
            Some(other) => return self.value_err(format!("unknown format code `{other}`"), span),
        };
        let mut body = body;
        if let Some(sep) = s.grouping
            && numeric
        {
            body = group_digits(&body, sep);
        }
        if numeric && !body.starts_with('-') {
            match s.sign {
                Some('+') => body.insert(0, '+'),
                Some(' ') => body.insert(0, ' '),
                _ => {}
            }
        }
        Ok(pad(body, &s, numeric))
    }

    /// C/Go-style `printf` formatting.
    pub fn printf(&mut self, fmt: &str, args: &[Value], span: Span) -> R<String> {
        let mut out = String::new();
        let chars: Vec<char> = fmt.chars().collect();
        let mut i = 0;
        let mut next = 0;
        while i < chars.len() {
            let c = chars[i];
            if c != '%' {
                out.push(c);
                i += 1;
                continue;
            }
            i += 1;
            if i < chars.len() && chars[i] == '%' {
                out.push('%');
                i += 1;
                continue;
            }
            // Flags, width, precision, length modifiers, verb.
            let mut flags = String::new();
            while i < chars.len() && matches!(chars[i], '-' | '+' | ' ' | '0' | '#') {
                flags.push(chars[i]);
                i += 1;
            }
            let mut width = String::new();
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '*') {
                if chars[i] == '*' {
                    let w = args.get(next).and_then(Value::as_int).unwrap_or(0);
                    next += 1;
                    width = w.to_string();
                } else {
                    width.push(chars[i]);
                }
                i += 1;
            }
            let mut precision = None;
            if i < chars.len() && chars[i] == '.' {
                i += 1;
                let mut p = String::new();
                while i < chars.len() && chars[i].is_ascii_digit() {
                    p.push(chars[i]);
                    i += 1;
                }
                precision = Some(p.parse::<usize>().unwrap_or(0));
            }
            while i < chars.len() && matches!(chars[i], 'l' | 'h' | 'z' | 'j' | 't' | 'L' | 'q') {
                i += 1;
            }
            let Some(&verb) = chars.get(i) else {
                out.push('%');
                break;
            };
            i += 1;
            let arg = args.get(next).cloned().unwrap_or(Value::Nil);
            next += 1;
            let mut spec = String::new();
            if flags.contains('-') {
                spec.push('<');
            } else if !flags.contains('0') {
                spec.push('>');
            }
            if flags.contains('+') {
                spec.push('+');
            } else if flags.contains(' ') {
                spec.push(' ');
            }
            if flags.contains('#') {
                spec.push('#');
            }
            if flags.contains('0') && !flags.contains('-') {
                spec.push('0');
            }
            spec.push_str(&width);
            let text = match verb {
                'd' | 'i' | 'u' => {
                    let v = match &arg {
                        Value::Float(f) => Value::Int(f.trunc() as i64),
                        other => other.clone(),
                    };
                    self.format_spec(&v, &format!("{spec}d"), span)?
                }
                'f' | 'F' | 'e' | 'E' | 'g' | 'G' => {
                    let p = precision.unwrap_or(6);
                    self.format_spec(&arg, &format!("{spec}.{p}{verb}"), span)?
                }
                'x' | 'X' | 'o' | 'b' => self.format_spec(&arg, &format!("{spec}{verb}"), span)?,
                'c' => self.format_spec(&arg, &format!("{spec}c"), span)?,
                't' => self.format_spec(
                    &Value::str(if arg.truthy() { "true" } else { "false" }),
                    &spec,
                    span,
                )?,
                'T' => self.format_spec(&Value::str(&arg.type_name()), &spec, span)?,
                'q' => {
                    let r = self.repr(&arg)?;
                    self.format_spec(&Value::str(&r), &spec, span)?
                }
                'p' => format!("0x{:x}", ptr_of(&arg)),
                _ => {
                    // %s, %v, %+v, %@ (Objective-C) and anything else.
                    let s = self.to_str(&arg)?;
                    let s = match precision {
                        Some(p) => s.chars().take(p).collect(),
                        None => s,
                    };
                    self.format_spec(&Value::str(&s), &spec, span)?
                }
            };
            out.push_str(&text);
        }
        Ok(out)
    }

    /// Rust/Python brace templates: `{}`, `{0}`, `{name}`, `{:.2}`, `{x:>8}`,
    /// `{:?}`. Names are looked up in `named`, then via `lookup`.
    pub fn brace_format(
        &mut self,
        template: &str,
        pos: &[Value],
        named: &[(crate::symbol::Sym, Value)],
        lookup: Option<&Rc<super::Env>>,
        span: Span,
    ) -> R<String> {
        let mut out = String::new();
        let chars: Vec<char> = template.chars().collect();
        let mut i = 0;
        let mut auto = 0;
        while i < chars.len() {
            let c = chars[i];
            if c == '{' {
                if chars.get(i + 1) == Some(&'{') {
                    out.push('{');
                    i += 2;
                    continue;
                }
                let Some(close) = chars[i..].iter().position(|&c| c == '}') else {
                    out.push('{');
                    i += 1;
                    continue;
                };
                let inner: String = chars[i + 1..i + close].iter().collect();
                i += close + 1;
                let (field, spec) = match inner.split_once(':') {
                    Some((f, s)) => (f.trim().to_owned(), Some(s.to_owned())),
                    None => (inner.trim().to_owned(), None),
                };
                let value = if field.is_empty() {
                    let v = pos.get(auto).cloned();
                    auto += 1;
                    v
                } else if let Ok(n) = field.parse::<usize>() {
                    pos.get(n).cloned()
                } else {
                    let (base, attr) = match field.split_once('.') {
                        Some((b, a)) => (b, Some(a)),
                        None => (field.as_str(), None),
                    };
                    let mut v = named
                        .iter()
                        .find(|(n, _)| n.is(base))
                        .map(|(_, v)| v.clone())
                        .or_else(|| lookup.and_then(|env| env.get(sym(base))));
                    if let (Some(obj), Some(attr)) = (&v, attr) {
                        v = Some(self.get_attr(obj, sym(attr), span)?);
                    }
                    v
                };
                let Some(value) = value else {
                    return Err(self.error(
                        "KeyError",
                        format!("no value for `{{{field}}}` in the format string"),
                        span,
                    ));
                };
                let text = match spec.as_deref() {
                    Some("?") | Some("#?") => self.repr(&value)?,
                    Some(s) => {
                        // Rust's `{:.2}` means fixed precision for floats.
                        let s = if s.starts_with('.')
                            && !s.ends_with(|c: char| c.is_alphabetic())
                            && value.as_f64().is_some()
                            && matches!(value, Value::Float(_))
                        {
                            format!("{s}f")
                        } else {
                            s.to_owned()
                        };
                        self.format_spec(&value, &s, span)?
                    }
                    None => self.to_str(&value)?,
                };
                out.push_str(&text);
            } else if c == '}' && chars.get(i + 1) == Some(&'}') {
                out.push('}');
                i += 2;
            } else {
                out.push(c);
                i += 1;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_like_python() {
        assert_eq!(fmt_float(5.0), "5.0");
        assert_eq!(fmt_float(0.1 + 0.2), "0.30000000000000004");
        assert_eq!(fmt_float(1e21), "1e+21");
        assert_eq!(fmt_float(1.5e-7), "1.5e-07");
        assert_eq!(fmt_float(-2.5), "-2.5");
        assert_eq!(fmt_float(f64::INFINITY), "inf");
    }

    #[test]
    fn digit_grouping() {
        assert_eq!(group_digits("1234567", ','), "1,234,567");
        assert_eq!(group_digits("-1234.5", ','), "-1,234.5");
        assert_eq!(group_digits("12", ','), "12");
    }
}
