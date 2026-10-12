//! JSON parsing and serialisation (`JSON.parse`, `json.dumps`, …).

use std::cell::RefCell;
use std::rc::Rc;

use super::objects::module as make_module;
use crate::interp::Interp;
use crate::value::*;

/// Parses JSON text into Chimera values.
pub fn parse(text: &str) -> Result<Value, String> {
    let mut p = JsonParser {
        s: text.as_bytes(),
        i: 0,
    };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(p.err("unexpected trailing characters"));
    }
    Ok(v)
}

struct JsonParser<'a> {
    s: &'a [u8],
    i: usize,
}

impl JsonParser<'_> {
    fn err(&self, msg: &str) -> String {
        let line = self.s[..self.i.min(self.s.len())]
            .iter()
            .filter(|&&b| b == b'\n')
            .count()
            + 1;
        let col = self.s[..self.i.min(self.s.len())]
            .iter()
            .rev()
            .take_while(|&&b| b != b'\n')
            .count()
            + 1;
        format!("invalid JSON at line {line}, column {col}: {msg}")
    }

    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\n' | b'\r' | b'\t') {
            self.i += 1;
        }
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.s.get(self.i) == Some(&b) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Value, String> {
        self.ws();
        match self.s.get(self.i) {
            Some(b'{') => {
                self.i += 1;
                let mut d = Dict::new();
                self.ws();
                if self.eat(b'}') {
                    return Ok(Value::Map(Rc::new(RefCell::new(d))));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.ws();
                    if !self.eat(b':') {
                        return Err(self.err("expected `:`"));
                    }
                    let v = self.value()?;
                    d.insert(Value::str(&k), v);
                    self.ws();
                    if self.eat(b',') {
                        continue;
                    }
                    if self.eat(b'}') {
                        break;
                    }
                    return Err(self.err("expected `,` or `}`"));
                }
                Ok(Value::Map(Rc::new(RefCell::new(d))))
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat(b']') {
                    return Ok(Value::list(items));
                }
                loop {
                    items.push(self.value()?);
                    self.ws();
                    if self.eat(b',') {
                        continue;
                    }
                    if self.eat(b']') {
                        break;
                    }
                    return Err(self.err("expected `,` or `]`"));
                }
                Ok(Value::list(items))
            }
            Some(b'"') => Ok(Value::str(&self.string()?)),
            Some(b't') if self.s[self.i..].starts_with(b"true") => {
                self.i += 4;
                Ok(Value::Bool(true))
            }
            Some(b'f') if self.s[self.i..].starts_with(b"false") => {
                self.i += 5;
                Ok(Value::Bool(false))
            }
            Some(b'n') if self.s[self.i..].starts_with(b"null") => {
                self.i += 4;
                Ok(Value::Nil)
            }
            Some(b'-' | b'0'..=b'9') => {
                let start = self.i;
                self.i += 1;
                let mut float = false;
                while let Some(&c) = self.s.get(self.i) {
                    match c {
                        b'0'..=b'9' => {}
                        b'.' | b'e' | b'E' | b'+' | b'-' => float = true,
                        _ => break,
                    }
                    self.i += 1;
                }
                let text = std::str::from_utf8(&self.s[start..self.i])
                    .map_err(|_| self.err("bad number"))?;
                if !float && let Ok(i) = text.parse::<i64>() {
                    return Ok(Value::Int(i));
                }
                text.parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| self.err("bad number"))
            }
            Some(_) => Err(self.err("unexpected character")),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        if !self.eat(b'"') {
            return Err(self.err("expected a string"));
        }
        let mut out = Vec::new();
        loop {
            let Some(&c) = self.s.get(self.i) else {
                return Err(self.err("unterminated string"));
            };
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let Some(&e) = self.s.get(self.i) else {
                        return Err(self.err("bad escape"));
                    };
                    self.i += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'/' => out.push(b'/'),
                        b'\\' => out.push(b'\\'),
                        b'"' => out.push(b'"'),
                        b'u' => {
                            let hex =
                                std::str::from_utf8(self.s.get(self.i..self.i + 4).unwrap_or(b""))
                                    .map_err(|_| self.err("bad \\u escape"))?;
                            let mut code = u32::from_str_radix(hex, 16)
                                .map_err(|_| self.err("bad \\u escape"))?;
                            self.i += 4;
                            // Surrogate pairs.
                            if (0xD800..0xDC00).contains(&code)
                                && self.s[self.i..].starts_with(b"\\u")
                            {
                                let hex2 = std::str::from_utf8(
                                    self.s.get(self.i + 2..self.i + 6).unwrap_or(b""),
                                )
                                .map_err(|_| self.err("bad \\u escape"))?;
                                if let Ok(low) = u32::from_str_radix(hex2, 16) {
                                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                                    self.i += 6;
                                }
                            }
                            let ch = char::from_u32(code).unwrap_or('\u{fffd}');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err(self.err("bad escape")),
                    }
                }
                other => out.push(other),
            }
        }
        String::from_utf8(out).map_err(|_| self.err("invalid UTF-8"))
    }
}

/// Serialises a value as JSON.
pub fn stringify(interp: &mut Interp, v: &Value, indent: Option<usize>, depth: usize) -> R<String> {
    stringify_with(interp, v, indent, depth, false)
}

/// Like [`stringify()`]; `spaced` uses Python's `", "` and `": "` separators.
pub fn stringify_with(
    interp: &mut Interp,
    v: &Value,
    indent: Option<usize>,
    depth: usize,
    spaced: bool,
) -> R<String> {
    if depth > 512 {
        return Err(interp.error(
            "ValueError",
            "value is too deeply nested (or circular) for JSON",
            Default::default(),
        ));
    }
    let nl = |d: usize| match indent {
        Some(n) => format!("\n{}", " ".repeat(n * d)),
        None => String::new(),
    };
    let sep = if indent.is_some() || spaced {
        ": "
    } else {
        ":"
    };
    let comma = if spaced && indent.is_none() {
        ", "
    } else {
        ","
    };
    Ok(match v {
        Value::Nil => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Big(b) => b.to_string(),
        Value::Float(f) => {
            if f.is_finite() {
                crate::interp::format::fmt_float(*f)
            } else {
                "null".into()
            }
        }
        Value::Str(s) => quote(s),
        Value::List(_) | Value::Tuple(_) | Value::Set(_) | Value::Range(_) => {
            let items = interp.collect(v, Default::default())?;
            if items.is_empty() {
                return Ok("[]".into());
            }
            let mut parts = Vec::with_capacity(items.len());
            for x in &items {
                parts.push(format!(
                    "{}{}",
                    nl(depth + 1),
                    stringify_with(interp, x, indent, depth + 1, spaced)?
                ));
            }
            format!("[{}{}]", parts.join(comma), nl(depth))
        }
        Value::Map(m) => {
            let entries: Vec<(Value, Value)> = m
                .borrow()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            if entries.is_empty() {
                return Ok("{}".into());
            }
            let mut parts = Vec::with_capacity(entries.len());
            for (k, x) in &entries {
                let key = interp.to_str(k)?;
                parts.push(format!(
                    "{}{}{sep}{}",
                    nl(depth + 1),
                    quote(&key),
                    stringify_with(interp, x, indent, depth + 1, spaced)?
                ));
            }
            format!("{{{}{}}}", parts.join(comma), nl(depth))
        }
        Value::Instance(i) => {
            let fields = i.fields.borrow().clone();
            let mut d = Dict::new();
            for (k, x) in fields {
                d.insert(Value::Str(k.as_str()), x);
            }
            stringify_with(
                interp,
                &Value::Map(Rc::new(RefCell::new(d))),
                indent,
                depth,
                spaced,
            )?
        }
        Value::Variant(var) => {
            if var.fields.is_empty() {
                quote(&var.name().as_str())
            } else {
                let inner = Value::list(var.fields.clone());
                let body = stringify_with(interp, &inner, indent, depth + 1, spaced)?;
                format!(
                    "{{{}{}{sep}{body}{}}}",
                    nl(depth + 1),
                    quote(&var.name().as_str()),
                    nl(depth)
                )
            }
        }
        Value::Tensor(t) => {
            let as_list = crate::ml::tensor_to_list(t);
            stringify_with(interp, &as_list, indent, depth, spaced)?
        }
        other => quote(&interp.to_str(other)?),
    })
}

pub(crate) fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The `json` / `JSON` module.
pub fn module(_interp: &mut Interp) -> Value {
    let parse_fn = Value::native("json.parse", |interp, args| {
        let s = super::str_arg(interp, &args, 0, "json.parse")?;
        parse(&s).map_err(|e| interp.error("ValueError", e, args.span))
    });
    let dumps_fn = Value::native("json.dumps", |interp, args| {
        let v = super::arg(interp, &args, 0, "json.dumps")?;
        let indent = args.named("indent").and_then(|i| match i {
            Value::Str(s) => Some(s.len()),
            other => other.as_int().map(|n| n.max(0) as usize),
        });
        let s = stringify_with(interp, &v, indent, 0, true)?;
        Ok(Value::str(&s))
    });
    let dump_fn = Value::native("json.dumps", |interp, args| {
        let v = super::arg(interp, &args, 0, "json.dumps")?;
        let indent = args
            .named("indent")
            .or_else(|| args.get(2).cloned())
            .and_then(|i| match i {
                Value::Str(s) => Some(s.len()),
                other => other.as_int().map(|n| n.max(0) as usize),
            });
        let s = stringify(interp, &v, indent, 0)?;
        Ok(Value::str(&s))
    });
    make_module(
        "json",
        vec![
            ("parse", parse_fn.clone()),
            ("loads", parse_fn.clone()),
            ("decode", parse_fn.clone()),
            ("Unmarshal", parse_fn),
            ("stringify", dump_fn.clone()),
            ("dumps", dumps_fn),
            ("encode", dump_fn.clone()),
            ("Marshal", dump_fn.clone()),
            ("generate", dump_fn),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roundtrip_shapes() {
        let v = parse(r#"{"a": [1, 2.5, "x\n", true, null], "b": {"c": -3}}"#).unwrap();
        let Value::Map(m) = v else { panic!() };
        let m = m.borrow();
        assert!(matches!(m.get_str("a"), Some(Value::List(_))));
        assert!(matches!(m.get_str("b"), Some(Value::Map(_))));
        assert!(parse("[1, 2").is_err());
        assert!(parse("{\"a\" 1}").unwrap_err().contains("line 1"));
        assert!(
            matches!(parse("\"\\u00e9\\ud83d\\ude00\"").unwrap(), Value::Str(s) if &*s == "é😀")
        );
    }
}
