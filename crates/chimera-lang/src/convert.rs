//! Source-to-source translation: any Chimera program, written in any mix of
//! dialects, rendered as Python, JavaScript, Ruby or Lua (or as canonical
//! Chimera).
//!
//! The emitter walks the syntax tree and prints each construct in the
//! target's own idiom: blocks become indentation, braces or `end`, lambdas
//! become `lambda`/`=>`/`->`/`function`, interpolated strings become
//! f-strings, template literals, `#{}` or `..` concatenation, and so on.
//! Constructs a target has no equivalent for (SQL queries, `please`,
//! derivatives, `model`/`train`) are carried over verbatim inside a
//! marker comment so nothing is silently lost.

use std::fmt::Write as _;

use crate::ast::*;
use crate::symbol::Sym;

/// A target language.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Canonical Chimera (Python-flavoured).
    Chimera,
    /// Python 3.
    Python,
    /// Modern JavaScript.
    JavaScript,
    /// Ruby.
    Ruby,
    /// Lua 5.4.
    Lua,
}

impl Target {
    /// Parses a target name (`py`, `python`, `js`, `ruby`, `lua`, …).
    pub fn from_name(s: &str) -> Option<Target> {
        Some(match s.to_ascii_lowercase().as_str() {
            "chimera" | "chi" | "canonical" => Target::Chimera,
            "python" | "py" | "python3" => Target::Python,
            "javascript" | "js" | "node" | "ecmascript" => Target::JavaScript,
            "ruby" | "rb" => Target::Ruby,
            "lua" => Target::Lua,
            _ => return None,
        })
    }

    /// The usual file extension.
    pub fn extension(self) -> &'static str {
        match self {
            Target::Chimera => "chi",
            Target::Python => "py",
            Target::JavaScript => "js",
            Target::Ruby => "rb",
            Target::Lua => "lua",
        }
    }

    fn comment(self) -> &'static str {
        match self {
            Target::JavaScript => "//",
            Target::Lua => "--",
            _ => "#",
        }
    }

    fn pythonic(self) -> bool {
        matches!(self, Target::Python | Target::Chimera)
    }
}

/// Translates a parsed program. `src` is the program text (used to carry
/// untranslatable constructs over verbatim).
pub fn convert(stmts: &[Stmt], src: &str, target: Target) -> String {
    let mut e = Emitter {
        out: String::new(),
        indent: 0,
        target,
        src,
        tmp: 0,
        scopes: vec![Vec::new()],
    };
    e.prelude(stmts);
    for s in stmts {
        e.stmt(s);
    }
    e.out
}

struct Emitter<'a> {
    out: String,
    indent: usize,
    target: Target,
    src: &'a str,
    tmp: usize,
    /// Names declared in each enclosing function (JavaScript `let`, Lua `local`).
    scopes: Vec<Vec<String>>,
}

/// Precedence levels, loosest first.
mod prec {
    pub const LAMBDA: u8 = 1;
    pub const OR: u8 = 2;
    pub const AND: u8 = 3;
    pub const NOT: u8 = 4;
    pub const CMP: u8 = 5;
    pub const BITOR: u8 = 6;
    pub const BITXOR: u8 = 7;
    pub const BITAND: u8 = 8;
    pub const SHIFT: u8 = 9;
    pub const ADD: u8 = 10;
    pub const MUL: u8 = 11;
    pub const UNARY: u8 = 12;
    pub const POW: u8 = 13;
    pub const POSTFIX: u8 = 14;
    pub const ATOM: u8 = 15;
}

impl Emitter<'_> {
    // ----- output helpers -----------------------------------------------------

    fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.out.push_str(match self.target {
                Target::JavaScript | Target::Ruby | Target::Lua => "  ",
                _ => "    ",
            });
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn original(&self, span: chimera_core::Span) -> String {
        self.src
            .get(span.start..span.end)
            .unwrap_or("")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// An expression the target can't express, kept verbatim as a call to
    /// a `chimera(…)` placeholder.
    fn untranslatable(&self, span: chimera_core::Span, what: &str) -> String {
        let text = self.original(span);
        format!("chimera({}, {})", self.string(what), self.string(&text))
    }

    /// Whether `name` needs a declaration keyword here (first assignment in
    /// this function, JavaScript and Lua only).
    fn declare(&mut self, name: &str) -> bool {
        if !matches!(self.target, Target::JavaScript | Target::Lua) {
            return false;
        }
        if self.scopes.iter().any(|s| s.iter().any(|n| n == name)) {
            return false;
        }
        if let Some(top) = self.scopes.last_mut() {
            top.push(name.to_owned());
        }
        true
    }

    fn fresh(&mut self, base: &str) -> String {
        self.tmp += 1;
        format!("_{base}{}", self.tmp)
    }

    fn prelude(&mut self, stmts: &[Stmt]) {
        let c = self.target.comment();
        self.line(&format!(
            "{c} Translated from Chimera by `chimera convert`."
        ));
        if self.target == Target::Python && uses_math(stmts) {
            self.line("from math import *");
        }
        if self.target == Target::Lua {
            // Small helpers the translation relies on.
            self.line("local function __range(a, b, step) local t = {} for i = a, b - 1, step or 1 do t[#t + 1] = i end return t end");
            self.line("local function __map(t, f) local r = {} for i, v in ipairs(t) do r[i] = f(v) end return r end");
            self.line("local function __filter(t, f) local r = {} for _, v in ipairs(t) do if f(v) then r[#r + 1] = v end end return r end");
        }
        self.out.push('\n');
    }

    // ----- names and literals ---------------------------------------------------

    fn name(&self, n: Sym) -> String {
        let s = n.as_str();
        match (self.target, &*s) {
            (Target::JavaScript, "self") => "this".into(),
            (Target::Ruby, "this") | (Target::Lua, "this") => "self".into(),
            (Target::Python, "this") | (Target::Chimera, "this") => "self".into(),
            _ => s.to_string(),
        }
    }

    fn nil(&self) -> &'static str {
        match self.target {
            Target::Python | Target::Chimera => "None",
            Target::JavaScript => "null",
            _ => "nil",
        }
    }

    fn bool(&self, b: bool) -> &'static str {
        match (self.target, b) {
            (Target::Python | Target::Chimera, true) => "True",
            (Target::Python | Target::Chimera, false) => "False",
            (_, true) => "true",
            (_, false) => "false",
        }
    }

    fn string(&self, s: &str) -> String {
        let mut out = String::from("\"");
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\t' => out.push_str("\\t"),
                '\r' => out.push_str("\\r"),
                '#' if self.target == Target::Ruby => out.push_str("\\#"),
                '$' if self.target == Target::JavaScript => out.push('$'),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    fn interpolation(&mut self, parts: &[InterpPart]) -> String {
        match self.target {
            Target::Python | Target::Chimera => {
                let mut s = String::from("f\"");
                for p in parts {
                    match p {
                        InterpPart::Lit(t) => {
                            s.push_str(&escape_inner(t).replace('{', "{{").replace('}', "}}"))
                        }
                        InterpPart::Expr(e, spec) => {
                            let code = self.expr(e).0;
                            match spec.as_deref().filter(|s| !s.starts_with('\u{1}')) {
                                Some(spec) if spec.starts_with("!r") => {
                                    let rest = &spec[2..];
                                    if rest.is_empty() {
                                        let _ = write!(s, "{{{code}!r}}");
                                    } else {
                                        let _ = write!(s, "{{{code}!r:{rest}}}");
                                    }
                                }
                                Some(spec) => {
                                    let _ = write!(s, "{{{code}:{spec}}}");
                                }
                                None => {
                                    let _ = write!(s, "{{{code}}}");
                                }
                            }
                        }
                    }
                }
                s.push('"');
                s
            }
            Target::JavaScript => {
                let mut s = String::from("`");
                for p in parts {
                    match p {
                        InterpPart::Lit(t) => s.push_str(
                            &t.replace('\\', "\\\\")
                                .replace('`', "\\`")
                                .replace("${", "\\${"),
                        ),
                        InterpPart::Expr(e, _) => {
                            let code = self.expr(e).0;
                            let _ = write!(s, "${{{code}}}");
                        }
                    }
                }
                s.push('`');
                s
            }
            Target::Ruby => {
                let mut s = String::from("\"");
                for p in parts {
                    match p {
                        InterpPart::Lit(t) => s.push_str(&escape_inner(t).replace('#', "\\#")),
                        InterpPart::Expr(e, _) => {
                            let code = self.expr(e).0;
                            let _ = write!(s, "#{{{code}}}");
                        }
                    }
                }
                s.push('"');
                s
            }
            Target::Lua => {
                let mut pieces = Vec::new();
                for p in parts {
                    match p {
                        InterpPart::Lit(t) if t.is_empty() => {}
                        InterpPart::Lit(t) => pieces.push(self.string(t)),
                        InterpPart::Expr(e, _) => {
                            let code = self.expr(e).0;
                            pieces.push(format!("tostring({code})"));
                        }
                    }
                }
                if pieces.is_empty() {
                    "\"\"".into()
                } else {
                    format!("({})", pieces.join(" .. "))
                }
            }
        }
    }

    // ----- expressions --------------------------------------------------------------

    fn sub(&mut self, e: &Expr, min: u8) -> String {
        let (s, p) = self.expr(e);
        if p < min { format!("({s})") } else { s }
    }

    fn list_items(&mut self, items: &[Expr]) -> String {
        items
            .iter()
            .map(|x| self.sub(x, prec::LAMBDA))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn args(&mut self, args: &[Arg]) -> String {
        let mut out = Vec::with_capacity(args.len());
        let mut named = Vec::new();
        for a in args {
            match a {
                Arg::Pos(e) => out.push(self.sub(e, prec::LAMBDA)),
                Arg::Spread(e) => {
                    let s = self.sub(e, prec::POSTFIX);
                    out.push(match self.target {
                        Target::JavaScript => format!("...{s}"),
                        Target::Lua => format!("table.unpack({s})"),
                        _ => format!("*{s}"),
                    });
                }
                Arg::Named(n, e) => {
                    let v = self.sub(e, prec::LAMBDA);
                    match self.target {
                        Target::Python | Target::Chimera => out.push(format!("{n}={v}")),
                        Target::Ruby => out.push(format!("{n}: {v}")),
                        _ => named.push(format!("{n}: {v}")),
                    }
                }
                Arg::KwSpread(e) => {
                    let s = self.sub(e, prec::POSTFIX);
                    out.push(match self.target {
                        Target::JavaScript => format!("...{s}"),
                        Target::Ruby => format!("**{s}"),
                        _ => format!("**{s}"),
                    });
                }
            }
        }
        if !named.is_empty() {
            // JavaScript and Lua take named arguments as a trailing object.
            let obj = match self.target {
                Target::Lua => format!(
                    "{{{}}}",
                    named
                        .iter()
                        .map(|n| n.replacen(": ", " = ", 1))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                _ => format!("{{ {} }}", named.join(", ")),
            };
            out.push(obj);
        }
        out.join(", ")
    }

    fn expr(&mut self, e: &Expr) -> (String, u8) {
        use prec::*;
        let t = self.target;
        match &e.kind {
            ExprKind::Nil => (self.nil().into(), ATOM),
            ExprKind::Bool(b) => (self.bool(*b).into(), ATOM),
            ExprKind::Int(i) => (i.to_string(), if *i < 0 { UNARY } else { ATOM }),
            ExprKind::BigInt(b) => (
                match t {
                    Target::JavaScript => format!("{b}n"),
                    _ => b.to_string(),
                },
                ATOM,
            ),
            ExprKind::Float(f) => {
                let s = crate::interp::format::fmt_float(*f);
                let s = match (t, s.as_str()) {
                    (Target::Python | Target::Chimera, "inf") => "float(\"inf\")".into(),
                    (Target::JavaScript, "inf") => "Infinity".into(),
                    (Target::Ruby, "inf") => "Float::INFINITY".into(),
                    (Target::Lua, "inf") => "math.huge".into(),
                    _ => s,
                };
                (s, ATOM)
            }
            ExprKind::Str(s) => (self.string(s), ATOM),
            ExprKind::Interp(parts) => (self.interpolation(parts), ATOM),
            ExprKind::Name(n) => (self.name(*n), ATOM),
            ExprKind::SelfRef => (
                match t {
                    Target::JavaScript => "this",
                    _ => "self",
                }
                .into(),
                ATOM,
            ),
            ExprKind::Super => ("super".into(), ATOM),
            ExprKind::List(items) | ExprKind::Tuple(items) if t == Target::Lua => {
                (format!("{{{}}}", self.list_items(items)), ATOM)
            }
            ExprKind::List(items) => (format!("[{}]", self.list_items(items)), ATOM),
            ExprKind::Tuple(items) => match t {
                Target::Python | Target::Chimera if items.len() == 1 => {
                    (format!("({},)", self.list_items(items)), ATOM)
                }
                Target::Python | Target::Chimera => (format!("({})", self.list_items(items)), ATOM),
                _ => (format!("[{}]", self.list_items(items)), ATOM),
            },
            ExprKind::Set(items) => match t {
                Target::Python | Target::Chimera if items.is_empty() => ("set()".into(), POSTFIX),
                Target::Python | Target::Chimera => {
                    (format!("{{{}}}", self.list_items(items)), ATOM)
                }
                Target::JavaScript => (format!("new Set([{}])", self.list_items(items)), POSTFIX),
                Target::Ruby => (format!("Set[{}]", self.list_items(items)), POSTFIX),
                Target::Lua => {
                    let entries: Vec<String> = items
                        .iter()
                        .map(|x| format!("[{}] = true", self.sub(x, LAMBDA)))
                        .collect();
                    (format!("{{{}}}", entries.join(", ")), ATOM)
                }
            },
            ExprKind::Map(entries) => {
                let mut parts = Vec::with_capacity(entries.len());
                for (k, v) in entries {
                    if let ExprKind::Spread(inner) = &k.kind {
                        let s = self.sub(inner, POSTFIX);
                        parts.push(match t {
                            Target::JavaScript => format!("...{s}"),
                            _ => format!("**{s}"),
                        });
                        continue;
                    }
                    let val = self.sub(v, LAMBDA);
                    let key = match (&k.kind, t) {
                        (ExprKind::Str(s), Target::JavaScript) if is_ident(s) => s.to_string(),
                        (ExprKind::Str(s), Target::Lua) if is_ident(s) => s.to_string(),
                        (ExprKind::Str(s), Target::Ruby) if is_ident(s) => format!("{s}:"),
                        (_, Target::JavaScript)
                            if !matches!(k.kind, ExprKind::Str(_) | ExprKind::Int(_)) =>
                        {
                            format!("[{}]", self.sub(k, LAMBDA))
                        }
                        (_, Target::Lua) => format!("[{}]", self.sub(k, LAMBDA)),
                        _ => self.sub(k, LAMBDA),
                    };
                    parts.push(match t {
                        Target::Lua => format!("{key} = {val}"),
                        Target::Ruby if key.ends_with(':') => format!("{key} {val}"),
                        Target::Ruby => format!("{key} => {val}"),
                        _ => format!("{key}: {val}"),
                    });
                }
                (format!("{{{}}}", parts.join(", ")), ATOM)
            }
            ExprKind::Spread(inner) => {
                let s = self.sub(inner, POSTFIX);
                (
                    match t {
                        Target::JavaScript => format!("...{s}"),
                        Target::Lua => format!("table.unpack({s})"),
                        _ => format!("*{s}"),
                    },
                    UNARY,
                )
            }
            ExprKind::Range {
                start,
                end,
                inclusive,
                step,
            } => self.range(
                start.as_deref(),
                end.as_deref(),
                *inclusive,
                step.as_deref(),
            ),
            ExprKind::Unary(op, x) => self.unary(*op, x),
            ExprKind::Binary(op, a, b) => self.binary(*op, a, b),
            ExprKind::Compare(first, rest) => self.compare(first, rest),
            ExprKind::And(a, b) => {
                let (a, b) = (self.sub(a, AND), self.sub(b, AND + 1));
                let op = match t {
                    Target::JavaScript | Target::Ruby => "&&",
                    _ => "and",
                };
                (format!("{a} {op} {b}"), AND)
            }
            ExprKind::Or(a, b) => {
                let (a, b) = (self.sub(a, OR), self.sub(b, OR + 1));
                let op = match t {
                    Target::JavaScript | Target::Ruby => "||",
                    _ => "or",
                };
                (format!("{a} {op} {b}"), OR)
            }
            ExprKind::Coalesce(a, b) => {
                let (sa, sb) = (self.sub(a, POSTFIX), self.sub(b, OR + 1));
                match t {
                    Target::JavaScript => (format!("{sa} ?? {sb}"), OR),
                    Target::Python | Target::Chimera => {
                        (format!("{sa} if {sa} is not None else {sb}"), LAMBDA)
                    }
                    _ => (format!("{sa} or {sb}"), OR),
                }
            }
            ExprKind::Call { callee, args } => self.call(callee, args),
            ExprKind::Member { object, name, .. }
                if t == Target::Ruby && matches!(object.kind, ExprKind::SelfRef)
                    || t == Target::Ruby
                        && matches!(&object.kind, ExprKind::Name(n) if n.is("self") || n.is("this")) =>
            {
                // Ruby instance variables.
                (format!("@{name}"), ATOM)
            }
            ExprKind::Member {
                object,
                name,
                optional,
            } => {
                let o = self.sub(object, POSTFIX);
                let dot = match (t, optional) {
                    (Target::JavaScript, true) => "?.",
                    (Target::Ruby, true) => "&.",
                    _ => ".",
                };
                (format!("{o}{dot}{name}"), POSTFIX)
            }
            ExprKind::Index { object, index } => {
                let o = self.sub(object, POSTFIX);
                if let ExprKind::Slice { start, end, step } = &index.kind {
                    return (
                        self.slice(&o, start.as_deref(), end.as_deref(), step.as_deref()),
                        POSTFIX,
                    );
                }
                let i = self.sub(index, LAMBDA);
                if t == Target::Lua && !matches!(index.kind, ExprKind::Str(_)) {
                    // Lua lists start at 1.
                    return (format!("{o}[({i}) + 1]"), POSTFIX);
                }
                (format!("{o}[{i}]"), POSTFIX)
            }
            ExprKind::Slice { start, end, step } => {
                let s = self.slice("", start.as_deref(), end.as_deref(), step.as_deref());
                (s, POSTFIX)
            }
            ExprKind::Lambda(def) => self.lambda(def),
            ExprKind::If {
                cond,
                then,
                otherwise,
            } => {
                let c = self.sub(cond, OR);
                let a = self.block_value(then);
                let b = match otherwise {
                    Some(o) => self.block_value(o),
                    None => self.nil().to_string(),
                };
                match t {
                    Target::Python | Target::Chimera => (format!("{a} if {c} else {b}"), LAMBDA),
                    Target::Lua => (format!("({c} and {a} or {b})"), ATOM),
                    _ => (format!("{c} ? {a} : {b}"), LAMBDA),
                }
            }
            ExprKind::Block(b) => (self.block_value(b), LAMBDA),
            ExprKind::Match { subject, arms } => self.match_value(subject.as_deref(), arms, e.span),
            ExprKind::Comprehension(c) => self.comprehension(c),
            ExprKind::Assign { target, value } => {
                let tgt = self.target_text(target);
                let v = self.sub(value, LAMBDA);
                (format!("{tgt} = {v}"), LAMBDA)
            }
            ExprKind::CompoundAssign { op, target, value } => {
                let tgt = self.target_text(target);
                let v = self.sub(value, LAMBDA);
                match (t, op) {
                    (Target::Lua, _) => {
                        let sym = binop_symbol(*op, t);
                        (format!("{tgt} = {tgt} {sym} {v}"), LAMBDA)
                    }
                    _ => (format!("{tgt} {}= {v}", binop_symbol(*op, t)), LAMBDA),
                }
            }
            ExprKind::IncDec {
                target,
                delta,
                prefix,
            } => {
                let tgt = self.target_text(target);
                match t {
                    Target::JavaScript => {
                        let op = if *delta > 0 { "++" } else { "--" };
                        (
                            if *prefix {
                                format!("{op}{tgt}")
                            } else {
                                format!("{tgt}{op}")
                            },
                            UNARY,
                        )
                    }
                    Target::Lua => {
                        let op = if *delta > 0 { "+" } else { "-" };
                        (format!("{tgt} = {tgt} {op} 1"), LAMBDA)
                    }
                    _ => {
                        let op = if *delta > 0 { "+=" } else { "-=" };
                        (format!("{tgt} {op} 1"), LAMBDA)
                    }
                }
            }
            ExprKind::Query(_) => (self.untranslatable(e.span, "query"), LAMBDA),
            ExprKind::Derivative { .. } | ExprKind::Gradient(_) => {
                (self.untranslatable(e.span, "derivative"), LAMBDA)
            }
            ExprKind::Please(_) => (self.untranslatable(e.span, "please"), LAMBDA),
            ExprKind::StructLit { ty, fields } => {
                let ty = self.sub(ty, POSTFIX);
                let parts: Vec<String> = fields
                    .iter()
                    .map(|(n, v)| (n, self.sub(v, LAMBDA)))
                    .map(|(n, v)| (n.to_string(), v))
                    .map(|(n, v)| match t {
                        Target::Python | Target::Chimera => format!("{n}={v}"),
                        Target::Ruby => format!("{n}: {v}"),
                        Target::Lua => format!("{n} = {v}"),
                        Target::JavaScript => format!("{n}: {v}"),
                    })
                    .collect();
                match t {
                    Target::JavaScript => {
                        (format!("new {ty}({{ {} }})", parts.join(", ")), POSTFIX)
                    }
                    Target::Ruby => (format!("{ty}.new({})", parts.join(", ")), POSTFIX),
                    Target::Lua => (format!("{ty}.new({{{}}})", parts.join(", ")), POSTFIX),
                    _ => (format!("{ty}({})", parts.join(", ")), POSTFIX),
                }
            }
            ExprKind::Let { pattern, value } => {
                let v = self.sub(value, LAMBDA);
                match (&**pattern, t) {
                    (Pattern::Bind(n), Target::Python | Target::Chimera) => {
                        (format!("({n} := {v}) is not None"), CMP)
                    }
                    (Pattern::Bind(n), _) => (format!("({n} = {v}) != {}", self.nil()), CMP),
                    _ => (self.untranslatable(e.span, "pattern test"), LAMBDA),
                }
            }
            ExprKind::Yield { values, delegate } => {
                let v = self.list_items(values);
                let kw = match (t, delegate) {
                    (Target::JavaScript, true) => "yield*",
                    (Target::Python | Target::Chimera, true) => "yield from",
                    (Target::Lua, _) => "coroutine.yield",
                    _ => "yield",
                };
                if t == Target::Lua {
                    (format!("{kw}({v})"), POSTFIX)
                } else if v.is_empty() {
                    (kw.into(), LAMBDA)
                } else {
                    (format!("{kw} {v}"), LAMBDA)
                }
            }
            ExprKind::Jump(s) => match &s.kind {
                StmtKind::Throw(x) => {
                    let v = self.sub(x, LAMBDA);
                    match t {
                        Target::JavaScript => (format!("(() => {{ throw {v}; }})()"), POSTFIX),
                        Target::Lua => (format!("error({v})"), POSTFIX),
                        _ => (format!("raise({v})"), POSTFIX),
                    }
                }
                _ => (self.untranslatable(e.span, "jump"), LAMBDA),
            },
        }
    }

    fn target_text(&mut self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Tuple(items) | ExprKind::List(items) => match self.target {
                Target::JavaScript => format!("[{}]", self.list_items(items)),
                _ => self.list_items(items),
            },
            _ => self.sub(e, prec::POSTFIX),
        }
    }

    fn range(
        &mut self,
        start: Option<&Expr>,
        end: Option<&Expr>,
        inclusive: bool,
        step: Option<&Expr>,
    ) -> (String, u8) {
        let s = start.map_or_else(|| "0".to_owned(), |x| self.sub(x, prec::ADD));
        let Some(end) = end else {
            return (format!("{s}.."), prec::LAMBDA);
        };
        let e = self.sub(end, prec::ADD);
        let st = step.map(|x| self.sub(x, prec::LAMBDA));
        let end_excl = if inclusive {
            format!("{e} + 1")
        } else {
            e.clone()
        };
        match self.target {
            Target::Python | Target::Chimera => match st {
                Some(st) => (format!("range({s}, {end_excl}, {st})"), prec::POSTFIX),
                None => (format!("range({s}, {end_excl})"), prec::POSTFIX),
            },
            Target::JavaScript => {
                let step = st.unwrap_or_else(|| "1".into());
                (
                    format!(
                        "Array.from({{ length: Math.ceil(({end_excl} - {s}) / {step}) }}, (_, i) => {s} + i * {step})"
                    ),
                    prec::POSTFIX,
                )
            }
            Target::Ruby => {
                let r = if inclusive {
                    format!("({s}..{e})")
                } else {
                    format!("({s}...{e})")
                };
                match st {
                    Some(st) => (format!("{r}.step({st})"), prec::POSTFIX),
                    None => (r, prec::ATOM),
                }
            }
            Target::Lua => (
                format!(
                    "__range({s}, {end_excl}, {})",
                    st.unwrap_or_else(|| "1".into())
                ),
                prec::POSTFIX,
            ),
        }
    }

    fn slice(
        &mut self,
        obj: &str,
        start: Option<&Expr>,
        end: Option<&Expr>,
        step: Option<&Expr>,
    ) -> String {
        let s = start.map(|x| self.sub(x, prec::LAMBDA));
        let e = end.map(|x| self.sub(x, prec::LAMBDA));
        match self.target {
            Target::Python | Target::Chimera => {
                let st = step.map(|x| self.sub(x, prec::LAMBDA));
                let mut inner = format!("{}:{}", s.unwrap_or_default(), e.unwrap_or_default());
                if let Some(st) = st {
                    inner.push(':');
                    inner.push_str(&st);
                }
                format!("{obj}[{inner}]")
            }
            Target::JavaScript => match e {
                Some(e) => format!("{obj}.slice({}, {e})", s.unwrap_or_else(|| "0".into())),
                None => format!("{obj}.slice({})", s.unwrap_or_else(|| "0".into())),
            },
            Target::Ruby => format!(
                "{obj}[{}...{}]",
                s.unwrap_or_else(|| "0".into()),
                e.unwrap_or_default()
            ),
            Target::Lua => format!(
                "table.move({obj}, ({}) + 1, {}, 1, {{}})",
                s.unwrap_or_else(|| "0".into()),
                e.unwrap_or_else(|| format!("#{obj}"))
            ),
        }
    }

    fn unary(&mut self, op: UnOp, x: &Expr) -> (String, u8) {
        let t = self.target;
        match op {
            UnOp::Neg => (format!("-{}", self.sub(x, prec::UNARY)), prec::UNARY),
            UnOp::Pos => (format!("+{}", self.sub(x, prec::UNARY)), prec::UNARY),
            UnOp::Not => match t {
                Target::JavaScript | Target::Ruby => {
                    (format!("!{}", self.sub(x, prec::UNARY)), prec::UNARY)
                }
                _ => (format!("not {}", self.sub(x, prec::NOT)), prec::NOT),
            },
            UnOp::BitNot => (format!("~{}", self.sub(x, prec::UNARY)), prec::UNARY),
            UnOp::Sqrt => {
                let v = self.sub(x, prec::LAMBDA);
                match t {
                    Target::JavaScript => (format!("Math.sqrt({v})"), prec::POSTFIX),
                    Target::Ruby => (format!("Math.sqrt({v})"), prec::POSTFIX),
                    Target::Lua => (format!("math.sqrt({v})"), prec::POSTFIX),
                    _ => (format!("sqrt({v})"), prec::POSTFIX),
                }
            }
            UnOp::TypeOf => {
                let v = self.sub(x, prec::LAMBDA);
                match t {
                    Target::JavaScript => {
                        (format!("typeof {}", self.sub(x, prec::UNARY)), prec::UNARY)
                    }
                    Target::Ruby => (
                        format!("{}.class", self.sub(x, prec::POSTFIX)),
                        prec::POSTFIX,
                    ),
                    Target::Lua => (format!("type({v})"), prec::POSTFIX),
                    _ => (format!("type({v})"), prec::POSTFIX),
                }
            }
            UnOp::Bang | UnOp::Try => self.expr(x),
        }
    }

    fn binary(&mut self, op: BinOp, a: &Expr, b: &Expr) -> (String, u8) {
        use prec::*;
        let t = self.target;
        let p = match op {
            BinOp::Add | BinOp::Sub | BinOp::Concat => ADD,
            BinOp::Mul | BinOp::Div | BinOp::FloorDiv | BinOp::Mod | BinOp::MatMul => MUL,
            BinOp::Pow => POW,
            BinOp::BitAnd => BITAND,
            BinOp::BitOr => BITOR,
            BinOp::BitXor => BITXOR,
            BinOp::Shl | BinOp::Shr => SHIFT,
            BinOp::Pipe => LAMBDA,
        };
        if op == BinOp::Pipe {
            // `x |> f` is `f(x)`.
            let x = self.sub(a, LAMBDA);
            let f = self.sub(b, POSTFIX);
            return (format!("{f}({x})"), POSTFIX);
        }
        if op == BinOp::FloorDiv && matches!(t, Target::JavaScript | Target::Lua) {
            let (x, y) = (self.sub(a, LAMBDA), self.sub(b, LAMBDA));
            return match t {
                Target::JavaScript => (format!("Math.floor({x} / {y})"), POSTFIX),
                _ => (format!("{x} // {y}"), MUL),
            };
        }
        if op == BinOp::Pow && t == Target::JavaScript {
            let (x, y) = (self.sub(a, POW + 1), self.sub(b, POW));
            return (format!("{x} ** {y}"), POW);
        }
        if op == BinOp::Concat && t == Target::Lua
            || op == BinOp::Add && t == Target::Lua && (is_stringish(a) || is_stringish(b))
        {
            let (x, y) = (self.sub(a, ADD), self.sub(b, ADD + 1));
            return (format!("{x} .. {y}"), ADD);
        }
        if op == BinOp::MatMul && t != Target::Python {
            let (x, y) = (self.sub(a, POSTFIX), self.sub(b, LAMBDA));
            return (format!("{x}.matmul({y})"), POSTFIX);
        }
        // Left-associative except `**`.
        let (lmin, rmin) = if op == BinOp::Pow {
            (p + 1, p)
        } else {
            (p, p + 1)
        };
        let x = self.sub(a, lmin);
        let y = self.sub(b, rmin);
        (format!("{x} {} {y}", binop_symbol(op, t)), p)
    }

    fn compare(&mut self, first: &Expr, rest: &[(CmpOp, Expr)]) -> (String, u8) {
        use prec::*;
        let t = self.target;
        // Chains `a < b < c` exist only in Python; elsewhere use `&&`.
        let mut parts = Vec::new();
        let mut left = first;
        for (op, right) in rest {
            let l = self.sub(left, CMP + 1);
            let r = self.sub(right, CMP + 1);
            let s = match (op, t) {
                (CmpOp::In, Target::Python | Target::Chimera) => format!("{l} in {r}"),
                (CmpOp::NotIn, Target::Python | Target::Chimera) => format!("{l} not in {r}"),
                (CmpOp::In, Target::JavaScript) => format!("{r}.includes({l})"),
                (CmpOp::NotIn, Target::JavaScript) => format!("!{r}.includes({l})"),
                (CmpOp::In, Target::Ruby) => format!("{r}.include?({l})"),
                (CmpOp::NotIn, Target::Ruby) => format!("!{r}.include?({l})"),
                (CmpOp::In | CmpOp::NotIn, Target::Lua) => format!("{r}[{l}] ~= nil"),
                (CmpOp::Is, Target::Python | Target::Chimera) => format!("{l} is {r}"),
                (CmpOp::IsNot, Target::Python | Target::Chimera) => format!("{l} is not {r}"),
                (CmpOp::Is | CmpOp::StrictEq, Target::JavaScript) => format!("{l} === {r}"),
                (CmpOp::IsNot | CmpOp::StrictNe, Target::JavaScript) => format!("{l} !== {r}"),
                (CmpOp::Ne | CmpOp::IsNot | CmpOp::StrictNe, Target::Lua) => format!("{l} ~= {r}"),
                (CmpOp::Spaceship, Target::Ruby) => format!("{l} <=> {r}"),
                (CmpOp::Approx, _) => format!("abs({l} - {r}) < 1e-9"),
                (op, _) => format!("{l} {} {r}", cmp_symbol(*op)),
            };
            parts.push(s);
            left = right;
        }
        if parts.len() == 1 {
            return (parts.pop().unwrap_or_default(), CMP);
        }
        let and = match t {
            Target::JavaScript | Target::Ruby => " && ",
            _ => " and ",
        };
        (parts.join(and), AND)
    }

    fn call(&mut self, callee: &Expr, args: &[Arg]) -> (String, u8) {
        let t = self.target;
        // Output functions map onto each target's idiom.
        if let ExprKind::Name(n) = &callee.kind {
            let name = n.as_str();
            let printish = matches!(
                &*name,
                "print"
                    | "println"
                    | "puts"
                    | "say"
                    | "echo"
                    | "display"
                    | "putStrLn"
                    | "print_endline"
            );
            if printish {
                let a = self.args(args);
                return match t {
                    Target::JavaScript => (format!("console.log({a})"), prec::POSTFIX),
                    // `puts a, b` prints two lines; join like Python's print.
                    Target::Ruby if args.len() > 1 => {
                        (format!("puts [{a}].join(\" \")"), prec::LAMBDA)
                    }
                    Target::Ruby => (format!("puts {a}"), prec::LAMBDA),
                    _ => (format!("print({a})"), prec::POSTFIX),
                };
            }
            if matches!(&*name, "len" | "length" | "size" | "count")
                && args.len() == 1
                && let Arg::Pos(x) = &args[0]
            {
                let v = self.sub(x, prec::POSTFIX);
                return match t {
                    Target::JavaScript | Target::Ruby => (format!("{v}.length"), prec::POSTFIX),
                    Target::Lua => (format!("#{v}"), prec::UNARY),
                    _ => (format!("len({v})"), prec::POSTFIX),
                };
            }
            if matches!(&*name, "str" | "string" | "tostring" | "to_s")
                && args.len() == 1
                && let Arg::Pos(x) = &args[0]
            {
                let v = self.sub(x, prec::LAMBDA);
                return match t {
                    Target::JavaScript => (format!("String({v})"), prec::POSTFIX),
                    Target::Ruby => (
                        format!("{}.to_s", self.sub(x, prec::POSTFIX)),
                        prec::POSTFIX,
                    ),
                    Target::Lua => (format!("tostring({v})"), prec::POSTFIX),
                    _ => (format!("str({v})"), prec::POSTFIX),
                };
            }
        }
        // Python's `range(…)` in other targets.
        if let ExprKind::Name(n) = &callee.kind
            && n.is("range")
            && !t.pythonic()
            && !args.is_empty()
            && args.len() <= 3
            && args.iter().all(|a| matches!(a, Arg::Pos(_)))
        {
            let exprs: Vec<&Expr> = args
                .iter()
                .filter_map(|a| if let Arg::Pos(e) = a { Some(e) } else { None })
                .collect();
            let (start, end, step) = match exprs.as_slice() {
                [e] => (None, *e, None),
                [s, e] => (Some(*s), *e, None),
                [s, e, st] => (Some(*s), *e, Some(*st)),
                _ => unreachable!(),
            };
            return self.range(start, Some(end), false, step);
        }
        // Lua methods are called with `:` (lowercase receivers are objects).
        if t == Target::Lua
            && let ExprKind::Member { object, name, .. } = &callee.kind
            && matches!(&object.kind, ExprKind::Name(o) if o.as_str().chars().next().is_some_and(char::is_lowercase) && !matches!(&*o.as_str(), "string" | "table" | "math" | "io" | "os"))
        {
            let o = self.sub(object, prec::POSTFIX);
            let a = self.args(args);
            return (format!("{o}:{name}({a})"), prec::POSTFIX);
        }
        let f = self.sub(callee, prec::POSTFIX);
        let a = self.args(args);
        // Constructors: `Point(1, 2)` is `new Point(1, 2)` / `Point.new(1, 2)`.
        let ctor = matches!(&callee.kind, ExprKind::Name(n) if n.as_str().chars().next().is_some_and(char::is_uppercase));
        match (t, ctor) {
            (Target::JavaScript, true) => (format!("new {f}({a})"), prec::POSTFIX),
            (Target::Ruby, true) => (format!("{f}.new({a})"), prec::POSTFIX),
            (Target::Lua, true) => (format!("{f}.new({a})"), prec::POSTFIX),
            (Target::Ruby, false) if matches!(callee.kind, ExprKind::Name(_)) && a.is_empty() => {
                (f, prec::POSTFIX)
            }
            (Target::Ruby, false)
                if matches!(
                    callee.kind,
                    ExprKind::Name(_)
                        | ExprKind::Lambda(_)
                        | ExprKind::Call { .. }
                        | ExprKind::Index { .. }
                ) && !matches!(callee.kind, ExprKind::Name(_)) =>
            {
                (format!("{f}.({a})"), prec::POSTFIX)
            }
            _ => (format!("{f}({a})"), prec::POSTFIX),
        }
    }

    fn param_list(&mut self, def: &FnDef) -> String {
        let mut ps = Vec::with_capacity(def.params.len());
        for p in &def.params {
            let name = match &p.pattern {
                Pattern::Bind(n) => n.to_string(),
                other => pattern_text(other, self.target),
            };
            let s = match p.kind {
                ParamKind::Rest => match self.target {
                    Target::JavaScript => format!("...{name}"),
                    Target::Lua => "...".into(),
                    _ => format!("*{name}"),
                },
                ParamKind::KwRest => format!("**{name}"),
                ParamKind::Normal => match &p.default {
                    Some(d) => {
                        let d = self.sub(d, prec::LAMBDA);
                        match self.target {
                            Target::Lua => name,
                            Target::JavaScript | Target::Ruby => format!("{name} = {d}"),
                            _ => format!("{name}={d}"),
                        }
                    }
                    None => name,
                },
                #[allow(unreachable_patterns)]
                _ => name,
            };
            ps.push(s);
        }
        ps.join(", ")
    }

    fn lambda(&mut self, def: &FnDef) -> (String, u8) {
        let params = if def.implicit_params {
            "it".to_owned()
        } else {
            self.param_list(def)
        };
        let single = match def.body.stmts.as_slice() {
            [
                Stmt {
                    kind: StmtKind::Expr(e),
                    ..
                },
            ] => Some(e.clone()),
            [] => Some(Expr::new(ExprKind::Nil, def.span)),
            _ => None,
        };
        match (self.target, single) {
            (Target::Python | Target::Chimera, Some(e)) => {
                let body = self.sub(&e, prec::LAMBDA);
                (
                    format!("lambda {params}: {body}").replace("lambda : ", "lambda: "),
                    prec::LAMBDA,
                )
            }
            (Target::JavaScript, Some(e)) => {
                let body = self.sub(&e, prec::LAMBDA + 1);
                let body = if body.starts_with('{') {
                    format!("({body})")
                } else {
                    body
                };
                (format!("({params}) => {body}"), prec::LAMBDA)
            }
            (Target::Ruby, Some(e)) => {
                let body = self.sub(&e, prec::LAMBDA);
                (format!("->({params}) {{ {body} }}"), prec::ATOM)
            }
            (Target::Lua, Some(e)) => {
                let body = self.sub(&e, prec::LAMBDA);
                (format!("function({params}) return {body} end"), prec::ATOM)
            }
            (target, None) => {
                // A multi-statement lambda: render the body on its own lines.
                let mut inner = Emitter {
                    out: String::new(),
                    indent: self.indent + 1,
                    target,
                    src: self.src,
                    tmp: self.tmp,
                    scopes: vec![
                        def.params
                            .iter()
                            .filter_map(|p| p.name())
                            .map(|n| n.to_string())
                            .collect(),
                    ],
                };
                inner.body_returning(&def.body);
                self.tmp = inner.tmp;
                let pad = "  ".repeat(self.indent);
                let body = inner.out.trim_end().to_owned();
                match target {
                    Target::JavaScript => {
                        (format!("({params}) => {{\n{body}\n{pad}}}"), prec::LAMBDA)
                    }
                    Target::Ruby => (
                        format!("lambda do |{params}|\n{body}\n{pad}end"),
                        prec::ATOM,
                    ),
                    Target::Lua => (format!("function({params})\n{body}\n{pad}end"), prec::ATOM),
                    _ => {
                        // Python lambdas hold one expression; hoist a def.
                        let name = self.fresh("fn");
                        let mut hoisted = Emitter {
                            out: String::new(),
                            indent: self.indent,
                            target,
                            src: self.src,
                            tmp: self.tmp,
                            scopes: vec![Vec::new()],
                        };
                        hoisted.line(&format!("def {name}({params}):"));
                        hoisted.indent += 1;
                        hoisted.body_returning(&def.body);
                        self.tmp = hoisted.tmp;
                        self.out.push_str(&hoisted.out);
                        (name, prec::ATOM)
                    }
                }
            }
        }
    }

    /// A `match` used as a value: a chain of conditionals when every arm
    /// tests a literal (or is a catch-all) and yields one expression.
    fn match_value(
        &mut self,
        subject: Option<&Expr>,
        arms: &[MatchArm],
        span: chimera_core::Span,
    ) -> (String, u8) {
        let t = self.target;
        let simple = arms.iter().all(|a| {
            a.guard.is_none()
                && matches!(a.pattern, Pattern::Literal(_) | Pattern::Wildcard)
                && matches!(
                    a.body.stmts.as_slice(),
                    [Stmt {
                        kind: StmtKind::Expr(_),
                        ..
                    }]
                )
        });
        let Some(subject) = subject.filter(|_| simple) else {
            return (self.untranslatable(span, "match"), prec::LAMBDA);
        };
        let subj = self.sub(subject, prec::CMP + 1);
        let eq = if t == Target::JavaScript { "===" } else { "==" };
        let mut out = self.nil().to_string();
        for arm in arms.iter().rev() {
            let value = self.block_value(&arm.body);
            out = match &arm.pattern {
                Pattern::Literal(lit) => {
                    let lit = self.sub(lit, prec::CMP + 1);
                    let cond = format!("{subj} {eq} {lit}");
                    match t {
                        Target::Python | Target::Chimera => format!("{value} if {cond} else {out}"),
                        Target::Lua => format!("({cond} and {value} or {out})"),
                        _ => format!("{cond} ? {value} : {out}"),
                    }
                }
                _ => value,
            };
        }
        (out, prec::LAMBDA)
    }

    /// The value of a block used as an expression.
    fn block_value(&mut self, b: &Block) -> String {
        match b.stmts.as_slice() {
            [] => self.nil().into(),
            [
                Stmt {
                    kind: StmtKind::Expr(e),
                    ..
                },
            ] => self.sub(e, prec::LAMBDA + 1),
            stmts => {
                let span = stmts[0].span.to(stmts[stmts.len() - 1].span);
                self.untranslatable(span, "block expression")
            }
        }
    }

    fn comprehension(&mut self, c: &Comprehension) -> (String, u8) {
        let t = self.target;
        if t.pythonic() {
            let el = match (c.kind, &c.value) {
                (CompKind::Map, Some(v)) => format!(
                    "{}: {}",
                    self.sub(&c.element, prec::LAMBDA),
                    self.sub(v, prec::LAMBDA)
                ),
                _ => self.sub(&c.element, prec::LAMBDA),
            };
            let mut s = el;
            for clause in &c.clauses {
                match clause {
                    CompClause::For(p, it) => {
                        let it = self.sub(it, prec::OR);
                        let _ = write!(s, " for {} in {it}", pattern_text(p, t));
                    }
                    CompClause::If(cond) => {
                        let cond = self.sub(cond, prec::OR);
                        let _ = write!(s, " if {cond}");
                    }
                }
            }
            let (open, close) = match c.kind {
                CompKind::List => ("[", "]"),
                _ => ("{", "}"),
            };
            return (format!("{open}{s}{close}"), prec::ATOM);
        }
        // Single-generator comprehensions become filter/map chains.
        if let [CompClause::For(p, it), rest @ ..] = c.clauses.as_slice()
            && rest.iter().all(|r| matches!(r, CompClause::If(_)))
        {
            let var = pattern_text(p, t);
            let mut chain = self.sub(it, prec::POSTFIX);
            for r in rest {
                if let CompClause::If(cond) = r {
                    let cond = self.sub(cond, prec::LAMBDA);
                    chain = match t {
                        Target::JavaScript => format!("{chain}.filter(({var}) => {cond})"),
                        Target::Ruby => format!("{chain}.select {{ |{var}| {cond} }}"),
                        _ => format!("__filter({chain}, function({var}) return {cond} end)"),
                    };
                }
            }
            let el = self.sub(&c.element, prec::LAMBDA);
            let mapped = match t {
                Target::JavaScript => format!("{chain}.map(({var}) => {el})"),
                Target::Ruby => format!("{chain}.map {{ |{var}| {el} }}"),
                _ => format!("__map({chain}, function({var}) return {el} end)"),
            };
            return (mapped, prec::POSTFIX);
        }
        (
            self.untranslatable(c.element.span, "comprehension"),
            prec::LAMBDA,
        )
    }

    // ----- statements ------------------------------------------------------------------

    fn open_block(&mut self, header: &str) {
        match self.target {
            Target::Python | Target::Chimera => self.line(&format!("{header}:")),
            Target::JavaScript => self.line(&format!("{header} {{")),
            Target::Ruby => self.line(header),
            Target::Lua => self.line(header),
        }
        self.indent += 1;
    }

    fn close_block(&mut self) {
        self.indent = self.indent.saturating_sub(1);
        match self.target {
            Target::Python | Target::Chimera => {}
            Target::JavaScript => self.line("}"),
            _ => self.line("end"),
        }
    }

    fn body(&mut self, b: &Block) {
        if b.stmts.is_empty() && self.target.pythonic() {
            self.line("pass");
        }
        for s in &b.stmts {
            self.stmt(s);
        }
    }

    /// A function body whose last expression is returned implicitly.
    fn body_returning(&mut self, b: &Block) {
        let n = b.stmts.len();
        if n == 0 && self.target.pythonic() {
            self.line("pass");
        }
        for (i, s) in b.stmts.iter().enumerate() {
            let last = i + 1 == n;
            match &s.kind {
                StmtKind::Expr(e) if last && self.target != Target::Ruby && returns_value(e) => {
                    let v = self.sub(e, prec::LAMBDA);
                    let semi = if self.target == Target::JavaScript {
                        ";"
                    } else {
                        ""
                    };
                    self.line(&format!("return {v}{semi}"));
                }
                _ => self.stmt(s),
            }
        }
    }

    fn end_stmt(&self) -> &'static str {
        if self.target == Target::JavaScript {
            ";"
        } else {
            ""
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        let t = self.target;
        let semi = self.end_stmt();
        match &s.kind {
            StmtKind::Expr(e) => self.expr_stmt(e),
            StmtKind::Let { pattern, value, .. } => {
                let v = match value {
                    Some(v) => self.sub(v, prec::LAMBDA),
                    None => self.nil().into(),
                };
                let p = pattern_text(pattern, t);
                match t {
                    Target::JavaScript => {
                        let p = match pattern {
                            Pattern::Tuple(_) => format!("[{}]", p.trim_matches(['(', ')'])),
                            _ => p,
                        };
                        self.line(&format!("let {p} = {v};"));
                    }
                    Target::Lua => self.line(&format!("local {p} = {v}")),
                    _ => self.line(&format!("{p} = {v}")),
                }
            }
            StmtKind::Fn(def) => self.function(def, false),
            StmtKind::Class(def) => self.class(def),
            StmtKind::Impl {
                target,
                methods,
                statics,
            } => {
                let c = self.untranslatable(target.span, "impl");
                let name = self.sub(target, prec::POSTFIX);
                let _ = c;
                for m in methods.iter().chain(statics) {
                    let mut f = (**m).clone();
                    if let Some(n) = f.name {
                        f.name = Some(Sym::new(&format!("{name}_{n}")));
                    }
                    self.function(&f, false);
                }
            }
            StmtKind::Enum(def) => self.enumeration(def),
            StmtKind::While {
                cond,
                body,
                negate,
                post,
            } => {
                let c = self.sub(cond, prec::LAMBDA);
                let c = if *negate { self.negate_text(&c) } else { c };
                if *post {
                    match t {
                        Target::JavaScript => {
                            self.line("do {");
                            self.indent += 1;
                            self.body(body);
                            self.indent -= 1;
                            self.line(&format!("}} while ({c});"));
                        }
                        Target::Lua => {
                            self.line("repeat");
                            self.indent += 1;
                            self.body(body);
                            self.indent -= 1;
                            self.line(&format!("until not ({c})"));
                        }
                        Target::Ruby => {
                            self.line("begin");
                            self.indent += 1;
                            self.body(body);
                            self.indent -= 1;
                            self.line(&format!("end while {c}"));
                        }
                        _ => {
                            self.open_block("while True");
                            self.body(body);
                            self.line(&format!("if not ({c}): break"));
                            self.close_block();
                        }
                    }
                    return;
                }
                let header = match t {
                    Target::JavaScript => format!("while ({c})"),
                    Target::Lua => format!("while {c} do"),
                    _ => format!("while {c}"),
                };
                self.open_block(&header);
                self.body(body);
                self.close_block();
            }
            StmtKind::Loop(body) => {
                let header = match t {
                    Target::JavaScript => "while (true)",
                    Target::Ruby => "loop do",
                    Target::Lua => "while true do",
                    _ => "while True",
                };
                self.open_block(header);
                self.body(body);
                self.close_block();
            }
            StmtKind::Repeat { count, body } => {
                let n = self.sub(count, prec::LAMBDA);
                let header = match t {
                    Target::JavaScript => format!("for (let _i = 0; _i < {n}; _i++)"),
                    Target::Ruby => format!("{n}.times do"),
                    Target::Lua => format!("for _ = 1, {n} do"),
                    _ => format!("for _ in range({n})"),
                };
                self.open_block(&header);
                self.body(body);
                self.close_block();
            }
            StmtKind::For {
                pattern,
                iter,
                body,
                ..
            } => self.for_loop(pattern, iter, body),
            StmtKind::CFor {
                init,
                cond,
                step,
                body,
            } => {
                let init_s = init
                    .as_ref()
                    .map(|s| self.inline_stmt(s))
                    .unwrap_or_default();
                let cond_s = cond.as_ref().map(|c| self.sub(c, prec::LAMBDA));
                let step_s = step
                    .as_ref()
                    .map(|c| self.sub(c, prec::LAMBDA))
                    .unwrap_or_default();
                if t == Target::JavaScript {
                    self.open_block(&format!(
                        "for ({init_s}; {}; {step_s})",
                        cond_s.unwrap_or_default()
                    ));
                    self.body(body);
                    self.close_block();
                    return;
                }
                if !init_s.is_empty() {
                    self.line(&init_s);
                }
                let c = cond_s.unwrap_or_else(|| self.bool(true).into());
                let header = match t {
                    Target::Lua => format!("while {c} do"),
                    _ => format!("while {c}"),
                };
                self.open_block(&header);
                self.body(body);
                if let Some(step) = step {
                    self.expr_stmt(step);
                }
                self.close_block();
            }
            StmtKind::Return(v) => {
                let kw = "return";
                match v {
                    Some(v) => {
                        let v = self.sub(v, prec::LAMBDA);
                        self.line(&format!("{kw} {v}{semi}"));
                    }
                    None => self.line(&format!("{kw}{semi}")),
                }
            }
            StmtKind::Break => self.line(&format!("break{semi}")),
            StmtKind::Continue => self.line(match t {
                Target::Ruby => "next",
                Target::Lua => "goto continue",
                Target::JavaScript => "continue;",
                _ => "continue",
            }),
            StmtKind::Throw(v) => {
                let v = self.sub(v, prec::LAMBDA);
                self.line(&match t {
                    Target::JavaScript => format!("throw {v};"),
                    Target::Lua => format!("error({v})"),
                    _ => format!("raise {v}"),
                });
            }
            StmtKind::Try {
                body,
                catches,
                else_block,
                finally,
            } => self.try_stmt(body, catches, else_block.as_ref(), finally.as_ref()),
            StmtKind::Switch {
                subject,
                cases,
                default,
            } => {
                let subj = self.sub(subject, prec::LAMBDA);
                let mut first = true;
                for c in cases {
                    let tests: Vec<String> = c
                        .values
                        .iter()
                        .map(|v| {
                            let v = self.sub(v, prec::CMP + 1);
                            format!(
                                "{subj} {} {v}",
                                if t == Target::JavaScript { "===" } else { "==" }
                            )
                        })
                        .collect();
                    let or = if matches!(t, Target::JavaScript | Target::Ruby) {
                        " || "
                    } else {
                        " or "
                    };
                    let cond = tests.join(or);
                    self.if_header(&cond, first);
                    first = false;
                    let body = strip_break(&c.body);
                    self.body(&body);
                    self.indent -= 1;
                }
                if let Some(d) = default {
                    self.else_header(first);
                    let body = strip_break(d);
                    self.body(&body);
                    self.indent -= 1;
                }
                if !first || default.is_some() {
                    self.indent += 1;
                    self.close_block();
                }
            }
            StmtKind::Import(imp) => {
                let path: Vec<String> = imp.path.iter().map(|p| p.to_string()).collect();
                let module = imp.file.clone().unwrap_or_else(|| path.join("."));
                let c = t.comment();
                self.line(&format!("{c} import {module}"));
            }
            StmtKind::Defer(b) => {
                let c = t.comment();
                self.line(&format!("{c} deferred to the end of the function:"));
                self.body(b);
            }
            StmtKind::Delete(target) => {
                let tg = self.sub(target, prec::POSTFIX);
                self.line(&match t {
                    Target::JavaScript => format!("delete {tg};"),
                    Target::Lua => format!("{tg} = nil"),
                    Target::Ruby => format!("{tg} = nil"),
                    _ => format!("del {tg}"),
                });
            }
            StmtKind::Model(def) => {
                let text = self.original(def.span);
                let c = t.comment();
                self.line(&format!("{c} chimera model (no equivalent): {text}"));
            }
            StmtKind::Train(spec) => {
                let text = self.original(spec.span);
                let c = t.comment();
                self.line(&format!("{c} chimera train (no equivalent): {text}"));
            }
            StmtKind::Pass => {
                if t.pythonic() {
                    self.line("pass");
                }
            }
        }
    }

    fn negate_text(&self, c: &str) -> String {
        match self.target {
            Target::JavaScript | Target::Ruby => format!("!({c})"),
            _ => format!("not ({c})"),
        }
    }

    /// A simple statement rendered on one line (for C-style `for` headers).
    fn inline_stmt(&mut self, s: &Stmt) -> String {
        match &s.kind {
            StmtKind::Let { pattern, value, .. } => {
                let v = value
                    .as_ref()
                    .map(|v| self.sub(v, prec::LAMBDA))
                    .unwrap_or_else(|| self.nil().into());
                let p = pattern_text(pattern, self.target);
                match self.target {
                    Target::JavaScript => format!("let {p} = {v}"),
                    Target::Lua => format!("local {p} = {v}"),
                    _ => format!("{p} = {v}"),
                }
            }
            StmtKind::Expr(e) => self.sub(e, prec::LAMBDA),
            _ => String::new(),
        }
    }

    fn expr_stmt(&mut self, e: &Expr) {
        let semi = self.end_stmt();
        match &e.kind {
            ExprKind::If {
                cond,
                then,
                otherwise,
            } => self.if_chain(cond, then, otherwise.as_ref()),
            ExprKind::Match { subject, arms } => self.match_stmt(subject.as_deref(), arms),
            ExprKind::Block(b) => {
                for s in &b.stmts {
                    self.stmt(s);
                }
            }
            ExprKind::Please(_) | ExprKind::Query(_) => {
                let text = self.original(e.span);
                let c = self.target.comment();
                self.line(&format!("{c} chimera (no equivalent): {text}"));
            }
            ExprKind::Assign { target, value } if matches!(target.kind, ExprKind::Name(_)) => {
                let ExprKind::Name(n) = &target.kind else {
                    unreachable!()
                };
                let name = self.name(*n);
                let kw = if self.declare(&name) {
                    match self.target {
                        Target::JavaScript => "let ",
                        _ => "local ",
                    }
                } else {
                    ""
                };
                let v = self.sub(value, prec::LAMBDA);
                self.line(&format!("{kw}{name} = {v}{semi}"));
            }
            _ => {
                let s = self.sub(e, prec::LAMBDA);
                for (i, l) in s.lines().enumerate() {
                    if i == 0 {
                        self.line(&format!(
                            "{l}{}",
                            if s.lines().count() == 1 { semi } else { "" }
                        ));
                    } else {
                        self.out.push_str(l);
                        self.out.push('\n');
                    }
                }
            }
        }
    }

    fn if_header(&mut self, cond: &str, first: bool) {
        let t = self.target;
        let kw_if = match (t, first) {
            (Target::Python | Target::Chimera, false) => "elif",
            (Target::Ruby, false) => "elsif",
            (Target::Lua, false) => "elseif",
            (Target::JavaScript, false) => "} else if",
            _ => "if",
        };
        let header = match t {
            Target::JavaScript => format!("{kw_if} ({cond}) {{"),
            Target::Lua => format!("{kw_if} {cond} then"),
            Target::Python | Target::Chimera => format!("{kw_if} {cond}:"),
            Target::Ruby => format!("{kw_if} {cond}"),
        };
        self.line(&header);
        self.indent += 1;
    }

    fn else_header(&mut self, first: bool) {
        if first {
            // `switch` with only a default: run it unconditionally.
            self.line(match self.target {
                Target::JavaScript => "{",
                Target::Lua => "do",
                Target::Ruby => "begin",
                _ => "if True:",
            });
        } else {
            self.line(match self.target {
                Target::JavaScript => "} else {",
                Target::Python | Target::Chimera => "else:",
                _ => "else",
            });
        }
        self.indent += 1;
    }

    fn if_chain(&mut self, cond: &Expr, then: &Block, otherwise: Option<&Block>) {
        let c = self.sub(cond, prec::LAMBDA);
        self.if_header(&c, true);
        self.body(then);
        self.indent -= 1;
        let mut rest = otherwise;
        while let Some(b) = rest {
            // `else if` chains arrive as an else-block holding one `if`.
            if let [
                Stmt {
                    kind:
                        StmtKind::Expr(Expr {
                            kind:
                                ExprKind::If {
                                    cond,
                                    then,
                                    otherwise,
                                },
                            ..
                        }),
                    ..
                },
            ] = b.stmts.as_slice()
            {
                let c = self.sub(cond, prec::LAMBDA);
                self.if_header(&c, false);
                self.body(then);
                self.indent -= 1;
                rest = otherwise.as_ref();
                continue;
            }
            self.else_header(false);
            self.body(b);
            self.indent -= 1;
            break;
        }
        self.indent += 1;
        self.close_block();
    }

    fn match_stmt(&mut self, subject: Option<&Expr>, arms: &[MatchArm]) {
        let t = self.target;
        let subj = subject.map(|s| self.sub(s, prec::LAMBDA));
        if t.pythonic()
            && let Some(subj) = &subj
        {
            self.open_block(&format!("match {subj}"));
            for arm in arms {
                let mut p = pattern_text(&arm.pattern, t);
                if let Some(g) = &arm.guard {
                    let g = self.sub(g, prec::LAMBDA);
                    p = format!("{p} if {g}");
                }
                self.open_block(&format!("case {p}"));
                self.body(&arm.body);
                self.close_block();
            }
            self.close_block();
            return;
        }
        if t == Target::Ruby
            && let Some(subj) = &subj
        {
            self.line(&format!("case {subj}"));
            for arm in arms {
                let p = match &arm.pattern {
                    Pattern::Wildcard | Pattern::Bind(_) => None,
                    other => Some(pattern_text(other, t)),
                };
                match p {
                    Some(p) => self.line(&format!("when {p}")),
                    None => self.line("else"),
                }
                self.indent += 1;
                self.body(&arm.body);
                self.indent -= 1;
            }
            self.line("end");
            return;
        }
        // JavaScript and Lua: an if/else chain over simple patterns.
        let tmp = match &subj {
            Some(s) => {
                let name = self.fresh("subject");
                let decl = match t {
                    Target::JavaScript => format!("const {name} = {s};"),
                    _ => format!("local {name} = {s}"),
                };
                self.line(&decl);
                Some(name)
            }
            None => None,
        };
        let mut first = true;
        let mut open = false;
        for arm in arms {
            let mut cond = match (&tmp, &arm.pattern) {
                (_, Pattern::Wildcard) => None,
                (Some(v), Pattern::Bind(n)) => {
                    // A catch-all binding.
                    self.if_or_else(first, None);
                    let decl = match t {
                        Target::JavaScript => format!("const {n} = {v};"),
                        _ => format!("local {n} = {v}"),
                    };
                    self.line(&decl);
                    self.body(&arm.body);
                    self.indent -= 1;
                    first = false;
                    open = true;
                    continue;
                }
                (Some(v), Pattern::Literal(e)) => {
                    let lit = self.sub(e, prec::CMP + 1);
                    Some(format!(
                        "{v} {} {lit}",
                        if t == Target::JavaScript { "===" } else { "==" }
                    ))
                }
                (Some(v), Pattern::Or(alts)) => {
                    let mut tests = Vec::new();
                    for a in alts {
                        if let Pattern::Literal(e) = a {
                            let lit = self.sub(e, prec::CMP + 1);
                            tests.push(format!(
                                "{v} {} {lit}",
                                if t == Target::JavaScript { "===" } else { "==" }
                            ));
                        }
                    }
                    Some(tests.join(if t == Target::JavaScript {
                        " || "
                    } else {
                        " or "
                    }))
                }
                (Some(v), Pattern::Range { lo, hi, inclusive }) => {
                    let (lo, hi) = (self.sub(lo, prec::CMP + 1), self.sub(hi, prec::CMP + 1));
                    let and = if t == Target::JavaScript { "&&" } else { "and" };
                    let op = if *inclusive { "<=" } else { "<" };
                    Some(format!("{v} >= {lo} {and} {v} {op} {hi}"))
                }
                (None, Pattern::CaseEq(e)) | (None, Pattern::Literal(e)) => {
                    Some(self.sub(e, prec::LAMBDA))
                }
                (_, other) => Some(format!("true /* pattern: {} */", pattern_text(other, t))),
            };
            if let Some(g) = &arm.guard {
                let g = self.sub(g, prec::AND + 1);
                let and = if t == Target::JavaScript { "&&" } else { "and" };
                cond = Some(match cond {
                    Some(c) => format!("({c}) {and} {g}"),
                    None => g,
                });
            }
            self.if_or_else(first, cond.as_deref());
            self.body(&arm.body);
            self.indent -= 1;
            first = false;
            open = true;
        }
        if open {
            self.indent += 1;
            self.close_block();
        }
    }

    fn if_or_else(&mut self, first: bool, cond: Option<&str>) {
        match cond {
            Some(c) => self.if_header(c, first),
            None => self.else_header(first),
        }
    }

    fn for_loop(&mut self, pattern: &Pattern, iter: &Expr, body: &Block) {
        let t = self.target;
        let var = pattern_text(pattern, t);
        // `for i in range(a, b)` counts like a range literal.
        if let ExprKind::Call { callee, args } = &iter.kind
            && matches!(&callee.kind, ExprKind::Name(n) if n.is("range"))
            && (1..=3).contains(&args.len())
            && args.iter().all(|a| matches!(a, Arg::Pos(_)))
        {
            let exprs: Vec<Expr> = args
                .iter()
                .filter_map(|a| {
                    if let Arg::Pos(e) = a {
                        Some(e.clone())
                    } else {
                        None
                    }
                })
                .collect();
            let (start, end, step) = match exprs.len() {
                1 => (None, exprs[0].clone(), None),
                2 => (Some(Box::new(exprs[0].clone())), exprs[1].clone(), None),
                _ => (
                    Some(Box::new(exprs[0].clone())),
                    exprs[1].clone(),
                    Some(Box::new(exprs[2].clone())),
                ),
            };
            let as_range = Expr::new(
                ExprKind::Range {
                    start,
                    end: Some(Box::new(end)),
                    inclusive: false,
                    step,
                },
                iter.span,
            );
            return self.for_loop(pattern, &as_range, body);
        }
        // Numeric ranges get native counting loops.
        if let ExprKind::Range {
            start,
            end: Some(end),
            inclusive,
            step,
        } = &iter.kind
        {
            let s = start
                .as_deref()
                .map_or_else(|| "0".into(), |x| self.sub(x, prec::LAMBDA));
            let e = self.sub(end, prec::ADD);
            let st = step.as_deref().map(|x| self.sub(x, prec::LAMBDA));
            let header = match t {
                Target::JavaScript => {
                    let cmp = if *inclusive { "<=" } else { "<" };
                    let inc = st.map_or_else(|| format!("{var}++"), |st| format!("{var} += {st}"));
                    format!("for (let {var} = {s}; {var} {cmp} {e}; {inc})")
                }
                Target::Lua => {
                    let last = if *inclusive { e } else { format!("{e} - 1") };
                    match st {
                        Some(st) => format!("for {var} = {s}, {last}, {st} do"),
                        None => format!("for {var} = {s}, {last} do"),
                    }
                }
                Target::Ruby => {
                    let dots = if *inclusive { ".." } else { "..." };
                    match st {
                        Some(st) => format!("({s}{dots}{e}).step({st}) do |{var}|"),
                        None => format!("for {var} in {s}{dots}{e}"),
                    }
                }
                _ => {
                    let end = if *inclusive { format!("{e} + 1") } else { e };
                    match st {
                        Some(st) => format!("for {var} in range({s}, {end}, {st})"),
                        None => format!("for {var} in range({s}, {end})"),
                    }
                }
            };
            self.open_block(&header);
            self.body(body);
            if t == Target::Lua && block_has_continue(body) {
                self.line("::continue::");
            }
            self.close_block();
            return;
        }
        let it = self.sub(iter, prec::POSTFIX);
        let header = match t {
            Target::JavaScript => {
                let var = match pattern {
                    Pattern::Tuple(_) | Pattern::List { .. } => {
                        format!("[{}]", var.trim_matches(['(', ')', '[', ']']))
                    }
                    _ => var,
                };
                format!("for (const {var} of {it})")
            }
            Target::Ruby => format!("{it}.each do |{}|", var.trim_matches(['(', ')'])),
            Target::Lua => match pattern {
                Pattern::Tuple(items) if items.len() == 2 => {
                    format!("for {} in pairs({it}) do", var.trim_matches(['(', ')']))
                }
                _ => format!("for _, {var} in ipairs({it}) do"),
            },
            _ => format!("for {} in {it}", var.trim_matches(['(', ')'])),
        };
        self.open_block(&header);
        self.body(body);
        if t == Target::Lua && block_has_continue(body) {
            self.line("::continue::");
        }
        self.close_block();
    }

    fn try_stmt(
        &mut self,
        body: &Block,
        catches: &[Catch],
        else_block: Option<&Block>,
        finally: Option<&Block>,
    ) {
        let t = self.target;
        match t {
            Target::Lua => {
                // pcall-based translation.
                self.line("local ok, err = pcall(function()");
                self.indent += 1;
                self.body(body);
                self.indent -= 1;
                self.line("end)");
                if !catches.is_empty() {
                    self.line("if not ok then");
                    self.indent += 1;
                    if let Some(b) = catches[0].binding {
                        self.line(&format!("local {b} = err"));
                    }
                    self.body(&catches[0].body);
                    self.indent -= 1;
                    self.line("end");
                }
                if let Some(f) = finally {
                    self.body(f);
                }
                return;
            }
            Target::JavaScript => {
                self.line("try {");
                self.indent += 1;
                self.body(body);
                self.indent -= 1;
                if !catches.is_empty() {
                    let b = catches[0]
                        .binding
                        .map_or_else(|| "e".into(), |b| b.to_string());
                    self.line(&format!("}} catch ({b}) {{"));
                    self.indent += 1;
                    self.body(&catches[0].body);
                    self.indent -= 1;
                }
                if let Some(f) = finally {
                    self.line("} finally {");
                    self.indent += 1;
                    self.body(f);
                    self.indent -= 1;
                }
                self.line("}");
                return;
            }
            _ => {}
        }
        let ruby = t == Target::Ruby;
        self.open_block(if ruby { "begin" } else { "try" });
        self.body(body);
        self.indent -= 1;
        for c in catches {
            let classes: Vec<String> = c
                .classes
                .iter()
                .map(|e| self.sub(e, prec::POSTFIX))
                .collect();
            let header = match (ruby, classes.is_empty(), c.binding) {
                (true, true, Some(b)) => format!("rescue => {b}"),
                (true, true, None) => "rescue".into(),
                (true, false, Some(b)) => format!("rescue {} => {b}", classes.join(", ")),
                (true, false, None) => format!("rescue {}", classes.join(", ")),
                (false, true, Some(b)) => format!("except Exception as {b}:"),
                (false, true, None) => "except Exception:".into(),
                (false, false, Some(b)) if classes.len() == 1 => {
                    format!("except {} as {b}:", classes[0])
                }
                (false, false, Some(b)) => format!("except ({}) as {b}:", classes.join(", ")),
                (false, false, None) => format!("except ({}):", classes.join(", ")),
            };
            self.line(&header);
            self.indent += 1;
            self.body(&c.body);
            self.indent -= 1;
        }
        if let Some(e) = else_block {
            self.line(if ruby { "else" } else { "else:" });
            self.indent += 1;
            self.body(e);
            self.indent -= 1;
        }
        if let Some(f) = finally {
            self.line(if ruby { "ensure" } else { "finally:" });
            self.indent += 1;
            self.body(f);
            self.indent -= 1;
        }
        if ruby {
            self.line("end");
        }
    }

    fn function(&mut self, def: &FnDef, method: bool) {
        let t = self.target;
        let name = def
            .name
            .map_or_else(|| "anonymous".into(), |n| n.to_string());
        let mut params = self.param_list(def);
        for d in &def.decorators {
            if t.pythonic() {
                let d = self.sub(d, prec::POSTFIX);
                self.line(&format!("@{d}"));
            }
        }
        let gen_star = if def.is_generator() && t == Target::JavaScript {
            "*"
        } else {
            ""
        };
        let header = match (t, method) {
            (Target::Python | Target::Chimera, true) => {
                params = if params.is_empty() {
                    "self".into()
                } else {
                    format!("self, {params}")
                };
                if def.is_static {
                    self.line("@staticmethod");
                    params = self.param_list(def);
                }
                format!("def {name}({params})")
            }
            (Target::Python | Target::Chimera, false) => format!("def {name}({params})"),
            (Target::JavaScript, true) => {
                let st = if def.is_static { "static " } else { "" };
                format!("{st}{gen_star}{name}({params})")
            }
            (Target::JavaScript, false) => format!("function{gen_star} {name}({params})"),
            (Target::Ruby, true) if def.is_static => format!("def self.{name}({params})"),
            (Target::Ruby, _) => format!("def {name}({params})"),
            (Target::Lua, _) => format!("local function {name}({params})"),
        };
        self.open_block(&header);
        self.scopes.push(
            def.params
                .iter()
                .filter_map(|p| p.name())
                .map(|n| n.to_string())
                .collect(),
        );
        if let Some(n) = def.name
            && let Some(outer) = self.scopes.iter_mut().rev().nth(1)
        {
            outer.push(n.to_string());
        }
        // Lua default parameters.
        if t == Target::Lua {
            for p in &def.params {
                if let (Some(n), Some(d)) = (p.name(), &p.default) {
                    let d = self.sub(d, prec::OR + 1);
                    self.line(&format!("if {n} == nil then {n} = {d} end"));
                }
            }
        }
        self.body_returning(&def.body);
        self.scopes.pop();
        self.close_block();
        if !method && t.pythonic() {
            self.out.push('\n');
        }
    }

    fn class(&mut self, def: &ClassDef) {
        let t = self.target;
        let name = def.name.to_string();
        let parent = def.superclass.as_ref().map(|s| self.sub(s, prec::POSTFIX));
        let has_init = def
            .methods
            .iter()
            .any(|m| m.name.is_some_and(|n| n.is("init")));
        let fields: Vec<(String, Option<String>)> = def
            .fields
            .iter()
            .map(|f| {
                (
                    f.name.to_string(),
                    f.default.as_ref().map(|d| self.sub(d, prec::LAMBDA)),
                )
            })
            .collect();
        match t {
            Target::Lua => {
                self.line(&format!("local {name} = {{}}"));
                self.line(&format!("{name}.__index = {name}"));
                if let Some(p) = &parent {
                    self.line(&format!("setmetatable({name}, {{ __index = {p} }})"));
                }
                if has_init {
                    self.line(&format!("function {name}.new(...)"));
                    self.indent += 1;
                    self.line(&format!("local self = setmetatable({{}}, {name})"));
                    self.line("self:init(...)");
                } else {
                    let args: Vec<String> = fields.iter().map(|(n, _)| n.clone()).collect();
                    self.line(&format!("function {name}.new({})", args.join(", ")));
                    self.indent += 1;
                    self.line(&format!("local self = setmetatable({{}}, {name})"));
                    for (n, d) in &fields {
                        match d {
                            Some(d) => self.line(&format!("self.{n} = {n} or {d}")),
                            None => self.line(&format!("self.{n} = {n}")),
                        }
                    }
                }
                self.line("return self");
                self.indent -= 1;
                self.line("end");
                for m in def.methods.iter().chain(&def.statics) {
                    let mname = m.name.map_or_else(|| "method".into(), |n| n.to_string());
                    let params = self.param_list(m);
                    let sep = if m.is_static { "." } else { ":" };
                    self.open_block(&format!("function {name}{sep}{mname}({params})"));
                    self.scopes.push(
                        m.params
                            .iter()
                            .filter_map(|p| p.name())
                            .map(|n| n.to_string())
                            .collect(),
                    );
                    for p in &m.params {
                        if let (Some(n), Some(d)) = (p.name(), &p.default) {
                            let d = self.sub(d, prec::OR + 1);
                            self.line(&format!("if {n} == nil then {n} = {d} end"));
                        }
                    }
                    self.body_returning(&m.body);
                    self.scopes.pop();
                    self.close_block();
                }
                return;
            }
            Target::Python | Target::Chimera => {
                let header = match &parent {
                    Some(p) => format!("class {name}({p})"),
                    None => format!("class {name}"),
                };
                self.open_block(&header);
                for (n, v) in &def.class_attrs {
                    let v = self.sub(v, prec::LAMBDA);
                    self.line(&format!("{n} = {v}"));
                }
                if !has_init && !fields.is_empty() {
                    let params: Vec<String> = fields
                        .iter()
                        .map(|(n, d)| match d {
                            Some(d) => format!("{n}={d}"),
                            None => n.clone(),
                        })
                        .collect();
                    self.open_block(&format!("def __init__(self, {})", params.join(", ")));
                    for (n, _) in &fields {
                        self.line(&format!("self.{n} = {n}"));
                    }
                    self.close_block();
                }
            }
            Target::JavaScript => {
                let header = match &parent {
                    Some(p) => format!("class {name} extends {p}"),
                    None => format!("class {name}"),
                };
                self.open_block(&header);
                if !has_init && !fields.is_empty() {
                    let params: Vec<String> = fields
                        .iter()
                        .map(|(n, d)| match d {
                            Some(d) => format!("{n} = {d}"),
                            None => n.clone(),
                        })
                        .collect();
                    self.open_block(&format!("constructor({})", params.join(", ")));
                    if parent.is_some() {
                        self.line("super();");
                    }
                    for (n, _) in &fields {
                        self.line(&format!("this.{n} = {n};"));
                    }
                    self.close_block();
                }
            }
            Target::Ruby => {
                let header = match &parent {
                    Some(p) => format!("class {name} < {p}"),
                    None => format!("class {name}"),
                };
                self.open_block(&header);
                // Accessors for declared fields and for `self.x = …` in methods.
                let mut names: Vec<String> = fields.iter().map(|(n, _)| n.clone()).collect();
                for m in &def.methods {
                    collect_self_fields(&m.body, &mut names);
                }
                if !names.is_empty() {
                    let syms: Vec<String> = names.iter().map(|n| format!(":{n}")).collect();
                    self.line(&format!("attr_accessor {}", syms.join(", ")));
                }
                if !has_init && !fields.is_empty() {
                    let params: Vec<String> = fields
                        .iter()
                        .map(|(n, d)| match d {
                            Some(d) => format!("{n} = {d}"),
                            None => n.clone(),
                        })
                        .collect();
                    self.open_block(&format!("def initialize({})", params.join(", ")));
                    for (n, _) in &fields {
                        self.line(&format!("@{n} = {n}"));
                    }
                    self.close_block();
                }
            }
        }
        for m in def.methods.iter().chain(&def.statics) {
            let mut m = (**m).clone();
            if m.name.is_some_and(|n| n.is("init")) {
                m.name = Some(Sym::new(match t {
                    Target::JavaScript => "constructor",
                    Target::Ruby => "initialize",
                    _ => "__init__",
                }));
            }
            self.function(&m, true);
        }
        if def.methods.is_empty()
            && def.statics.is_empty()
            && fields.is_empty()
            && def.class_attrs.is_empty()
            && t.pythonic()
        {
            self.line("pass");
        }
        self.close_block();
        if t.pythonic() {
            self.out.push('\n');
        }
    }

    fn enumeration(&mut self, def: &EnumDef) {
        let t = self.target;
        let name = def.name.to_string();
        match t {
            Target::Python | Target::Chimera => {
                self.line("from enum import Enum");
                self.open_block(&format!("class {name}(Enum)"));
                for (i, v) in def.variants.iter().enumerate() {
                    let val = v
                        .value
                        .as_ref()
                        .map_or_else(|| (i + 1).to_string(), |e| self.sub(e, prec::LAMBDA));
                    self.line(&format!("{} = {val}", v.name));
                }
                self.close_block();
            }
            Target::JavaScript => {
                let items: Vec<String> = def
                    .variants
                    .iter()
                    .map(|v| format!("{0}: \"{0}\"", v.name))
                    .collect();
                self.line(&format!(
                    "const {name} = Object.freeze({{ {} }});",
                    items.join(", ")
                ));
            }
            Target::Ruby => {
                self.line(&format!("module {name}"));
                for v in &def.variants {
                    self.line(&format!(
                        "  {} = :{}",
                        v.name.as_str().to_uppercase(),
                        v.name
                    ));
                }
                self.line("end");
            }
            Target::Lua => {
                let items: Vec<String> = def
                    .variants
                    .iter()
                    .map(|v| format!("{0} = \"{0}\"", v.name))
                    .collect();
                self.line(&format!("local {name} = {{ {} }}", items.join(", ")));
            }
        }
    }
}

// ----- helpers ------------------------------------------------------------------------

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c == '_' || c.is_alphabetic())
        && chars.all(|c| c == '_' || c.is_alphanumeric())
}

fn is_stringish(e: &Expr) -> bool {
    matches!(e.kind, ExprKind::Str(_) | ExprKind::Interp(_))
}

fn escape_inner(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
}

fn returns_value(e: &Expr) -> bool {
    !matches!(
        e.kind,
        ExprKind::Assign { .. }
            | ExprKind::CompoundAssign { .. }
            | ExprKind::IncDec { .. }
            | ExprKind::If { .. }
            | ExprKind::Match { .. }
            | ExprKind::Block(_)
            | ExprKind::Jump(_)
            | ExprKind::Yield { .. }
    ) && !matches!(&e.kind, ExprKind::Call { callee, .. } if matches!(&callee.kind, ExprKind::Name(n) if matches!(&*n.as_str(), "print" | "println" | "puts" | "say")))
}

fn binop_symbol(op: BinOp, t: Target) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::FloorDiv => match t {
            Target::Python | Target::Chimera | Target::Lua => "//",
            _ => "/",
        },
        BinOp::Mod => "%",
        BinOp::Pow => match t {
            Target::Lua => "^",
            _ => "**",
        },
        BinOp::MatMul => "@",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => match t {
            Target::Lua => "~",
            _ => "^",
        },
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        BinOp::Concat => match t {
            Target::Lua => "..",
            _ => "+",
        },
        BinOp::Pipe => "|>",
    }
}

fn cmp_symbol(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq | CmpOp::StrictEq | CmpOp::Is | CmpOp::Approx => "==",
        CmpOp::Ne | CmpOp::StrictNe | CmpOp::IsNot => "!=",
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Gt => ">",
        CmpOp::Ge => ">=",
        CmpOp::In => "in",
        CmpOp::NotIn => "not in",
        CmpOp::Spaceship => "<=>",
    }
}

fn pattern_text(p: &Pattern, t: Target) -> String {
    match p {
        Pattern::Wildcard => "_".into(),
        Pattern::Bind(n) => n.to_string(),
        Pattern::Literal(e) => match &e.kind {
            ExprKind::Int(i) => i.to_string(),
            ExprKind::Str(s) => format!("\"{}\"", escape_inner(s)),
            ExprKind::Bool(b) => if t.pythonic() {
                if *b { "True" } else { "False" }
            } else if *b {
                "true"
            } else {
                "false"
            }
            .into(),
            ExprKind::Nil => if t.pythonic() { "None" } else { "nil" }.into(),
            ExprKind::Float(f) => crate::interp::format::fmt_float(*f),
            _ => "_".into(),
        },
        Pattern::Tuple(items) => format!(
            "({})",
            items
                .iter()
                .map(|p| pattern_text(p, t))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Pattern::List { items, rest } => {
            let mut parts: Vec<String> = items.iter().map(|p| pattern_text(p, t)).collect();
            if let Some((at, name)) = rest {
                let r = match (t, name) {
                    (Target::JavaScript, Some(n)) => format!("...{n}"),
                    (_, Some(n)) => format!("*{n}"),
                    (_, None) => "*_".into(),
                };
                parts.insert((*at).min(parts.len()), r);
            }
            format!("[{}]", parts.join(", "))
        }
        Pattern::Map(entries, rest) => {
            let mut parts: Vec<String> = entries
                .iter()
                .map(|(k, p)| {
                    let key = match &k.kind {
                        ExprKind::Str(s) => s.to_string(),
                        _ => "_".into(),
                    };
                    format!("\"{key}\": {}", pattern_text(p, t))
                })
                .collect();
            if let Some(r) = rest {
                parts.push(format!("**{r}"));
            }
            format!("{{{}}}", parts.join(", "))
        }
        Pattern::Default(inner, _) => pattern_text(inner, t),
        Pattern::Ctor {
            path, args, fields, ..
        } => {
            let name = path
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(".");
            let mut parts: Vec<String> = args.iter().map(|p| pattern_text(p, t)).collect();
            parts.extend(
                fields
                    .iter()
                    .map(|(n, p)| format!("{n}={}", pattern_text(p, t))),
            );
            format!("{name}({})", parts.join(", "))
        }
        Pattern::Or(alts) => alts
            .iter()
            .map(|p| pattern_text(p, t))
            .collect::<Vec<_>>()
            .join(" | "),
        Pattern::Range { .. } => "_".into(),
        Pattern::CaseEq(_) | Pattern::In(_) => "_".into(),
        Pattern::Type(ty, inner) => {
            let ty = crate::interp::format::type_expr_name(ty);
            match inner {
                Some(p) => format!("{ty}() as {}", pattern_text(p, t)),
                None => format!("{ty}()"),
            }
        }
        Pattern::As(inner, n) => format!("{} as {n}", pattern_text(inner, t)),
    }
}

fn strip_break(b: &Block) -> Block {
    let mut stmts = b.stmts.clone();
    if matches!(stmts.last().map(|s| &s.kind), Some(StmtKind::Break)) {
        stmts.pop();
    }
    Block::new(stmts)
}

/// Field names assigned through `self.name = …` anywhere in a body.
fn collect_self_fields(b: &Block, out: &mut Vec<String>) {
    fn expr(e: &Expr, out: &mut Vec<String>) {
        match &e.kind {
            ExprKind::Assign { target, value } | ExprKind::CompoundAssign { target, value, .. } => {
                if let ExprKind::Member { object, name, .. } = &target.kind
                    && (matches!(object.kind, ExprKind::SelfRef)
                        || matches!(&object.kind, ExprKind::Name(n) if n.is("self") || n.is("this")))
                    && !out.iter().any(|o| o == &*name.as_str())
                {
                    out.push(name.to_string());
                }
                expr(value, out);
            }
            ExprKind::If {
                then, otherwise, ..
            } => {
                collect_self_fields(then, out);
                if let Some(o) = otherwise {
                    collect_self_fields(o, out);
                }
            }
            ExprKind::Block(b) => collect_self_fields(b, out),
            _ => {}
        }
    }
    for s in &b.stmts {
        match &s.kind {
            StmtKind::Expr(e) => expr(e, out),
            StmtKind::While { body, .. }
            | StmtKind::For { body, .. }
            | StmtKind::Loop(body)
            | StmtKind::Repeat { body, .. } => {
                collect_self_fields(body, out);
            }
            _ => {}
        }
    }
}

/// Whether a loop body continues (outside nested loops): Lua needs a
/// `::continue::` label for `goto continue`.
fn block_has_continue(b: &Block) -> bool {
    b.stmts.iter().any(|s| match &s.kind {
        StmtKind::Continue => true,
        StmtKind::Expr(Expr {
            kind: ExprKind::If {
                then, otherwise, ..
            },
            ..
        }) => block_has_continue(then) || otherwise.as_ref().is_some_and(block_has_continue),
        StmtKind::Expr(Expr {
            kind: ExprKind::Block(b),
            ..
        }) => block_has_continue(b),
        _ => false,
    })
}

fn uses_math(stmts: &[Stmt]) -> bool {
    // Cheap check: the Debug text mentions a math function by name.
    let text = format!("{stmts:?}");
    ["\"sqrt\"", "\"sin\"", "\"cos\"", "\"floor\"", "\"pi\""]
        .iter()
        .any(|m| text.contains(m))
        || text.contains("Sqrt")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(src: &str, t: Target) -> String {
        let (stmts, errors) = crate::parser::parse(src, 0);
        assert!(errors.is_empty(), "{errors:?}");
        convert(&stmts, src, t)
    }

    #[test]
    fn functions_and_loops_translate() {
        let src = "def add(a, b):\n    return a + b\nfor i in range(3):\n    print(add(i, 1))\n";
        let js = conv(src, Target::JavaScript);
        assert!(js.contains("function add(a, b) {"), "{js}");
        assert!(js.contains("for (let i = 0; i < 3; i++) {"), "{js}");
        assert!(js.contains("console.log(add(i, 1));"), "{js}");
        let rb = conv(src, Target::Ruby);
        assert!(rb.contains("def add(a, b)"), "{rb}");
        assert!(rb.contains("puts add(i, 1)"), "{rb}");
        let lua = conv(src, Target::Lua);
        assert!(lua.contains("local function add(a, b)"), "{lua}");
        assert!(lua.contains("end"), "{lua}");
    }

    #[test]
    fn rust_ranges_and_interpolation() {
        let src = "fn main() {\n    for i in 0..3 {\n        println!(\"{}\", i);\n    }\n}\nname = \"x\"\nputs \"hi #{name}\"\n";
        let py = conv(src, Target::Python);
        assert!(py.contains("for i in range(0, 3):"), "{py}");
        assert!(py.contains("print(f\"hi {name}\")"), "{py}");
        let lua = conv(src, Target::Lua);
        assert!(lua.contains("for i = 0, 3 - 1 do"), "{lua}");
    }

    #[test]
    fn classes_translate() {
        let src = "class Dog(Animal):\n    def speak(self):\n        return \"woof\"\n";
        let js = conv(src, Target::JavaScript);
        assert!(js.contains("class Dog extends Animal {"), "{js}");
        assert!(js.contains("speak() {"), "{js}");
        let rb = conv(src, Target::Ruby);
        assert!(rb.contains("class Dog < Animal"), "{rb}");
    }
}

/// Guesses which language a snippet is written in, from telltale tokens.
/// Used by `chimera explain`; the parser itself never needs to know.
pub fn detect_dialect(text: &str) -> &'static str {
    let t = text.trim();
    let first = t.split_whitespace().next().unwrap_or("");
    let has = |s: &str| t.contains(s);
    let starts = |s: &str| t.starts_with(s);
    let upper_sql = ["SELECT ", "INSERT INTO", "UPDATE ", "DELETE FROM"];
    let scores: [(&'static str, i32); 17] = [
        ("SQL", upper_sql.iter().any(|k| t.contains(k)) as i32 * 5),
        (
            "LINQ (C#)",
            (has(" in ") && starts("from ") && has("select")) as i32 * 5
                + (has("var ") && has("from ") && has(" select ")) as i32 * 3,
        ),
        (
            "English",
            (["set ", "say ", "add ", "repeat ", "increase ", "the "]
                .contains(&format!("{first} ").as_str())) as i32
                * 3
                + (has(" plus ") || has(" times\n") || has(" to ") && starts("set")) as i32 * 2,
        ),
        ("English + AI (please)", starts("please") as i32 * 6),
        (
            "Chimera AI",
            (starts("ai ") || has("= ai ") || has("ask \"") || has("ai(")) as i32 * 3,
        ),
        (
            "Chimera ML",
            (starts("model ") || starts("train ") || has("∇") || has("′") || has("tensor ["))
                as i32
                * 4,
        ),
        (
            "Python",
            (starts("def ") && t.contains("):")) as i32 * 4
                + (starts("class ") && first_line_ends(t, ':')) as i32 * 4
                + (has("elif ")
                    || has("lambda ")
                    || has(" for ") && has(" in ") && has("[")
                    || has("print(f\"")) as i32
                    * 2
                + (starts("import ") || starts("from ") && has(" import ")) as i32 * 2,
        ),
        (
            "JavaScript",
            (starts("const ")
                || starts("let ") && !has("let mut") && has(";")
                || starts("function")) as i32
                * 3
                + (has("=>") || has("console.log") || has("===")) as i32 * 2,
        ),
        (
            "Rust",
            (starts("fn ")
                || starts("impl ")
                || has("let mut ")
                || has("println!")
                || has("vec![")
                || has("#[derive")) as i32
                * 4
                + (has("&self") || has("->") && has("fn ")) as i32 * 2,
        ),
        (
            "Go",
            (starts("func ") || starts("package ") || has(":= ") || has("fmt.Print")) as i32 * 4,
        ),
        (
            "C/C++",
            (starts("int ")
                || starts("void ")
                || starts("#include")
                || has("printf(")
                || has("std::")
                || has("cout <<")) as i32
                * 3,
        ),
        (
            "Java/C#",
            (starts("public ")
                || starts("private ")
                || has("System.out")
                || has("Console.Write")
                || has("new ") && has(";")) as i32
                * 3,
        ),
        (
            "Ruby",
            (has("puts ")
                || has(" do |")
                || has("{ |")
                || starts("def ") && !has(":")
                || has(".each")
                || has("#{")) as i32
                * 3
                + (t.ends_with("end") && !has("then")) as i32 * 2,
        ),
        (
            "Lua",
            (starts("local ")
                || has(" then") && has("end")
                || has("~=")
                || has("..") && has("\"")
                || has("ipairs(")
                || has("pairs(")) as i32
                * 3,
        ),
        (
            "Haskell",
            (has(" :: ")
                || has(" <- ")
                || has("$ ")
                || has("where") && has(" = ")
                || has("\\x ->")
                || has("putStrLn")) as i32
                * 3,
        ),
        (
            "OCaml",
            (starts("let rec ")
                || starts("let () ")
                || has(" in\n")
                || has("Printf.printf")
                || has("(*")) as i32
                * 3
                + (has("match ") && has(" with")) as i32 * 2,
        ),
        (
            "Kotlin/Swift",
            (starts("fun ")
                || starts("val ")
                || starts("func ") && has("->")
                || starts("data class")
                || has("guard ")
                || has("\\(")) as i32
                * 3,
        ),
    ];
    let mut best = ("Chimera", 0);
    for (name, score) in scores {
        if score > best.1 {
            best = (name, score);
        }
    }
    best.0
}

fn first_line_ends(t: &str, c: char) -> bool {
    t.lines().next().is_some_and(|l| l.trim_end().ends_with(c))
}

#[cfg(test)]
mod dialect_tests {
    use super::detect_dialect;

    #[test]
    fn detects_common_dialects() {
        assert_eq!(detect_dialect("def f(x):\n    return x"), "Python");
        assert_eq!(detect_dialect("fn main() { println!(\"hi\"); }"), "Rust");
        assert_eq!(
            detect_dialect("const xs = [1, 2].map(x => x * 2);"),
            "JavaScript"
        );
        assert_eq!(detect_dialect("func main() { fmt.Println(1) }"), "Go");
        assert_eq!(detect_dialect("SELECT name FROM people"), "SQL");
        assert_eq!(detect_dialect("please sort xs"), "English + AI (please)");
        assert_eq!(detect_dialect("local t = {}"), "Lua");
    }
}
