//! Patterns (for `match`, destructuring and parameters) and runtime-checked
//! type annotations.

use super::{PResult, Parser, describe};
use crate::ast::*;
use crate::lexer::Tok;
use crate::symbol::Sym;

/// Words treated as type names in C/Java/Go-style declarations.
const TYPE_WORDS: &[&str] = &[
    "int", "long", "short", "char", "byte", "float", "double", "bool", "boolean", "void", "string",
    "auto", "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "isize", "usize",
    "f32", "f64", "uint", "ulong", "size_t", "decimal", "number", "any", "str", "rune", "complex",
    "integer", "real", "object", "dynamic", "unit", "int8", "int16", "int32", "int64", "uint8",
    "uint16", "uint32", "uint64", "float32", "float64", "error",
];

/// Whether `w` is a lowercase built-in type name (`int`, `f64`, `string`, …).
pub(crate) fn is_type_word(w: &str) -> bool {
    TYPE_WORDS.contains(&w)
}

/// Whether `w` starts with an uppercase letter (type/constructor names).
pub(crate) fn looks_capitalized(w: &str) -> bool {
    w.chars().next().is_some_and(char::is_uppercase)
}

impl Parser<'_> {
    /// A binding pattern: a name, `_`, or a tuple/list/map destructuring.
    pub(crate) fn binding_pattern(&mut self) -> PResult<Pattern> {
        self.eat_word("mut");
        self.eat_op("&");
        match &self.peek().tok {
            Tok::Op("[") => self.list_pattern(Self::binding_pattern),
            Tok::Op("(") => {
                let start = self.bump().span;
                let items = self.nested(|p| {
                    let mut items = Vec::new();
                    while !p.at_op(")") && !p.at_eof() {
                        items.push(p.binding_pattern()?);
                        if !p.eat_op(",") {
                            break;
                        }
                    }
                    p.expect_op(")", "to close the pattern")?;
                    Ok(items)
                })?;
                let _ = start;
                if items.len() == 1 && !self.prev_was_trailing_comma() {
                    Ok(items.into_iter().next().unwrap_or(Pattern::Wildcard))
                } else {
                    Ok(Pattern::Tuple(items))
                }
            }
            Tok::Op("{") => self.map_pattern(Self::binding_pattern),
            Tok::Ident(w) if w == "_" => {
                self.bump();
                Ok(Pattern::Wildcard)
            }
            Tok::Ident(_) => {
                let (name, _) = self.expect_ident("")?;
                Ok(Pattern::Bind(name))
            }
            _ => {
                let span = self.peek().span;
                let found = describe(self.peek());
                self.err(span, format!("expected a name or pattern, found {found}"))
            }
        }
    }

    fn prev_was_trailing_comma(&self) -> bool {
        // `(x,)` — the token before `)` is a comma.
        self.pos >= 2 && self.nth_back(2).is_op(",")
    }

    fn nth_back(&self, n: usize) -> &crate::lexer::Token {
        &self.toks_ref()[self.pos_ref() - n]
    }

    /// `[a, b, ...rest]` with any element parser.
    fn list_pattern(&mut self, elem: fn(&mut Self) -> PResult<Pattern>) -> PResult<Pattern> {
        self.expect_op("[", "")?;
        self.nested(|p| {
            let mut items = Vec::new();
            let mut rest = None;
            while !p.at_op("]") && !p.at_eof() {
                if p.at_op("...") || p.at_op("..") || p.at_op("*") {
                    p.bump();
                    let name = match &p.peek().tok {
                        Tok::Ident(w) if w != "_" => Some(p.expect_ident("")?.0),
                        Tok::Ident(_) => {
                            p.bump();
                            None
                        }
                        _ => None,
                    };
                    rest = Some((items.len(), name));
                } else if p.at_op(",") {
                    // A hole: `[first, , third]`.
                    items.push(Pattern::Wildcard);
                } else {
                    let pat = elem(p)?;
                    items.push(p.pattern_default(pat)?);
                }
                if !p.eat_op(",") {
                    break;
                }
            }
            p.expect_op("]", "to close the list pattern")?;
            Ok(Pattern::List { items, rest })
        })
    }

    /// `{name, age: a, ...rest}`.
    fn map_pattern(&mut self, elem: fn(&mut Self) -> PResult<Pattern>) -> PResult<Pattern> {
        self.expect_op("{", "")?;
        self.nested(|p| {
            let mut entries = Vec::new();
            let mut rest = None;
            loop {
                p.skip_newlines();
                if p.at_op("}") || p.at_eof() {
                    break;
                }
                if p.at_op("...") || p.at_op("..") || p.at_op("**") {
                    p.bump();
                    if let Some(w) = p.peek().ident() {
                        rest = Some(Sym::new(w));
                        p.bump();
                    }
                } else {
                    let t = p.bump();
                    let key_name = match &t.tok {
                        Tok::Ident(w) => w.clone(),
                        Tok::Str(s) => s.as_plain().unwrap_or_default(),
                        _ => {
                            let found = describe(&t);
                            return p.err(
                                t.span,
                                format!("expected a key in the pattern, found {found}"),
                            );
                        }
                    };
                    let key = Expr::new(ExprKind::Str(key_name.as_str().into()), t.span);
                    let pat = if p.eat_op(":") {
                        elem(p)?
                    } else {
                        Pattern::Bind(Sym::new(&key_name))
                    };
                    entries.push((key, p.pattern_default(pat)?));
                }
                p.skip_newlines();
                if !p.eat_op(",") {
                    break;
                }
            }
            p.skip_newlines();
            p.expect_op("}", "to close the map pattern")?;
            Ok(Pattern::Map(entries, rest))
        })
    }

    /// `pat = default` inside a destructuring pattern.
    fn pattern_default(&mut self, pat: Pattern) -> PResult<Pattern> {
        if self.at_op("=") {
            self.bump();
            let d = self.expr_no_assign()?;
            return Ok(Pattern::Default(Box::new(pat), d));
        }
        Ok(pat)
    }

    /// A full `match` pattern, including alternatives and `as` bindings.
    pub(crate) fn match_pattern(&mut self) -> PResult<Pattern> {
        let first = self.as_pattern()?;
        if !(self.at_op("|") && !self.nth_is_op(1, "|")) {
            return Ok(first);
        }
        let mut alts = vec![first];
        while self.at_op("|") {
            self.bump();
            alts.push(self.as_pattern()?);
        }
        Ok(Pattern::Or(alts))
    }

    fn as_pattern(&mut self) -> PResult<Pattern> {
        let mut p = self.primary_pattern()?;
        // Haskell/OCaml cons outside parentheses: `x :: xs`.
        if self.at_op("::") {
            self.bump();
            let tail = self.as_pattern()?;
            p = cons(p, tail);
        }
        if self.at_word("as") && self.nth(1).ident().is_some() {
            self.bump();
            let (name, _) = self.expect_ident("after `as`")?;
            p = Pattern::As(Box::new(p), name);
        }
        Ok(p)
    }

    fn literal_pattern_expr(&mut self) -> PResult<Expr> {
        let start = self.peek().span;
        let neg = self.eat_op("-");
        let t = self.bump();
        let kind = match &t.tok {
            Tok::Int(v) => ExprKind::Int(if neg { -v } else { *v }),
            Tok::Float(v) => ExprKind::Float(if neg { -v } else { *v }),
            Tok::BigInt(s) => {
                let text = if neg { format!("-{s}") } else { s.clone() };
                ExprKind::BigInt(std::rc::Rc::new(
                    crate::bigint::BigInt::parse(&text).unwrap_or_default(),
                ))
            }
            Tok::Str(s) => ExprKind::Str(s.as_plain().unwrap_or_default().into()),
            _ => {
                let found = describe(&t);
                return self.err(t.span, format!("expected a literal, found {found}"));
            }
        };
        Ok(Expr::new(kind, self.span_from(start)))
    }

    fn at_literal(&self) -> bool {
        let t = self.peek();
        matches!(
            t.tok,
            Tok::Int(_) | Tok::Float(_) | Tok::BigInt(_) | Tok::Str(_)
        ) || (t.is_op("-")
            && matches!(
                self.nth(1).tok,
                Tok::Int(_) | Tok::Float(_) | Tok::BigInt(_)
            ))
    }

    fn primary_pattern(&mut self) -> PResult<Pattern> {
        if self.at_literal() {
            let lo = self.literal_pattern_expr()?;
            if self.at_op("..") || self.at_op("..=") || self.at_op("...") || self.at_word("to") {
                let inclusive = !self.bump().is_op("..");
                let hi = self.literal_pattern_expr()?;
                return Ok(Pattern::Range { lo, hi, inclusive });
            }
            return Ok(Pattern::Literal(lo));
        }
        match &self.peek().tok {
            Tok::Op("[") => return self.list_pattern(Self::match_pattern),
            Tok::Op("{") => return self.map_pattern(Self::match_pattern),
            Tok::Op("(") => return self.paren_pattern(),
            Tok::Op(".") if self.nth(1).ident().is_some() => {
                // Swift enum shorthand `.north`, `.circle(r)`.
                self.bump();
                return self.ctor_pattern();
            }
            Tok::Ident(_) => {}
            _ => {
                let span = self.peek().span;
                let found = describe(self.peek());
                return self.err(span, format!("expected a pattern, found {found}"));
            }
        }
        let word = self.peek().ident().unwrap_or("").to_owned();
        match word.as_str() {
            "_" | "else" | "default" | "otherwise" => {
                self.bump();
                return Ok(Pattern::Wildcard);
            }
            "true" | "True" => {
                let t = self.bump();
                return Ok(Pattern::Literal(Expr::new(ExprKind::Bool(true), t.span)));
            }
            "false" | "False" => {
                let t = self.bump();
                return Ok(Pattern::Literal(Expr::new(ExprKind::Bool(false), t.span)));
            }
            "nil" | "null" | "None" | "NULL" | "undefined" => {
                let t = self.bump();
                return Ok(Pattern::Literal(Expr::new(ExprKind::Nil, t.span)));
            }
            "is" => {
                self.bump();
                let ty = self.type_expr()?;
                return Ok(Pattern::Type(ty, None));
            }
            "in" => {
                self.bump();
                let e = self.expr_bp(super::expr::bp::COMPARE + 1)?;
                return Ok(Pattern::In(e));
            }
            _ => {}
        }
        // `name @ subpattern` (Rust).
        if self.nth_is_op(1, "@") && !looks_capitalized(&word) {
            let (name, _) = self.expect_ident("")?;
            self.bump();
            let sub = self.primary_pattern()?;
            return Ok(Pattern::As(Box::new(sub), name));
        }
        // `n: Int` typed binding.
        if !looks_capitalized(&word)
            && self.nth_is_op(1, ":")
            && self
                .nth(2)
                .ident()
                .is_some_and(|w| looks_capitalized(w) || is_type_word(w))
            && !self.nth_is_op(3, "(")
        {
            let (name, _) = self.expect_ident("")?;
            self.bump();
            let ty = self.type_expr()?;
            return Ok(Pattern::Type(ty, Some(Box::new(Pattern::Bind(name)))));
        }
        self.ctor_pattern()
    }

    /// A constructor/class pattern or a plain binding.
    fn ctor_pattern(&mut self) -> PResult<Pattern> {
        let (first, _) = self.expect_ident("in the pattern")?;
        let mut path = vec![first];
        while (self.at_op("::") || self.at_op("."))
            && self.nth(1).ident().is_some()
            && !self.nth(1).space_before
        {
            self.bump();
            path.push(self.expect_ident("")?.0);
        }
        let last = *path.last().unwrap_or(&first);
        let capital = looks_capitalized(&last.as_str());
        if self.at_op("(") && (!self.peek().space_before || capital) {
            self.bump();
            let (args, fields) = self.nested(|p| {
                let mut args = Vec::new();
                let mut fields = Vec::new();
                while !p.at_op(")") && !p.at_eof() {
                    if p.peek().ident().is_some()
                        && (p.nth_is_op(1, "=") || p.nth_is_op(1, ":"))
                        && !p.nth_is_op(1, "::")
                    {
                        let (n, _) = p.expect_ident("")?;
                        p.bump();
                        fields.push((n, p.match_pattern()?));
                    } else if p.at_op("..") || p.at_op("...") {
                        p.bump();
                    } else {
                        args.push(p.match_pattern()?);
                    }
                    if !p.eat_op(",") {
                        break;
                    }
                }
                p.expect_op(")", "to close the constructor pattern")?;
                Ok((args, fields))
            })?;
            return Ok(Pattern::Ctor {
                path,
                args,
                fields,
                has_args: true,
            });
        }
        if self.at_op("{") && capital && !self.no_brace_struct_pattern() {
            self.bump();
            let fields = self.nested(|p| {
                let mut fields = Vec::new();
                loop {
                    p.skip_newlines();
                    if p.at_op("}") || p.at_eof() {
                        break;
                    }
                    if p.at_op("..") || p.at_op("...") {
                        p.bump();
                    } else {
                        let (n, _) = p.expect_ident("for the field")?;
                        let pat = if p.eat_op(":") {
                            p.match_pattern()?
                        } else {
                            Pattern::Bind(n)
                        };
                        fields.push((n, pat));
                    }
                    p.skip_newlines();
                    if !p.eat_op(",") {
                        break;
                    }
                }
                p.skip_newlines();
                p.expect_op("}", "to close the struct pattern")?;
                Ok(fields)
            })?;
            return Ok(Pattern::Ctor {
                path,
                args: Vec::new(),
                fields,
                has_args: true,
            });
        }
        if capital {
            // OCaml/Haskell juxtaposition: `Circle r`, `Node l v r`.
            let mut args = Vec::new();
            while self.same_line() && self.juxtaposed_pattern_ahead() {
                args.push(self.simple_pattern()?);
            }
            let has_args = !args.is_empty();
            return Ok(Pattern::Ctor {
                path,
                args,
                fields: Vec::new(),
                has_args,
            });
        }
        if path.len() > 1 {
            return Ok(Pattern::Ctor {
                path,
                args: Vec::new(),
                fields: Vec::new(),
                has_args: false,
            });
        }
        Ok(Pattern::Bind(first))
    }

    fn no_brace_struct_pattern(&self) -> bool {
        // In a Rust `match x { Foo { a } => … }` the brace is a struct
        // pattern; in `case Foo {` (Swift) it would be a body. Struct patterns
        // need a field name or `}`/`..` right after the brace.
        let n1 = self.nth(1);
        let n2 = self.nth(2);
        !(n1.is_op("}")
            || n1.is_op("..")
            || n1.ident().is_some() && (n2.is_op(",") || n2.is_op(":") || n2.is_op("}")))
    }

    fn juxtaposed_pattern_ahead(&self) -> bool {
        let t = self.peek();
        match &t.tok {
            Tok::Int(_) | Tok::Float(_) | Tok::Str(_) | Tok::BigInt(_) => true,
            Tok::Op("(") | Tok::Op("[") => t.space_before,
            Tok::Ident(w) => !matches!(
                w.as_str(),
                "if" | "when" | "as" | "then" | "where" | "do" | "and" | "or" | "in" | "of" | "is"
            ),
            _ => false,
        }
    }

    /// `( … )` in a pattern: tuple, grouping, or Haskell cons `(x:xs)`.
    fn paren_pattern(&mut self) -> PResult<Pattern> {
        self.bump();
        self.nested(|p| {
            if p.eat_op(")") {
                return Ok(Pattern::Tuple(Vec::new()));
            }
            let first = p.match_pattern()?;
            if p.at_op(":") || p.at_op("::") {
                let mut parts = vec![first];
                while p.at_op(":") || p.at_op("::") {
                    p.bump();
                    parts.push(p.match_pattern()?);
                }
                p.expect_op(")", "to close the pattern")?;
                let mut acc = parts.pop().unwrap_or(Pattern::Wildcard);
                while let Some(head) = parts.pop() {
                    acc = cons(head, acc);
                }
                return Ok(acc);
            }
            if p.eat_op(")") {
                return Ok(first);
            }
            let mut items = vec![first];
            while p.eat_op(",") {
                if p.at_op(")") {
                    break;
                }
                items.push(p.match_pattern()?);
            }
            p.expect_op(")", "to close the tuple pattern")?;
            Ok(Pattern::Tuple(items))
        })
    }

    /// An atomic pattern: names, literals, `_`, parenthesised or list patterns.
    pub(crate) fn simple_pattern(&mut self) -> PResult<Pattern> {
        if self.at_literal() {
            return Ok(Pattern::Literal(self.literal_pattern_expr()?));
        }
        match &self.peek().tok {
            Tok::Op("(") => self.paren_pattern(),
            Tok::Op("[") => self.list_pattern(Self::match_pattern),
            Tok::Ident(w) if w == "_" => {
                self.bump();
                Ok(Pattern::Wildcard)
            }
            Tok::Ident(w) if looks_capitalized(w) => {
                let (name, _) = self.expect_ident("")?;
                Ok(Pattern::Ctor {
                    path: vec![name],
                    args: Vec::new(),
                    fields: Vec::new(),
                    has_args: false,
                })
            }
            Tok::Ident(_) => Ok(Pattern::Bind(self.expect_ident("")?.0)),
            _ => {
                let span = self.peek().span;
                let found = describe(self.peek());
                self.err(span, format!("expected a parameter pattern, found {found}"))
            }
        }
    }

    // ----- types -------------------------------------------------------------

    /// A type annotation in any dialect's spelling.
    pub(crate) fn type_expr(&mut self) -> PResult<TypeExpr> {
        let first = self.type_postfix()?;
        if !self.at_op("|") || self.nth_is_op(1, "|") || self.no_bitor {
            return Ok(first);
        }
        let mut alts = vec![first];
        while self.eat_op("|") {
            alts.push(self.type_postfix()?);
        }
        Ok(TypeExpr::Union(alts))
    }

    fn type_postfix(&mut self) -> PResult<TypeExpr> {
        let mut t = self.type_atom()?;
        loop {
            if self.at_op("[")
                && (self.nth_is_op(1, "]")
                    || matches!(self.nth(1).tok, Tok::Int(_)) && self.nth_is_op(2, "]"))
            {
                self.bump();
                if !self.at_op("]") {
                    self.bump();
                }
                self.bump();
                t = TypeExpr::List(Box::new(t));
            } else if self.at_op("?") && !self.peek().space_before {
                self.bump();
                t = TypeExpr::Optional(Box::new(t));
            } else if (self.at_op("*") || self.at_op("&")) && !self.nth(1).is_op("(") {
                self.bump();
            } else {
                return Ok(t);
            }
        }
    }

    /// One type without suffixes.
    pub(crate) fn type_atom(&mut self) -> PResult<TypeExpr> {
        while self.at_op("&")
            || self.at_op("*")
            || self.at_any_word(&[
                "mut", "const", "impl", "dyn", "struct", "unsigned", "signed",
            ])
        {
            self.bump();
        }
        if self.at_op("(") {
            self.bump();
            let items = self.nested(|p| {
                let mut items = Vec::new();
                while !p.at_op(")") && !p.at_eof() {
                    items.push(p.type_expr()?);
                    if !p.eat_op(",") {
                        break;
                    }
                }
                p.expect_op(")", "in the type")?;
                Ok(items)
            })?;
            if self.eat_op("->") || self.eat_op("=>") {
                self.type_expr()?;
                return Ok(TypeExpr::Any);
            }
            if items.len() == 1 {
                return Ok(items.into_iter().next().unwrap_or(TypeExpr::Any));
            }
            return Ok(TypeExpr::Tuple(items));
        }
        if self.at_op("[")
            && (self.nth_is_op(1, "]")
                || matches!(self.nth(1).tok, Tok::Int(_)) && self.nth_is_op(2, "]"))
        {
            // Go slices and arrays: `[]int`, `[5]string`.
            self.bump();
            if !self.at_op("]") {
                self.bump();
            }
            self.bump();
            let elem = self.type_postfix()?;
            return Ok(TypeExpr::List(Box::new(elem)));
        }
        if self.at_word("chan") && self.nth(1).ident().is_some() {
            self.bump();
            self.type_postfix()?;
            return Ok(TypeExpr::Any);
        }
        if self.at_op("[") {
            self.bump();
            let inner = self.nested(|p| {
                let k = p.type_expr()?;
                if p.eat_op(":") || p.eat_op(";") {
                    let _v = p.type_expr()?;
                    p.expect_op("]", "in the type")?;
                    return Ok(TypeExpr::Named(Sym::new("Map"), Vec::new()));
                }
                p.expect_op("]", "in the type")?;
                Ok(TypeExpr::List(Box::new(k)))
            })?;
            return Ok(inner);
        }
        if self.at_any_word(&["fn", "func", "Fn", "FnMut", "FnOnce", "function"])
            && (self.nth_is_op(1, "(") || self.nth_is_op(1, "<"))
        {
            self.bump();
            self.skip_generics();
            if self.at_op("(") {
                self.skip_balanced();
            }
            if self.eat_op("->") || self.eat_op(":") {
                self.type_expr()?;
            }
            return Ok(TypeExpr::Any);
        }
        let (mut name, _) = self.expect_ident("for the type")?;
        // Kotlin/Swift nullable types lexed with the `?` attached: `String?`.
        if let Some(base) = name.as_str().strip_suffix('?') {
            return Ok(TypeExpr::Optional(Box::new(TypeExpr::Named(
                Sym::new(base),
                Vec::new(),
            ))));
        }
        while (self.at_op(".") || self.at_op("::"))
            && self.nth(1).ident().is_some()
            && !self.nth(1).space_before
        {
            self.bump();
            name = self.expect_ident("")?.0;
        }
        if name.is("map") && self.at_op("[") {
            // Go: `map[string]int`.
            self.bump();
            self.nested(|p| {
                p.type_expr()?;
                p.expect_op("]", "in the map type")
            })?;
            self.type_postfix()?;
            return Ok(TypeExpr::Named(Sym::new("Map"), Vec::new()));
        }
        let mut args = Vec::new();
        if self.at_op("<") && !self.peek().space_before {
            self.bump();
            loop {
                if self.at_op(">") {
                    break;
                }
                if self.at_op(">>") {
                    break;
                }
                args.push(self.type_expr()?);
                if !self.eat_op(",") {
                    break;
                }
            }
            self.close_angle()?;
        } else if self.at_op("[")
            && !self.peek().space_before
            && !self.nth_is_op(1, "]")
            && !matches!(self.nth(1).tok, Tok::Int(_))
        {
            // Python generics: `list[int]`, `dict[str, int]`.
            self.bump();
            self.nested(|p| {
                loop {
                    args.push(p.type_expr()?);
                    if !p.eat_op(",") {
                        break;
                    }
                }
                p.expect_op("]", "in the type")?;
                Ok(())
            })?;
        }
        let ty = match &*name.as_str() {
            "Optional" | "Option" | "Maybe" if args.len() == 1 => {
                TypeExpr::Optional(Box::new(args.pop().unwrap_or(TypeExpr::Any)))
            }
            "Union" if !args.is_empty() => TypeExpr::Union(args),
            "Any" | "any" | "object" | "Object" | "auto" | "var" | "dynamic" | "interface" => {
                TypeExpr::Any
            }
            _ => TypeExpr::Named(name, args),
        };
        Ok(ty)
    }

    /// Consumes one `>` closing a generic argument list, splitting `>>`.
    fn close_angle(&mut self) -> PResult<()> {
        if self.at_op(">>") {
            self.split_shift_token();
            return Ok(());
        }
        self.expect_op(">", "to close the generic arguments")?;
        Ok(())
    }
}

/// Builds the list pattern for `head : tail`.
fn cons(head: Pattern, tail: Pattern) -> Pattern {
    match tail {
        Pattern::Bind(n) => Pattern::List {
            items: vec![head],
            rest: Some((1, Some(n))),
        },
        Pattern::Wildcard => Pattern::List {
            items: vec![head],
            rest: Some((1, None)),
        },
        Pattern::List { mut items, rest } => {
            items.insert(0, head);
            let rest = rest.map(|(i, n)| (i + 1, n));
            Pattern::List { items, rest }
        }
        other => Pattern::List {
            items: vec![head, other],
            rest: None,
        },
    }
}
