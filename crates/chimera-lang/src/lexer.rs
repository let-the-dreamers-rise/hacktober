//! The lexer: source text → tokens.
//!
//! Chimera accepts the surface syntax of many languages at once, so the lexer
//! is deliberately permissive and keeps a little *layout* information on every
//! token instead of committing to one block style:
//!
//! * `line_start`, `col` and `indent` let the parser apply Python's offside
//!   rule when a block is introduced by `:` + newline, while ignoring
//!   indentation everywhere else (braces, `do … end`, `then … end`).
//! * `space_before` lets the parser tell `f (x)` from `f(x)`, `2x` (implicit
//!   multiplication) from `2 x`, and `foo -1` (command call) from `foo - 1`.
//!
//! Newlines are significant at the top level and inside `{ … }`, and are
//! suppressed inside `( … )` and `[ … ]`, so expressions can wrap freely.
//!
//! Comments: `// …`, `# …`, `/* … */` (nestable) and `-- …` (Lua/SQL/Haskell,
//! when followed by a space). Strings: `"…"`, `'…'`, `` `…` ``, triple-quoted,
//! raw (`r"…"`) and formatted (`f"…"`, `$"…"`) strings, with interpolation via
//! `${expr}`, `#{expr}`, `\(expr)` or, in formatted strings, `{expr:spec}`.

use chimera_core::{Diagnostic, Span};

/// A token's payload.
#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    /// An integer that fits in `i64`.
    Int(i64),
    /// An integer literal too large for `i64` (decimal digits).
    BigInt(String),
    /// A floating-point literal.
    Float(f64),
    /// A string literal, possibly with interpolated code.
    Str(StrLit),
    /// An identifier or (contextual) keyword.
    Ident(String),
    /// A Ruby-style instance variable: `@name`.
    IVar(String),
    /// An operator or punctuation, in canonical ASCII spelling.
    Op(&'static str),
    /// A significant line break.
    Newline,
    /// End of input.
    Eof,
}

/// A string literal split into literal text and interpolated code.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrLit {
    /// The pieces, in order.
    pub parts: Vec<StrPart>,
}

impl StrLit {
    /// The literal text if the string has no interpolation.
    pub fn as_plain(&self) -> Option<String> {
        let mut out = String::new();
        for part in &self.parts {
            match part {
                StrPart::Text(t) => out.push_str(t),
                StrPart::Code { .. } => return None,
            }
        }
        Some(out)
    }
}

/// One piece of a string literal.
#[derive(Clone, Debug, PartialEq)]
pub enum StrPart {
    /// Literal text with escapes already processed.
    Text(String),
    /// Interpolated source code at `offset..offset + len` in the file, with an
    /// optional Python-style format spec (`.2f`, `>10`, …).
    Code {
        /// Byte offset of the code in the source file.
        offset: usize,
        /// Length of the code in bytes.
        len: usize,
        /// Format spec after a top-level `:` in `{expr:spec}`.
        spec: Option<String>,
    },
}

/// A token with its location and layout facts.
#[derive(Clone, Debug, PartialEq)]
pub struct Token {
    /// The payload.
    pub tok: Tok,
    /// Byte range in the source.
    pub span: Span,
    /// Zero-based line number.
    pub line: usize,
    /// Display column (tabs count as four).
    pub col: usize,
    /// Column of the first token on this token's line.
    pub indent: usize,
    /// Whether this is the first token on its line.
    pub line_start: bool,
    /// Whether whitespace (or a comment) precedes the token.
    pub space_before: bool,
}

impl Token {
    /// Whether the token is the operator `op`.
    pub fn is_op(&self, op: &str) -> bool {
        matches!(&self.tok, Tok::Op(o) if *o == op)
    }

    /// Whether the token is the identifier/keyword `word`.
    pub fn is_word(&self, word: &str) -> bool {
        matches!(&self.tok, Tok::Ident(w) if w == word)
    }

    /// The identifier text, if this is an identifier.
    pub fn ident(&self) -> Option<&str> {
        match &self.tok {
            Tok::Ident(w) => Some(w),
            _ => None,
        }
    }
}

/// Format-spec marker for a `$name` interpolation that falls back to the
/// literal text when the name is undefined.
pub const SOFT_INTERPOLATION: &str = "\u{1}soft";

/// Operators, longest first so that maximal munch works by linear scan.
const OPERATORS: &[&str] = &[
    "...", "..=", "===", "!==", "<=>", "**=", "<<=", ">>=", "&&=", "||=", "??=", "..", "::", ":=",
    "//=", "=>", "==", "!=", "->", "<-", "<=", ">=", "<<", ">>", "+=", "-=", "*=", "/=", "%=",
    "^=", "&=", "//", "|=", "++", "--", "**", "&&", "||", "|>", "?.", "??", "?:", ".", ":", "=",
    "<", ">", "+", "-", "*", "/", "%", "^", "&", "|", "!", "?", "~", "@", ",", ";", "(", ")", "[",
    "]", "{", "}", "\\",
];

/// Lexes a whole source file.
pub fn lex(src: &str) -> (Vec<Token>, Vec<Diagnostic>) {
    lex_range(src, 0, src.len())
}

/// Lexes `src[start..end]`, reporting spans relative to the whole of `src`.
/// Used for code embedded in string interpolations.
pub fn lex_range(src: &str, start: usize, end: usize) -> (Vec<Token>, Vec<Diagnostic>) {
    let line_start_pos = src[..start].rfind('\n').map_or(0, |i| i + 1);
    let line = src[..start].matches('\n').count();
    let mut lexer = Lexer {
        src,
        pos: start,
        end,
        tokens: Vec::new(),
        errors: Vec::new(),
        brackets: Vec::new(),
        line,
        line_start_pos,
        at_line_start: true,
        cur_indent: 0,
        space_before: false,
    };
    lexer.run();
    (lexer.tokens, lexer.errors)
}

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
    end: usize,
    tokens: Vec<Token>,
    errors: Vec<Diagnostic>,
    brackets: Vec<char>,
    line: usize,
    line_start_pos: usize,
    at_line_start: bool,
    cur_indent: usize,
    space_before: bool,
}

fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

fn is_ident_continue(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

impl<'a> Lexer<'a> {
    fn peek(&self) -> Option<char> {
        self.src[self.pos..self.end].chars().next()
    }

    fn peek_at(&self, n: usize) -> Option<char> {
        self.src[self.pos..self.end].chars().nth(n)
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..self.end]
    }

    fn newlines_significant(&self) -> bool {
        matches!(self.brackets.last(), None | Some('{'))
    }

    fn column_of(&self, offset: usize) -> usize {
        // A token that started on an earlier line (a multi-line string)
        // measures from its own line's start.
        let line_start = if offset < self.line_start_pos {
            self.src[..offset].rfind('\n').map_or(0, |i| i + 1)
        } else {
            self.line_start_pos
        };
        self.src[line_start..offset]
            .chars()
            .map(|c| if c == '\t' { 4 } else { 1 })
            .sum()
    }

    /// Whether a `'` starts a Rust lifetime or loop label rather than a
    /// string: `'ident` not closed by another `'`, right after `&`, `<`,
    /// `+`, `break`/`continue`, or at the start of a line before `:`.
    fn at_lifetime(&self) -> bool {
        let rest = &self.src[self.pos + 1..self.end];
        let mut chars = rest.chars();
        if !chars.next().is_some_and(is_ident_start) {
            return false;
        }
        let ident_len: usize = rest
            .chars()
            .take_while(|&c| is_ident_continue(c))
            .map(char::len_utf8)
            .sum();
        let after = rest[ident_len..].chars().next();
        if after == Some('\'') {
            return false;
        }
        match self.tokens.last().map(|t| &t.tok) {
            Some(Tok::Op("&" | "<" | "+" | "&&")) => true,
            Some(Tok::Op(",")) => matches!(after, Some('>' | ',')),
            Some(Tok::Ident(w)) if w == "break" || w == "continue" => true,
            _ => self.at_line_start && after == Some(':'),
        }
    }

    /// `(* comment *)` (OCaml, Pascal), as opposed to the Haskell section
    /// `(* 3)`: the comment closes with `*)` before any other `)`.
    fn at_ocaml_comment(&self) -> bool {
        let rest = self.rest();
        if !rest.starts_with("(*") || rest.starts_with("(*)") {
            return false;
        }
        let body = &rest[2..];
        match (body.find("*)"), body.find(')')) {
            (Some(close), Some(paren)) => close + 1 == paren,
            _ => false,
        }
    }

    fn skip_ocaml_comment(&mut self) {
        let rest = self.rest();
        let len = rest[2..].find("*)").map_or(rest.len(), |i| i + 4);
        let text = &rest[..len];
        self.line += text.matches('\n').count();
        if let Some(nl) = text.rfind('\n') {
            self.line_start_pos = self.pos + nl + 1;
        }
        self.pos += len;
        self.space_before = true;
    }

    /// Ruby's `%w[a b c]` / `%i(a b)` word arrays (not after an operand,
    /// where `%` is modulo).
    fn at_word_array(&self) -> bool {
        let operand_before = self.tokens.last().is_some_and(|t| {
            t.line == self.line
                && (matches!(t.tok, Tok::Int(_) | Tok::Float(_) | Tok::BigInt(_) | Tok::Str(_))
                    || matches!(&t.tok, Tok::Ident(w) if w != "puts" && w != "p" && w != "print" && w != "return")
                    || matches!(t.tok, Tok::Op(")" | "]" | "}")))
        });
        !operand_before
            && matches!(self.peek_at(1), Some('w' | 'i' | 'W' | 'I'))
            && matches!(self.peek_at(2), Some('[' | '(' | '{' | '<'))
    }

    fn word_array(&mut self, start: usize) {
        let open = self.peek_at(2).unwrap_or('[');
        let close = match open {
            '(' => ')',
            '{' => '}',
            '<' => '>',
            _ => ']',
        };
        self.pos += 3;
        let body_start = self.pos;
        let body_end = self.src[body_start..self.end]
            .find(close)
            .map_or(self.end, |i| body_start + i);
        let body = &self.src[body_start..body_end];
        self.line += body.matches('\n').count();
        self.push(Tok::Op("["), start);
        for (i, w) in body.split_whitespace().enumerate() {
            if i > 0 {
                self.push(Tok::Op(","), start);
            }
            self.push(
                Tok::Str(StrLit {
                    parts: vec![StrPart::Text(w.to_owned())],
                }),
                start,
            );
        }
        self.pos = (body_end + 1).min(self.end);
        self.push(Tok::Op("]"), start);
        self.space_before = false;
    }

    /// Whether `#` is Lua's length operator rather than a comment: it is
    /// glued to an operand and follows an operator, `(`/`[`/`,`/`=` or
    /// `return`.
    fn at_length_op(&self) -> bool {
        let next = self.peek_at(1);
        if !next.is_some_and(|c| is_ident_start(c) || matches!(c, '(' | '{' | '"' | '\'')) {
            return false;
        }
        match self.tokens.last() {
            Some(t) if t.line == self.line => match &t.tok {
                Tok::Op(op) => !matches!(*op, ")" | "]" | "}"),
                Tok::Ident(w) => matches!(
                    w.as_str(),
                    "return" | "and" | "or" | "not" | "then" | "do" | "in" | "print"
                ),
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether `//` is floor division (Python, Lua) rather than a comment:
    /// it follows an operand on the same line and what comes after reads
    /// like an expression (`7 // 2`, `n // 10)`), not prose
    /// (`x = 1 // the answer`).
    fn at_floor_div(&self) -> bool {
        let operand_before = self.tokens.last().is_some_and(|t| {
            t.line == self.line
                && match &t.tok {
                    Tok::Int(_) | Tok::Float(_) | Tok::BigInt(_) => true,
                    Tok::Op(op) => matches!(*op, ")" | "]"),
                    Tok::Ident(w) => !matches!(w.as_str(), "return" | "else" | "then" | "do"),
                    _ => false,
                }
        });
        if !operand_before {
            return false;
        }
        let rest = &self.src[self.pos + 2..self.end];
        let line = rest.split('\n').next().unwrap_or("").trim();
        let Some(first) = line.chars().next() else {
            return false;
        };
        if !(first.is_ascii_digit() || is_ident_start(first) || first == '(' || first == '-') {
            return false;
        }
        if line.contains([':', '!', '?', '\'', '"']) && !line.contains('#') {
            return false;
        }
        // Prose has consecutive words: `// set x to 5`.
        let words: Vec<&str> = line
            .split('#')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        let wordy = |w: &str| {
            w.chars().next().is_some_and(char::is_alphabetic)
                && w.chars().all(|c| c.is_alphanumeric() || c == '_')
        };
        !words.windows(2).any(|p| wordy(p[0]) && wordy(p[1]))
    }

    fn push(&mut self, tok: Tok, start: usize) {
        let col = self.column_of(start);
        let line_start = self.at_line_start;
        if line_start {
            self.cur_indent = col;
        }
        self.tokens.push(Token {
            tok,
            span: Span::new(start, self.pos),
            line: self.line,
            col,
            indent: self.cur_indent,
            line_start,
            space_before: self.space_before || line_start,
        });
        self.at_line_start = false;
        self.space_before = false;
    }

    fn error(&mut self, span: Span, msg: impl Into<String>) {
        self.errors.push(
            Diagnostic::error(msg)
                .with_code("E0001")
                .with_label(span, ""),
        );
    }

    fn newline(&mut self) {
        // `self.pos` points just past the '\n'.
        self.line += 1;
        self.line_start_pos = self.pos;
        self.at_line_start = true;
    }

    fn run(&mut self) {
        while let Some(c) = self.peek() {
            let start = self.pos;
            match c {
                '\n' => {
                    self.pos += 1;
                    if self.newlines_significant()
                        && self
                            .tokens
                            .last()
                            .is_some_and(|t| !matches!(t.tok, Tok::Newline))
                    {
                        let line_start = self.at_line_start;
                        self.tokens.push(Token {
                            tok: Tok::Newline,
                            span: Span::new(start, start + 1),
                            line: self.line,
                            col: self.column_of(start),
                            indent: self.cur_indent,
                            line_start,
                            space_before: false,
                        });
                    }
                    self.newline();
                    self.space_before = false;
                }
                ' ' | '\t' | '\r' | '\u{feff}' => {
                    self.pos += c.len_utf8();
                    self.space_before = true;
                }
                '\\' if matches!(self.peek_at(1), Some('\n')) => {
                    self.pos += 2;
                    self.line += 1;
                    self.line_start_pos = self.pos;
                    self.space_before = true;
                }
                '\\' if matches!(self.peek_at(1), Some('\r')) && self.peek_at(2) == Some('\n') => {
                    self.pos += 3;
                    self.line += 1;
                    self.line_start_pos = self.pos;
                    self.space_before = true;
                }
                '#' if self.at_length_op() => {
                    // Lua's length operator: `#t`, `t[#t + 1]`, `print(#s)`.
                    self.pos += 1;
                    self.push(Tok::Op("#"), start);
                }
                '#' => self.skip_line_comment(),
                '/' if self.rest().starts_with("//") && !self.at_floor_div() => {
                    self.skip_line_comment()
                }
                '(' if self.at_ocaml_comment() => self.skip_ocaml_comment(),
                '/' if self.rest().starts_with("/*") => self.skip_block_comment(),
                '-' if self.is_dash_comment() => self.skip_line_comment(),
                '0'..='9' => self.number(),
                '%' if self.at_word_array() => self.word_array(start),
                '\'' if self.at_lifetime() => {
                    // Rust lifetimes and loop labels (`&'a str`, `<'a>`,
                    // `'outer: loop`) carry no runtime meaning.
                    self.pos += 1;
                    let _ = self.ident_text();
                    if self.peek() == Some(':')
                        && self.peek_at(1) != Some(':')
                        && self.at_line_start
                    {
                        self.pos += 1;
                    }
                    self.space_before = true;
                }
                '"' | '\'' | '`' => self.string(start, ""),
                '$' => self.dollar(start),
                '@' if self.peek_at(1).is_some_and(is_ident_start) => {
                    self.pos += 1;
                    let name = self.ident_text();
                    self.push(Tok::IVar(name), start);
                }
                c if is_ident_start(c) => self.ident(start),
                _ => self.operator(start, c),
            }
        }
        let eof = self.end;
        self.pos = eof;
        if !matches!(
            self.tokens.last(),
            None | Some(Token {
                tok: Tok::Newline,
                ..
            })
        ) {
            let col = self.column_of(eof);
            self.tokens.push(Token {
                tok: Tok::Newline,
                span: Span::point(eof),
                line: self.line,
                col,
                indent: self.cur_indent,
                line_start: false,
                space_before: false,
            });
        }
        self.at_line_start = true;
        self.push(Tok::Eof, eof);
    }

    fn is_dash_comment(&self) -> bool {
        // `-- comment` (Lua, SQL, Haskell) but not `x--` or `--x`.
        self.rest().starts_with("--")
            && (self.at_line_start || self.space_before)
            && self
                .peek_at(2)
                .is_none_or(|c| c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '-')
    }

    fn skip_line_comment(&mut self) {
        while let Some(c) = self.peek() {
            if c == '\n' {
                break;
            }
            self.pos += c.len_utf8();
        }
        self.space_before = true;
    }

    fn skip_block_comment(&mut self) {
        let start = self.pos;
        self.pos += 2;
        let mut depth = 1;
        while depth > 0 {
            let rest = self.rest();
            if rest.is_empty() {
                self.error(Span::new(start, start + 2), "unterminated block comment");
                return;
            }
            if rest.starts_with("/*") {
                depth += 1;
                self.pos += 2;
            } else if rest.starts_with("*/") {
                depth -= 1;
                self.pos += 2;
            } else {
                let c = self.peek().unwrap_or(' ');
                self.pos += c.len_utf8();
                if c == '\n' {
                    self.line += 1;
                    self.line_start_pos = self.pos;
                }
            }
        }
        self.space_before = true;
    }

    fn ident_text(&mut self) -> String {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if is_ident_continue(c) {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
        self.src[start..self.pos].to_owned()
    }

    fn ident(&mut self, start: usize) {
        let mut name = self.ident_text();
        // String prefixes: f"…", r"…", b"…", s"…", fr"…", rf"…".
        if let Some(q) = self.peek()
            && matches!(q, '"' | '\'')
        {
            let lower = name.to_ascii_lowercase();
            if !lower.is_empty() && lower.len() <= 2 && lower.chars().all(|c| "frbsu".contains(c)) {
                self.string(start, &lower);
                return;
            }
        }
        // Haskell/OCaml primes: `abs'`, `foldl'`, `x''`.
        while self.peek() == Some('\'')
            && !self
                .peek_at(1)
                .is_some_and(|c| is_ident_continue(c) || c == '\'')
        {
            self.pos += 1;
            name.push('\'');
        }
        // Rust macros (`println!(…)`, `vec![…]`) and Ruby bang/predicate methods.
        // A trailing `?`/`!` belongs to a Ruby method name after `def`, `.`
        // or a symbol colon (`def empty?`, `xs.empty?`, `&:even?`).
        let method_context = match self.tokens.last().map(|t| &t.tok) {
            Some(Tok::Op("." | "&." | ":" | "&:")) => true,
            Some(Tok::Ident(w)) => w == "def",
            _ => false,
        };
        let ends_name = |c: Option<char>| {
            c.is_none_or(|c| c.is_whitespace() || matches!(c, ')' | ',' | ']' | '}' | ';'))
        };
        match (self.peek(), self.peek_at(1)) {
            (Some('!'), Some('(' | '[')) => {
                self.pos += 1;
                name.push('!');
            }
            (Some('?'), Some('(')) => {
                self.pos += 1;
                name.push('?');
            }
            (Some(c @ ('?' | '!')), next) if method_context && ends_name(next) => {
                self.pos += 1;
                name.push(c);
            }
            _ => {}
        }
        if name == "λ" {
            name = "lambda".to_owned();
        }
        self.push(Tok::Ident(name), start);
    }

    fn dollar(&mut self, start: usize) {
        match self.peek_at(1) {
            Some('"') | Some('\'') => {
                self.pos += 1;
                self.string(start, "f");
            }
            Some(c) if is_ident_start(c) => {
                // Perl/PHP/shell sigils: `$x` is just `x`.
                self.pos += 1;
                let name = self.ident_text();
                self.push(Tok::Ident(name), start);
            }
            Some(c) if c.is_ascii_digit() => {
                // Swift closure shorthand `$0`, `$1`.
                self.pos += 1;
                let digits = self.ident_text();
                self.push(Tok::Ident(format!("${digits}")), start);
            }
            Some(c) if c.is_whitespace() || c == '(' => {
                // Haskell application: `print $ sum xs`.
                self.pos += 1;
                self.push(Tok::Op("$"), start);
            }
            _ => {
                self.pos += 1;
                self.error(Span::new(start, self.pos), "stray `$`");
            }
        }
    }

    fn number(&mut self) {
        let start = self.pos;
        let after_dot = self
            .tokens
            .last()
            .is_some_and(|t| t.is_op(".") && !self.space_before);
        let rest = self.rest();
        let radix = if rest.starts_with("0x") || rest.starts_with("0X") {
            16
        } else if rest.starts_with("0b") || rest.starts_with("0B") {
            2
        } else if rest.starts_with("0o") || rest.starts_with("0O") {
            8
        } else {
            10
        };
        if radix != 10 {
            self.pos += 2;
            let digits_start = self.pos;
            while let Some(c) = self.peek() {
                if c.is_digit(radix) || c == '_' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            let digits: String = self.src[digits_start..self.pos]
                .chars()
                .filter(|&c| c != '_')
                .collect();
            match i64::from_str_radix(&digits, radix) {
                Ok(v) => self.push(Tok::Int(v), start),
                Err(_) => match crate::bigint::BigInt::parse_radix(&digits, radix) {
                    Some(big) => self.push(Tok::BigInt(big.to_string()), start),
                    None => {
                        self.error(Span::new(start, self.pos), "invalid number literal");
                        self.push(Tok::Int(0), start);
                    }
                },
            }
            self.number_suffix();
            return;
        }

        let mut is_float = false;
        self.eat_digits();
        if !after_dot
            && self.peek() == Some('.')
            && self.peek_at(1).is_some_and(|c| c.is_ascii_digit())
        {
            is_float = true;
            self.pos += 1;
            self.eat_digits();
        }
        if !after_dot && matches!(self.peek(), Some('e' | 'E')) {
            let sign = matches!(self.peek_at(1), Some('+' | '-'));
            let digit_at = if sign { 2 } else { 1 };
            if self.peek_at(digit_at).is_some_and(|c| c.is_ascii_digit()) {
                is_float = true;
                self.pos += digit_at;
                self.eat_digits();
            }
        }
        let text: String = self.src[start..self.pos]
            .chars()
            .filter(|&c| c != '_')
            .collect();
        if !is_float
            && matches!(self.peek(), Some('f' | 'F' | 'd' | 'D'))
            && !self.peek_at(1).is_some_and(is_ident_continue)
        {
            // C-style float suffix: `2f`, `1.5f`.
            self.pos += 1;
            is_float = true;
        }
        if is_float {
            match text.parse::<f64>() {
                Ok(v) => self.push(Tok::Float(v), start),
                Err(_) => {
                    self.error(Span::new(start, self.pos), "invalid float literal");
                    self.push(Tok::Float(0.0), start);
                }
            }
        } else {
            match text.parse::<i64>() {
                Ok(v) => self.push(Tok::Int(v), start),
                Err(_) => self.push(Tok::BigInt(text), start),
            }
            self.number_suffix();
        }
    }

    fn eat_digits(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_digit()
                || (c == '_' && self.peek_at(1).is_some_and(|d| d.is_ascii_digit()))
            {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Swallows integer suffixes such as `10L`, `10n`, `10u`, `10i64`.
    fn number_suffix(&mut self) {
        let rest = self.rest();
        for suffix in [
            "i64", "i32", "u64", "u32", "usize", "isize", "u8", "i8", "L", "l", "n", "u", "U",
        ] {
            if rest.starts_with(suffix)
                && !rest[suffix.len()..]
                    .chars()
                    .next()
                    .is_some_and(is_ident_continue)
            {
                self.pos += suffix.len();
                if let Some(last) = self.tokens.last_mut() {
                    last.span.end = self.pos;
                }
                return;
            }
        }
    }

    fn operator(&mut self, start: usize, c: char) {
        let unicode = match c {
            '×' | '·' => Some(Tok::Op("*")),
            '÷' => Some(Tok::Op("/")),
            '≤' => Some(Tok::Op("<=")),
            '≥' => Some(Tok::Op(">=")),
            '≠' => Some(Tok::Op("!=")),
            '→' => Some(Tok::Op("->")),
            '⇒' => Some(Tok::Op("=>")),
            '←' => Some(Tok::Op("<-")),
            '√' => Some(Tok::Op("√")),
            '∇' => Some(Tok::Op("∇")),
            '′' => Some(Tok::Op("′")),
            '″' => Some(Tok::Op("″")),
            '²' => Some(Tok::Op("²")),
            '³' => Some(Tok::Op("³")),
            '≈' => Some(Tok::Op("≈")),
            '∉' => Some(Tok::Op("∉")),
            '∧' => Some(Tok::Ident("and".into())),
            '∨' => Some(Tok::Ident("or".into())),
            '¬' => Some(Tok::Ident("not".into())),
            '∈' => Some(Tok::Ident("in".into())),
            _ => None,
        };
        if let Some(tok) = unicode {
            self.pos += c.len_utf8();
            self.push(tok, start);
            return;
        }
        let rest = self.rest();
        // OCaml float operators `+.`, `-.`, `*.`, `/.` behave like the plain ones.
        if let Some(op) = ["+.", "-.", "*.", "/."]
            .iter()
            .find(|op| rest.starts_with(**op))
            && !rest[2..].starts_with(|c: char| c.is_ascii_digit() || c == '.')
        {
            self.pos += 2;
            let plain = match *op {
                "+." => "+",
                "-." => "-",
                "*." => "*",
                _ => "/",
            };
            self.push(Tok::Op(plain), start);
            return;
        }
        let Some(op) = OPERATORS.iter().find(|op| rest.starts_with(**op)) else {
            self.pos += c.len_utf8();
            self.error(
                Span::new(start, self.pos),
                format!("unexpected character `{c}`"),
            );
            return;
        };
        self.pos += op.len();
        match *op {
            "(" | "[" | "{" => self.brackets.push(op.chars().next().unwrap_or('(')),
            ")" | "]" | "}" => {
                let open = match *op {
                    ")" => '(',
                    "]" => '[',
                    _ => '{',
                };
                if let Some(idx) = self.brackets.iter().rposition(|&b| b == open) {
                    self.brackets.truncate(idx);
                }
            }
            _ => {}
        }
        self.push(Tok::Op(op), start);
    }

    /// Lexes a string literal starting at the quote at `self.pos`.
    /// `token_start` includes any prefix; `prefix` is lowercase prefix letters.
    fn string(&mut self, token_start: usize, prefix: &str) {
        let quote = self.peek().unwrap_or('"');
        let raw = prefix.contains('r');
        let formatted = prefix.contains('f') || prefix.contains('s');
        let triple = self.rest().starts_with(&quote.to_string().repeat(3)) && quote != '`';
        let delim_len = if triple { 3 } else { 1 };
        self.pos += delim_len;
        let closing: String = quote.to_string().repeat(delim_len);

        // Which interpolation syntaxes are live in this literal.
        let dollar = !raw && (quote == '`' || (quote == '"' && !formatted));
        let hash = !raw && quote == '"' && !formatted;
        let swift = !raw && quote == '"' && !formatted;
        let braces = formatted;

        let mut parts = Vec::new();
        let mut text = String::new();
        loop {
            let rest = self.rest();
            if rest.is_empty() {
                self.error(
                    Span::new(token_start, self.pos),
                    "unterminated string literal",
                );
                break;
            }
            if rest.starts_with(closing.as_str()) {
                self.pos += delim_len;
                break;
            }
            let c = self.peek().unwrap_or(' ');
            if c == '\n' && !triple && quote != '`' {
                self.error(
                    Span::new(token_start, self.pos),
                    "unterminated string literal (strings end at the line break; use `\"\"\"` or a backtick string for multi-line text)",
                );
                break;
            }
            if c == '\\' && !raw {
                self.escape(&mut text, &mut parts, swift);
                continue;
            }
            if braces && c == '{' {
                if self.peek_at(1) == Some('{') {
                    text.push('{');
                    self.pos += 2;
                    continue;
                }
                self.interpolation(&mut text, &mut parts, 1, true);
                continue;
            }
            if braces && c == '}' && self.peek_at(1) == Some('}') {
                text.push('}');
                self.pos += 2;
                continue;
            }
            if (dollar && c == '$' || hash && c == '#') && self.peek_at(1) == Some('{') {
                self.interpolation(&mut text, &mut parts, 2, false);
                continue;
            }
            if dollar && quote == '"' && c == '$' && self.peek_at(1).is_some_and(is_ident_start) {
                // Kotlin/Groovy/PHP/shell `"$name"`. Soft: a name that isn't
                // defined stays literal text, so "$HOME" in other languages
                // is safe.
                let name_start = self.pos + 1;
                let len: usize = self.src[name_start..self.end]
                    .chars()
                    .take_while(|&c| is_ident_continue(c))
                    .map(char::len_utf8)
                    .sum();
                if !text.is_empty() {
                    parts.push(StrPart::Text(std::mem::take(&mut text)));
                }
                parts.push(StrPart::Code {
                    offset: name_start,
                    len,
                    spec: Some(SOFT_INTERPOLATION.to_owned()),
                });
                self.pos = name_start + len;
                continue;
            }
            if c == '\n' {
                text.push('\n');
                self.pos += 1;
                self.line += 1;
                self.line_start_pos = self.pos;
                continue;
            }
            text.push(c);
            self.pos += c.len_utf8();
        }
        if !text.is_empty() || parts.is_empty() {
            parts.push(StrPart::Text(text));
        }
        self.push(Tok::Str(StrLit { parts }), token_start);
    }

    fn escape(&mut self, text: &mut String, parts: &mut Vec<StrPart>, swift: bool) {
        let start = self.pos;
        self.pos += 1;
        let Some(c) = self.peek() else {
            return;
        };
        self.pos += c.len_utf8();
        match c {
            'n' => text.push('\n'),
            't' => text.push('\t'),
            'r' => text.push('\r'),
            '0' => text.push('\0'),
            'a' => text.push('\u{7}'),
            'b' => text.push('\u{8}'),
            'f' => text.push('\u{c}'),
            'v' => text.push('\u{b}'),
            'e' => text.push('\u{1b}'),
            '\n' => {
                self.line += 1;
                self.line_start_pos = self.pos;
            }
            'x' => {
                let hex: String = self.rest().chars().take(2).collect();
                match u8::from_str_radix(&hex, 16) {
                    Ok(b) if hex.len() == 2 => {
                        self.pos += 2;
                        text.push(char::from(b));
                    }
                    _ => self.error(Span::new(start, self.pos), "invalid `\\x` escape"),
                }
            }
            'u' => {
                let (digits, consumed) = if self.peek() == Some('{') {
                    let close = self.rest().find('}').unwrap_or(0);
                    (self.rest()[1..close.max(1)].to_owned(), close + 1)
                } else {
                    (self.rest().chars().take(4).collect::<String>(), 4)
                };
                match u32::from_str_radix(&digits, 16)
                    .ok()
                    .and_then(char::from_u32)
                {
                    Some(ch) => {
                        self.pos += consumed.min(self.end - self.pos);
                        text.push(ch);
                    }
                    None => self.error(Span::new(start, self.pos), "invalid `\\u` escape"),
                }
            }
            '(' if swift => {
                // Swift interpolation: "\(expr)".
                self.pos -= 1;
                self.interpolation_with(text, parts, 1, ('(', ')'), false);
            }
            other => text.push(other),
        }
    }

    fn interpolation(
        &mut self,
        text: &mut String,
        parts: &mut Vec<StrPart>,
        open_len: usize,
        allow_spec: bool,
    ) {
        self.interpolation_with(text, parts, open_len, ('{', '}'), allow_spec);
    }

    /// Splits off an interpolated expression. `self.pos` is at the start of
    /// the opener (`{`, `${`, `#{` or `(`), `open_len` bytes long.
    fn interpolation_with(
        &mut self,
        text: &mut String,
        parts: &mut Vec<StrPart>,
        open_len: usize,
        (open, close): (char, char),
        allow_spec: bool,
    ) {
        let start = self.pos;
        let code_start = start + open_len;
        let Some(code_end) = find_closing(self.src, code_start, self.end, open, close) else {
            self.error(
                Span::new(start, (start + open_len).min(self.end)),
                format!("unclosed interpolation: missing `{close}`"),
            );
            self.pos = self.end;
            return;
        };
        if !text.is_empty() {
            parts.push(StrPart::Text(std::mem::take(text)));
        }
        let code = &self.src[code_start..code_end];
        let (len, spec) = match allow_spec.then(|| split_format_spec(code)).flatten() {
            Some(colon) => (colon, Some(code[colon + 1..].to_owned())),
            None => (code.len(), None),
        };
        // Python conversions: `!r` and `!a` use repr (marked with a leading
        // `!r` in the spec), `!s` is the default.
        let c = &code[..len];
        let (trimmed_len, spec) = if c.ends_with("!r") || c.ends_with("!a") {
            (len - 2, Some(format!("!r{}", spec.unwrap_or_default())))
        } else if c.ends_with("!s") {
            (len - 2, spec)
        } else {
            (len, spec)
        };
        parts.push(StrPart::Code {
            offset: code_start,
            len: trimmed_len,
            spec,
        });
        self.line += code.matches('\n').count();
        self.pos = code_end + 1;
    }
}

/// Finds the byte index of the `close` matching an already-consumed `open`,
/// skipping nested brackets and quoted strings.
fn find_closing(src: &str, from: usize, end: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 0usize;
    let mut chars = src[from..end].char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        match c {
            '"' | '\'' | '`' => {
                let quote = c;
                while let Some((_, d)) = chars.next() {
                    if d == '\\' {
                        chars.next();
                    } else if d == quote {
                        break;
                    }
                }
            }
            c if c == open => depth += 1,
            c if c == close => {
                if depth == 0 {
                    return Some(from + i);
                }
                depth -= 1;
            }
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    None
}

/// Index of a top-level `:` that starts a format spec in `{expr:spec}`.
fn split_format_spec(code: &str) -> Option<usize> {
    let mut depth = 0i32;
    let bytes = code.as_bytes();
    let mut in_str: Option<u8> = None;
    let mut found = None;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = in_str {
            if b == b'\\' {
                i += 1;
            } else if b == q {
                in_str = None;
            }
        } else {
            match b {
                b'"' | b'\'' => in_str = Some(b),
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth -= 1,
                b':' if depth == 0 => {
                    let next = bytes.get(i + 1).copied();
                    let prev = i.checked_sub(1).map(|p| bytes[p]);
                    if next != Some(b':') && prev != Some(b':') && next != Some(b'=') {
                        found = Some(i);
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Tok> {
        let (toks, errs) = lex(src);
        assert!(errs.is_empty(), "unexpected lex errors: {errs:?}");
        toks.into_iter().map(|t| t.tok).collect()
    }

    fn ident(s: &str) -> Tok {
        Tok::Ident(s.into())
    }

    #[test]
    fn numbers() {
        assert_eq!(
            kinds("1 1_000 0xff 0b101 0o17 1.5 2e3 1.5e-2"),
            vec![
                Tok::Int(1),
                Tok::Int(1000),
                Tok::Int(255),
                Tok::Int(5),
                Tok::Int(15),
                Tok::Float(1.5),
                Tok::Float(2000.0),
                Tok::Float(0.015),
                Tok::Newline,
                Tok::Eof
            ]
        );
        assert_eq!(
            kinds("99999999999999999999")[0],
            Tok::BigInt("99999999999999999999".into())
        );
        assert_eq!(kinds("10L")[0], Tok::Int(10));
        assert_eq!(kinds("2f")[0], Tok::Float(2.0));
    }

    #[test]
    fn ranges_and_tuple_fields_are_not_floats() {
        assert_eq!(
            kinds("1..10")[..3],
            [Tok::Int(1), Tok::Op(".."), Tok::Int(10)]
        );
        assert_eq!(
            kinds("t.0.1")[..5],
            [
                ident("t"),
                Tok::Op("."),
                Tok::Int(0),
                Tok::Op("."),
                Tok::Int(1)
            ]
        );
    }

    #[test]
    fn comments_of_every_flavour() {
        let src = "a // a comment\nb # py\n/* block /* nested */ */ c\n-- lua\nd";
        let words: Vec<Tok> = kinds(src)
            .into_iter()
            .filter(|t| matches!(t, Tok::Ident(_)))
            .collect();
        assert_eq!(words, vec![ident("a"), ident("b"), ident("c"), ident("d")]);
        // `x--` is a decrement, not a comment.
        assert!(kinds("x--").contains(&Tok::Op("--")));
        // `//` after an operand and before an expression is floor division.
        assert!(kinds("7 // 2").contains(&Tok::Op("//")));
        assert!(kinds("x = n // k").contains(&Tok::Op("//")));
        assert!(!kinds("x = 1 // the answer").contains(&Tok::Op("//")));
        // OCaml comments vs Haskell sections.
        assert_eq!(kinds("(* note *) 1")[0], Tok::Int(1));
        assert!(kinds("(* 3)").contains(&Tok::Op("*")));
    }

    #[test]
    fn newlines_suppressed_inside_parens_but_not_braces() {
        let toks = kinds("f(1,\n2)\n{a\nb}");
        let newlines = toks.iter().filter(|t| **t == Tok::Newline).count();
        assert_eq!(newlines, 3);
    }

    #[test]
    fn strings_and_escapes() {
        let toks = kinds(r#""a\tb" 'c\'d' r"raw\n""#);
        assert_eq!(
            toks[0],
            Tok::Str(StrLit {
                parts: vec![StrPart::Text("a\tb".into())]
            })
        );
        assert_eq!(
            toks[1],
            Tok::Str(StrLit {
                parts: vec![StrPart::Text("c'd".into())]
            })
        );
        assert_eq!(
            toks[2],
            Tok::Str(StrLit {
                parts: vec![StrPart::Text("raw\\n".into())]
            })
        );
    }

    #[test]
    fn interpolation_flavours() {
        let src = r#""a ${x} b #{y}" f"{z:.2f}" `t ${w}` "\(v)""#;
        let (toks, errs) = lex(src);
        assert!(errs.is_empty());
        let code = |t: &Token| -> Vec<(String, Option<String>)> {
            match &t.tok {
                Tok::Str(s) => s
                    .parts
                    .iter()
                    .filter_map(|p| match p {
                        StrPart::Code { offset, len, spec } => {
                            Some((src[*offset..offset + len].to_owned(), spec.clone()))
                        }
                        StrPart::Text(_) => None,
                    })
                    .collect(),
                _ => panic!("not a string"),
            }
        };
        assert_eq!(code(&toks[0]), vec![("x".into(), None), ("y".into(), None)]);
        assert_eq!(code(&toks[1]), vec![("z".into(), Some(".2f".into()))]);
        assert_eq!(code(&toks[2]), vec![("w".into(), None)]);
        assert_eq!(code(&toks[3]), vec![("v".into(), None)]);
    }

    #[test]
    fn interpolation_with_nested_braces_and_strings() {
        let src = r#"f"{ {'a': 1}['a'] }""#;
        let (toks, errs) = lex(src);
        assert!(errs.is_empty());
        let Tok::Str(s) = &toks[0].tok else { panic!() };
        let StrPart::Code { offset, len, .. } = &s.parts[0] else {
            panic!()
        };
        assert_eq!(&src[*offset..offset + len], " {'a': 1}['a'] ");
    }

    #[test]
    fn single_quotes_and_fstring_braces_do_not_interpolate_dollars() {
        let toks = kinds(r#"'${x}' f"{{literal}}""#);
        assert_eq!(
            toks[0],
            Tok::Str(StrLit {
                parts: vec![StrPart::Text("${x}".into())]
            })
        );
        assert_eq!(
            toks[1],
            Tok::Str(StrLit {
                parts: vec![StrPart::Text("{literal}".into())]
            })
        );
    }

    #[test]
    fn sigils_macros_and_unicode() {
        let toks = kinds("$x @name println!( x ≤ y λ π");
        assert_eq!(toks[0], ident("x"));
        assert_eq!(toks[1], Tok::IVar("name".into()));
        assert_eq!(toks[2], ident("println!"));
        assert!(toks.contains(&Tok::Op("<=")));
        assert!(toks.contains(&ident("lambda")));
        assert!(toks.contains(&ident("π")));
    }

    #[test]
    fn layout_information() {
        let (toks, _) = lex("if x:\n    y\nz");
        let y = toks.iter().find(|t| t.is_word("y")).unwrap();
        assert!(y.line_start);
        assert_eq!(y.col, 4);
        assert_eq!(y.indent, 4);
        let colon = toks.iter().find(|t| t.is_op(":")).unwrap();
        assert!(!colon.space_before);
        let z = toks.iter().find(|t| t.is_word("z")).unwrap();
        assert_eq!(z.col, 0);
        assert_eq!(z.line, 2);
    }

    #[test]
    fn errors_are_reported_not_panicked() {
        for src in ["\"open", "/* never", "f\"{x\"", "a ` b", "\u{2603}"] {
            let (_, errs) = lex(src);
            assert!(!errs.is_empty(), "expected an error for {src:?}");
        }
    }

    #[test]
    fn triple_quoted_strings_span_lines() {
        let toks = kinds("\"\"\"one\ntwo\"\"\" x");
        assert_eq!(
            toks[0],
            Tok::Str(StrLit {
                parts: vec![StrPart::Text("one\ntwo".into())]
            })
        );
        assert_eq!(toks[1], ident("x"));
    }
}
