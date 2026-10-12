//! The polyglot parser.
//!
//! Chimera's parser accepts the surface syntax of many languages and lowers
//! all of it to the single core tree in [`crate::ast`]. It is a hand-written
//! recursive-descent parser with a Pratt expression core and three tricks
//! that make "no fixed structure" workable:
//!
//! 1. **Three interchangeable block styles.** After a header (`if x`,
//!    `def f()`, `while c`, …) a body may be `{ … }`, `:` + an indented block
//!    (Python's offside rule, enforced by *columns* rather than INDENT tokens
//!    so it nests inside braces), `do/then/begin … end`, a single `=> expr`,
//!    or just a newline followed by statements up to an `end`-like keyword
//!    (`end`, `fi`, `done`, `endif`, …). See `Parser::body`.
//! 2. **Contextual keywords.** The lexer never reserves words; the parser
//!    decides from context whether `end`, `to`, `times` or `select` is a
//!    keyword or an identifier.
//! 3. **Backtracking.** Where dialects genuinely collide (`(a, b) => …` vs a
//!    tuple, `int x = 1` vs a command call, `a if c else b` vs a statement
//!    modifier) the parser snapshots its position, tries one reading, and
//!    rewinds if it does not fit.
//!
//! Submodules: `stmt` (declarations and control flow), `expr`
//! (expressions), `pattern` (patterns and type annotations), `query`
//! (SQL/LINQ) and `ml` (`model`/`train`).

mod expr;
mod ml;
mod pattern;
mod query;
mod stmt;

use std::rc::Rc;

use chimera_core::{Diagnostic, Span};

use crate::ast::*;
use crate::lexer::{self, Tok, Token};
use crate::symbol::Sym;

/// Result of a parsing step; the error has already been recorded.
pub(crate) type PResult<T> = Result<T, ()>;

/// Parses one item of a block (a statement, or a class member).
pub(crate) type ItemFn<'s> = fn(&mut Parser<'s>) -> PResult<Stmt>;

/// Block terminator classes (bit flags).
pub(crate) mod terms {
    /// `end`, `endif`, `fi`, `done`, …
    pub const END: u8 = 1;
    /// `else`, `elif`, `elsif`, `elseif`, `otherwise`
    pub const ELSE: u8 = 2;
    /// `catch`, `except`, `rescue`, `finally`, `ensure`
    pub const CATCH: u8 = 4;
    /// `case`, `default` (C-style switch bodies)
    pub const CASE: u8 = 8;
    /// `when` (Ruby `case` bodies)
    pub const WHEN: u8 = 16;
    /// `until` (`repeat … until`)
    pub const UNTIL: u8 = 32;
    /// The usual set.
    pub const BASE: u8 = END | ELSE | CATCH;
}

const END_WORDS: &[&str] = &[
    "end",
    "endif",
    "endwhile",
    "endfor",
    "endforeach",
    "endfunction",
    "endfn",
    "enddef",
    "endclass",
    "endsub",
    "endswitch",
    "endmodule",
    "fi",
    "done",
    "esac",
    "wend",
];

const ELSE_WORDS: &[&str] = &["else", "elif", "elsif", "elseif", "otherwise"];
const CATCH_WORDS: &[&str] = &["catch", "except", "rescue", "finally", "ensure"];

/// Where a block header started, for layout decisions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Header {
    /// Indentation of the header's line.
    pub indent: usize,
    /// Line of the header keyword.
    pub line: usize,
}

/// A saved parser position for backtracking.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Snapshot {
    pos: usize,
    errors: usize,
}

/// Parses a source file into statements, collecting diagnostics.
pub fn parse(src: &str, file: FileId) -> (Vec<Stmt>, Vec<Diagnostic>) {
    let (toks, mut errors) = lexer::lex(src);
    let mut parser = Parser::new(src, file, toks);
    let stmts = parser.program();
    errors.extend(parser.errors);
    (stmts, errors)
}

/// Parses a single expression from `src[start..end]` (used for string
/// interpolation and the REPL's type queries).
pub(crate) fn parse_expr_range(
    src: &str,
    file: FileId,
    start: usize,
    end: usize,
) -> (Option<Expr>, Vec<Diagnostic>) {
    let (toks, mut errors) = lexer::lex_range(src, start, end);
    let mut parser = Parser::new(src, file, toks);
    parser.skip_newlines();
    let expr = parser.expr().ok();
    parser.skip_newlines();
    if expr.is_some() && !parser.at_eof() {
        let span = parser.peek().span;
        parser.error(span, "unexpected tokens after interpolated expression");
    }
    errors.extend(parser.errors);
    (expr, errors)
}

pub(crate) struct Parser<'s> {
    pub(crate) src: &'s str,
    pub(crate) file: FileId,
    toks: Vec<Token>,
    pos: usize,
    pub(crate) errors: Vec<Diagnostic>,
    /// Header context: `{` cannot start a map, struct literal or trailing block.
    pub(crate) no_brace: bool,
    /// Header context: `=` means `==` (VB/SQL/Pascal style conditions).
    pub(crate) eq_is_cmp: bool,
    /// `x -> …` is not a lambda here (Kotlin `when` arms, layer specs).
    pub(crate) no_arrow_lambda: bool,
    /// `expr as name` is an alias, not a cast (SQL select lists).
    pub(crate) no_as_cast: bool,
    /// Inside a class body: methods default to instance methods.
    pub(crate) in_class: bool,
    /// `f x` is not function application here (SQL aliases).
    pub(crate) no_juxtapose: bool,
    /// `|` is not bitwise-or here (Haskell guards).
    pub(crate) no_bitor: bool,
    /// Inside `[…]` indexing: `a:b` is a slice, not a Lua method call.
    pub(crate) in_index: bool,
    /// Inside Rust code (`fn` bodies, `for x in a..b {`): `a..b` excludes
    /// `b`. Elsewhere `..` includes it (Ruby, Kotlin, Haskell, Perl).
    pub(crate) rust_ranges: bool,
    /// Inside a Kotlin `fun`: `print` doesn't add a newline (`println` does).
    pub(crate) kotlin_fun: bool,
    /// Offsets of parsed `yield`s not yet claimed by a function.
    pub(crate) yields: std::cell::RefCell<Vec<usize>>,
    depth: usize,
}

impl<'s> Parser<'s> {
    pub(crate) fn new(src: &'s str, file: FileId, toks: Vec<Token>) -> Self {
        Parser {
            src,
            file,
            toks,
            pos: 0,
            errors: Vec::new(),
            no_brace: false,
            eq_is_cmp: false,
            no_arrow_lambda: false,
            no_as_cast: false,
            in_class: false,
            no_juxtapose: false,
            no_bitor: false,
            yields: std::cell::RefCell::new(Vec::new()),
            in_index: false,
            rust_ranges: false,
            kotlin_fun: false,
            depth: 0,
        }
    }

    // ----- token access -------------------------------------------------

    pub(crate) fn peek(&self) -> &Token {
        &self.toks[self.pos.min(self.toks.len() - 1)]
    }

    pub(crate) fn nth(&self, n: usize) -> &Token {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)]
    }

    /// The last consumed token.
    pub(crate) fn prev(&self) -> &Token {
        &self.toks[self.pos.saturating_sub(1).min(self.toks.len() - 1)]
    }

    pub(crate) fn bump(&mut self) -> Token {
        let t = self.peek().clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    pub(crate) fn toks_ref(&self) -> &[Token] {
        &self.toks
    }

    pub(crate) fn pos_ref(&self) -> usize {
        self.pos
    }

    /// Turns a `>>` token into `>` after consuming one `>` of it (closing
    /// nested generics such as `List<List<Int>>`).
    pub(crate) fn split_shift_token(&mut self) {
        if let Some(t) = self.toks.get_mut(self.pos) {
            t.tok = Tok::Op(">");
            t.span.start += 1;
            t.col += 1;
        }
    }

    /// Splits a `<-` token into `<` and `-` (for `a <- b` meaning `a < -b`).
    pub(crate) fn split_arrow_token(&mut self) {
        let Some(t) = self.toks.get(self.pos).cloned() else {
            return;
        };
        let mut lt = t.clone();
        lt.tok = Tok::Op("<");
        lt.span = Span::new(t.span.start, t.span.start + 1);
        let mut minus = t;
        minus.tok = Tok::Op("-");
        minus.span = Span::new(lt.span.end, lt.span.end + 1);
        minus.col += 1;
        minus.space_before = false;
        minus.line_start = false;
        self.toks[self.pos] = lt;
        self.toks.insert(self.pos + 1, minus);
    }

    pub(crate) fn at_eof(&self) -> bool {
        matches!(self.peek().tok, Tok::Eof)
    }

    pub(crate) fn at_newline(&self) -> bool {
        matches!(self.peek().tok, Tok::Newline)
    }

    pub(crate) fn at_op(&self, op: &str) -> bool {
        self.peek().is_op(op)
    }

    pub(crate) fn at_word(&self, w: &str) -> bool {
        self.peek().is_word(w)
    }

    pub(crate) fn at_any_word(&self, ws: &[&str]) -> bool {
        self.peek().ident().is_some_and(|w| ws.contains(&w))
    }

    pub(crate) fn nth_is_op(&self, n: usize, op: &str) -> bool {
        self.nth(n).is_op(op)
    }

    pub(crate) fn nth_is_word(&self, n: usize, w: &str) -> bool {
        self.nth(n).is_word(w)
    }

    pub(crate) fn eat_op(&mut self, op: &str) -> bool {
        if self.at_op(op) {
            self.bump();
            true
        } else {
            false
        }
    }

    pub(crate) fn eat_word(&mut self, w: &str) -> bool {
        if self.at_word(w) {
            self.bump();
            true
        } else {
            false
        }
    }

    pub(crate) fn expect_op(&mut self, op: &str, context: &str) -> PResult<Span> {
        if self.at_op(op) {
            Ok(self.bump().span)
        } else {
            let span = self.peek().span;
            let found = describe(self.peek());
            self.err(span, format!("expected `{op}` {context}, found {found}"))
        }
    }

    pub(crate) fn expect_word(&mut self, w: &str, context: &str) -> PResult<Span> {
        if self.at_word(w) {
            Ok(self.bump().span)
        } else {
            let span = self.peek().span;
            let found = describe(self.peek());
            self.err(span, format!("expected `{w}` {context}, found {found}"))
        }
    }

    pub(crate) fn expect_ident(&mut self, context: &str) -> PResult<(Sym, Span)> {
        match &self.peek().tok {
            Tok::Ident(w) => {
                let sym = Sym::new(w);
                Ok((sym, self.bump().span))
            }
            _ => {
                let span = self.peek().span;
                let found = describe(self.peek());
                self.err(span, format!("expected a name {context}, found {found}"))
            }
        }
    }

    pub(crate) fn skip_newlines(&mut self) {
        while self.at_newline() {
            self.bump();
        }
    }

    pub(crate) fn skip_separators(&mut self) {
        while self.at_newline() || self.at_op(";") {
            self.bump();
        }
    }

    /// Whether the next token is on the same line as the previous one.
    pub(crate) fn same_line(&self) -> bool {
        !self.at_newline() && !self.at_eof() && self.peek().line == self.prev().line
    }

    /// The token after any newlines, without consuming anything.
    pub(crate) fn peek_past_newlines(&self) -> &Token {
        let mut i = self.pos;
        while i < self.toks.len() - 1 && matches!(self.toks[i].tok, Tok::Newline) {
            i += 1;
        }
        &self.toks[i]
    }

    pub(crate) fn span_from(&self, start: Span) -> Span {
        start.to(self.prev().span)
    }

    pub(crate) fn header(&self) -> Header {
        let t = self.peek();
        Header {
            indent: t.indent,
            line: t.line,
        }
    }

    // ----- diagnostics and backtracking ----------------------------------

    pub(crate) fn error(&mut self, span: Span, msg: impl Into<String>) {
        self.errors.push(
            Diagnostic::error(msg)
                .with_code("E0002")
                .with_label(span, ""),
        );
    }

    pub(crate) fn err<T>(&mut self, span: Span, msg: impl Into<String>) -> PResult<T> {
        self.error(span, msg);
        Err(())
    }

    pub(crate) fn snapshot(&self) -> Snapshot {
        Snapshot {
            pos: self.pos,
            errors: self.errors.len(),
        }
    }

    pub(crate) fn restore(&mut self, snap: Snapshot) {
        self.pos = snap.pos;
        self.errors.truncate(snap.errors);
    }

    /// Runs `f`; on failure rewinds and returns `None` without diagnostics.
    pub(crate) fn attempt<T>(&mut self, f: impl FnOnce(&mut Self) -> PResult<T>) -> Option<T> {
        let snap = self.snapshot();
        match f(self) {
            Ok(v) => Some(v),
            Err(()) => {
                self.restore(snap);
                None
            }
        }
    }

    /// Runs `f` with header flags set (no brace literals, `=` compares).
    pub(crate) fn in_header<T>(&mut self, f: impl FnOnce(&mut Self) -> PResult<T>) -> PResult<T> {
        let saved = (self.no_brace, self.eq_is_cmp);
        self.no_brace = true;
        self.eq_is_cmp = true;
        let r = f(self);
        (self.no_brace, self.eq_is_cmp) = saved;
        r
    }

    /// Runs `f` with header flags cleared (inside brackets).
    pub(crate) fn nested<T>(&mut self, f: impl FnOnce(&mut Self) -> PResult<T>) -> PResult<T> {
        let saved = (
            self.no_brace,
            self.eq_is_cmp,
            self.no_arrow_lambda,
            self.no_as_cast,
            self.no_juxtapose,
        );
        self.no_brace = false;
        self.eq_is_cmp = false;
        self.no_arrow_lambda = false;
        self.no_as_cast = false;
        self.no_juxtapose = false;
        let r = f(self);
        (
            self.no_brace,
            self.eq_is_cmp,
            self.no_arrow_lambda,
            self.no_as_cast,
            self.no_juxtapose,
        ) = saved;
        r
    }

    /// Runs `f` with juxtaposition application disabled.
    pub(crate) fn no_jux<T>(&mut self, f: impl FnOnce(&mut Self) -> PResult<T>) -> PResult<T> {
        let saved = self.no_juxtapose;
        self.no_juxtapose = true;
        let r = f(self);
        self.no_juxtapose = saved;
        r
    }

    /// Runs `f` with `|` not treated as an operator.
    pub(crate) fn no_bar<T>(&mut self, f: impl FnOnce(&mut Self) -> PResult<T>) -> PResult<T> {
        let saved = self.no_bitor;
        self.no_bitor = true;
        let r = f(self);
        self.no_bitor = saved;
        r
    }

    pub(crate) fn enter(&mut self) -> PResult<()> {
        self.depth += 1;
        if self.depth > 400 {
            let span = self.peek().span;
            return self.err(span, "program nests too deeply to parse");
        }
        Ok(())
    }

    pub(crate) fn leave(&mut self) {
        self.depth -= 1;
    }

    /// Skips to the end of the current statement after an error.
    fn recover(&mut self) {
        let mut depth = 0usize;
        let start = self.pos;
        loop {
            match &self.peek().tok {
                Tok::Eof => break,
                Tok::Newline if depth == 0 => break,
                Tok::Op(";") if depth == 0 => break,
                Tok::Op("(" | "[" | "{") => depth += 1,
                Tok::Op(")" | "]" | "}") => {
                    if depth == 0 {
                        if self.pos == start {
                            self.bump();
                        }
                        break;
                    }
                    depth -= 1;
                }
                _ => {}
            }
            self.bump();
        }
    }

    // ----- terminators -----------------------------------------------------

    /// Whether the next token ends a block of the given terminator classes.
    pub(crate) fn at_terminator(&self, set: u8) -> bool {
        let t = self.peek();
        if matches!(t.tok, Tok::Eof) || t.is_op("}") || t.is_op(")") || t.is_op("]") {
            return true;
        }
        let Some(w) = t.ident() else {
            return false;
        };
        // A terminator word used as a variable (`done = true`) is not one.
        let next = self.nth(1);
        let used_as_value = ["=", ".", "(", "[", ":=", "+=", "-=", "?."]
            .iter()
            .any(|op| next.is_op(op) && (*op != "(" || !next.space_before));
        if used_as_value {
            return false;
        }
        (set & terms::END != 0 && END_WORDS.contains(&w))
            || (set & terms::ELSE != 0 && ELSE_WORDS.contains(&w))
            || (set & terms::CATCH != 0 && CATCH_WORDS.contains(&w))
            || (set & terms::CASE != 0 && (w == "case" || w == "default"))
            || (set & terms::WHEN != 0 && (w == "when" || w == "in"))
            || (set & terms::UNTIL != 0 && w == "until")
    }

    pub(crate) fn at_end_word(&self) -> bool {
        self.at_terminator(terms::END) && self.peek().ident().is_some()
    }

    /// Consumes an optional `end`-like keyword closing a construct that
    /// started at `h` (`end`, `end if`, `endwhile`, `fi`, …).
    pub(crate) fn maybe_end(&mut self, h: Header) {
        let snap = self.snapshot();
        let prev_line = self.prev().line;
        self.skip_newlines();
        let t = self.peek();
        let attachable = t.line == prev_line || t.col >= h.indent;
        if self.at_end_word() && attachable {
            let line = self.peek().line;
            self.bump();
            // `end if`, `end function`, `end while` (VB/Fortran/Ada style).
            if self.peek().line == line
                && self.at_any_word(&[
                    "if", "while", "for", "function", "sub", "def", "class", "select", "loop",
                    "do", "case", "try", "module", "struct", "match", "switch", "fn",
                ])
            {
                self.bump();
            }
            self.eat_op(";");
        } else {
            self.restore(snap);
        }
    }

    // ----- programs and blocks ---------------------------------------------

    fn program(&mut self) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        loop {
            self.skip_separators();
            if self.at_eof() {
                break;
            }
            // `case`, `when` and `until` start statements at the top level
            // (Ruby `case x`, Kotlin `when`, Ruby `until`), so only `end`/
            // `else`-like words are stray here.
            if self.at_terminator(terms::BASE) {
                let t = self.bump();
                self.error(t.span, format!("unexpected {}", describe(&t)));
                continue;
            }
            match self.statement() {
                Ok(s) => stmts.push(s),
                Err(()) => self.recover(),
            }
        }
        merge_clauses(&mut stmts);
        stmts
    }

    fn finish_block(&self, mut stmts: Vec<Stmt>) -> Block {
        merge_clauses(&mut stmts);
        Block::new(stmts)
    }

    /// `{ stmts }`
    pub(crate) fn brace_block(&mut self) -> PResult<Block> {
        self.brace_block_with(Self::statement)
    }

    pub(crate) fn brace_block_with(&mut self, item: ItemFn<'s>) -> PResult<Block> {
        let open = self.expect_op("{", "to open a block")?;
        let stmts = self.nested(|p| {
            let mut stmts = Vec::new();
            loop {
                p.skip_separators();
                if p.eat_op("}") {
                    break;
                }
                if p.at_eof() {
                    return p.err(open, "this `{` is never closed");
                }
                if p.at_op(")") || p.at_op("]") {
                    let t = p.bump();
                    p.error(
                        t.span,
                        format!("unexpected {} inside a block", describe(&t)),
                    );
                    continue;
                }
                match item(p) {
                    Ok(s) => stmts.push(s),
                    Err(()) => p.recover(),
                }
            }
            Ok(stmts)
        })?;
        Ok(self.finish_block(stmts))
    }

    /// Statements on the current line (`if x: a; b`, `if x then a else b end`).
    fn inline_block_with(&mut self, set: u8, item: ItemFn<'s>) -> PResult<Block> {
        let mut stmts = Vec::new();
        loop {
            if self.at_newline() || self.at_terminator(set) {
                break;
            }
            stmts.push(item(self)?);
            if self.at_op(";") {
                self.bump();
                continue;
            }
            if !self.same_line() {
                break;
            }
        }
        Ok(self.finish_block(stmts))
    }

    /// Statements whose first tokens sit at column `col` or deeper (offside rule).
    fn offside_block_with(&mut self, col: usize, set: u8, item: ItemFn<'s>) -> PResult<Block> {
        let mut stmts = Vec::new();
        loop {
            self.skip_separators();
            let t = self.peek();
            if self.at_terminator(set) || (t.line_start && t.col < col) {
                break;
            }
            match item(self) {
                Ok(s) => stmts.push(s),
                Err(()) => self.recover(),
            }
        }
        Ok(self.finish_block(stmts))
    }

    /// Statements up to a terminator keyword, regardless of indentation.
    fn terminated_block_with(&mut self, set: u8, item: ItemFn<'s>) -> PResult<Block> {
        let mut stmts = Vec::new();
        let start = self.peek().span;
        loop {
            self.skip_separators();
            if self.at_eof() {
                return self.err(
                    start,
                    "this block is never closed: indent its body or end it with `end`",
                );
            }
            if self.at_terminator(set) {
                break;
            }
            match item(self) {
                Ok(s) => stmts.push(s),
                Err(()) => self.recover(),
            }
        }
        Ok(self.finish_block(stmts))
    }

    /// The body after a newline: braces, indented (offside) or `end`-terminated.
    fn newline_block_with(&mut self, h: Header, set: u8, item: ItemFn<'s>) -> PResult<Block> {
        self.skip_separators();
        if self.at_op("{") {
            return self.brace_block_with(item);
        }
        if self.at_terminator(set) {
            return Ok(Block::default());
        }
        let t = self.peek();
        if t.col > h.indent {
            let col = t.col;
            self.offside_block_with(col, set, item)
        } else {
            self.terminated_block_with(set, item)
        }
    }

    /// Parses the body of a construct whose header started at `h`.
    ///
    /// Accepts `{ … }`, `: stmt`, `:` + indented block, `then`/`do`/`begin`
    /// bodies, `=> expr` / `= expr` (when `expr_body`), a newline followed by
    /// an indented or `end`-terminated block, or a single statement.
    pub(crate) fn body(&mut self, h: Header, expr_body: bool, set: u8) -> PResult<Block> {
        self.body_with(h, expr_body, set, Self::statement)
    }

    pub(crate) fn body_with(
        &mut self,
        h: Header,
        expr_body: bool,
        set: u8,
        item: ItemFn<'s>,
    ) -> PResult<Block> {
        if self.at_op("{") {
            return self.brace_block_with(item);
        }
        if self.at_op(":") {
            self.bump();
            if self.at_newline() {
                return self.newline_block_with(h, set, item);
            }
            return self.inline_block_with(set, item);
        }
        if expr_body && (self.at_op("=>") || self.at_op("=")) {
            self.bump();
            self.skip_newlines();
            if self.at_op("{") && self.prev().is_op("=>") {
                return self.brace_block_with(item);
            }
            let e = self.nested(|p| p.expr())?;
            return Ok(Block::of_expr(e));
        }
        if self.at_any_word(&["then", "do", "begin"]) && !self.nth_is_op(1, "=") {
            self.bump();
            if self.at_newline() {
                return self.newline_block_with(h, set, item);
            }
            if self.at_op("{") {
                return self.brace_block_with(item);
            }
            return self.inline_block_with(set, item);
        }
        if self.at_newline() {
            return self.newline_block_with(h, set, item);
        }
        if self.at_eof() || self.at_op(";") {
            return Ok(Block::default());
        }
        // C-style single statement body: `if (x) y();`
        let s = item(self)?;
        Ok(self.finish_block(vec![s]))
    }

    // ----- statements --------------------------------------------------------

    pub(crate) fn statement(&mut self) -> PResult<Stmt> {
        self.enter()?;
        let r = self.statement_inner();
        self.leave();
        r
    }

    /// Ends a simple statement: `;`, newline, or something that closes the
    /// enclosing block. Anything else on the same line is reported.
    pub(crate) fn end_stmt(&mut self) -> PResult<()> {
        if self.eat_op(";") || self.at_newline() || self.at_eof() {
            return Ok(());
        }
        if self.at_terminator(terms::BASE | terms::CASE | terms::WHEN | terms::UNTIL) {
            return Ok(());
        }
        if !self.same_line() {
            return Ok(());
        }
        let span = self.peek().span;
        let found = describe(self.peek());
        self.err(
            span,
            format!("expected the end of the statement, found {found}"),
        )
    }

    pub(crate) fn make_stmt(&self, kind: StmtKind, start: Span) -> Stmt {
        Stmt {
            kind,
            span: self.span_from(start),
        }
    }

    pub(crate) fn expr_stmt_from(&mut self, e: Expr) -> PResult<Stmt> {
        let start = e.span;
        let e = self.statement_modifiers(e)?;
        self.end_stmt()?;
        Ok(self.make_stmt(StmtKind::Expr(e), start))
    }

    /// Ruby/Perl statement modifiers: `x if c`, `x unless c`, `x while c`.
    pub(crate) fn statement_modifiers(&mut self, mut e: Expr) -> PResult<Expr> {
        loop {
            if !self.same_line() {
                return Ok(e);
            }
            let start = e.span;
            if self.eat_word("if") {
                let cond = self.in_header(|p| p.expr())?;
                let span = self.span_from(start);
                e = Expr::new(
                    ExprKind::If {
                        cond: Box::new(cond),
                        then: Block::of_expr(e),
                        otherwise: None,
                    },
                    span,
                );
            } else if self.eat_word("unless") {
                let cond = self.in_header(|p| p.expr())?;
                let span = self.span_from(start);
                let cond = Expr::new(ExprKind::Unary(UnOp::Not, Box::new(cond)), span);
                e = Expr::new(
                    ExprKind::If {
                        cond: Box::new(cond),
                        then: Block::of_expr(e),
                        otherwise: None,
                    },
                    span,
                );
            } else if self.at_word("while") || self.at_word("until") {
                let negate = self.bump().is_word("until");
                let cond = self.in_header(|p| p.expr())?;
                let span = self.span_from(start);
                // Ruby: `begin … end while c` tests after the body.
                let post = matches!(e.kind, ExprKind::Block(_));
                let body = Block::of_expr(e);
                let stmt = Stmt {
                    kind: StmtKind::While {
                        cond,
                        body,
                        negate,
                        post,
                    },
                    span,
                };
                e = Expr::new(
                    ExprKind::Block(Block {
                        stmts: vec![stmt],
                        scoped: false,
                    }),
                    span,
                );
            } else {
                return Ok(e);
            }
        }
    }

    /// Parses a header expression (an `if`/`while` condition, …).
    pub(crate) fn header_expr(&mut self) -> PResult<Expr> {
        self.in_header(|p| p.expr())
    }

    /// Parses a full expression and wraps it as an `If` taking `then`.
    pub(crate) fn mk_if(
        &self,
        cond: Expr,
        then: Block,
        otherwise: Option<Block>,
        span: Span,
    ) -> Expr {
        Expr::new(
            ExprKind::If {
                cond: Box::new(cond),
                then,
                otherwise,
            },
            span,
        )
    }

    pub(crate) fn sym_expr(&self, name: &str, span: Span) -> Expr {
        Expr::new(ExprKind::Name(Sym::new(name)), span)
    }

    pub(crate) fn call_expr(&self, name: &str, args: Vec<Expr>, span: Span) -> Expr {
        Expr::new(
            ExprKind::Call {
                callee: Box::new(self.sym_expr(name, span)),
                args: args.into_iter().map(Arg::Pos).collect(),
            },
            span,
        )
    }

    /// Takes the unclaimed `yield` offsets inside `span`.
    pub(crate) fn claim_yields(&self, span: Span) -> Vec<usize> {
        let mut all = self.yields.borrow_mut();
        let (mut mine, rest): (Vec<usize>, Vec<usize>) =
            all.iter().partition(|&&o| o >= span.start && o < span.end);
        *all = rest;
        mine.sort_unstable();
        mine.dedup();
        mine
    }

    /// A function definition. Named functions claim the yields in their
    /// body; anonymous blocks don't (a `yield` inside a Ruby block belongs
    /// to the enclosing method).
    pub(crate) fn new_fn(
        &self,
        name: Option<Sym>,
        params: Vec<Param>,
        body: Block,
        span: Span,
    ) -> FnDef {
        let yields = if name.is_some() {
            self.claim_yields(span)
        } else {
            Vec::new()
        };
        FnDef {
            yields,
            name,
            params,
            body,
            span,
            file: self.file,
            ret: None,
            receiver: None,
            is_static: false,
            implicit_params: false,
            decorators: Vec::new(),
            bare: false,
        }
    }

    pub(crate) fn lambda(&self, params: Vec<Param>, body: Block, span: Span) -> Expr {
        Expr::new(
            ExprKind::Lambda(Rc::new(self.new_fn(None, params, body, span))),
            span,
        )
    }
}

/// A plain positional parameter.
pub(crate) fn simple_param(name: Sym) -> Param {
    Param {
        pattern: Pattern::Bind(name),
        default: None,
        ty: None,
        kind: ParamKind::Normal,
        label: None,
    }
}

/// Human-readable description of a token for error messages.
pub(crate) fn describe(t: &Token) -> String {
    match &t.tok {
        Tok::Int(v) => format!("number `{v}`"),
        Tok::BigInt(v) => format!("number `{v}`"),
        Tok::Float(v) => format!("number `{v}`"),
        Tok::Str(_) => "a string".to_owned(),
        Tok::Ident(w) => format!("`{w}`"),
        Tok::IVar(w) => format!("`@{w}`"),
        Tok::Op(o) => format!("`{o}`"),
        Tok::Newline => "a line break".to_owned(),
        Tok::Eof => "the end of the file".to_owned(),
    }
}

/// Merges consecutive Haskell-style clauses of the same function
/// (`fib 0 = 0`, `fib 1 = 1`, `fib n = …`) into one function that matches
/// its arguments against each clause in turn.
fn merge_clauses(stmts: &mut Vec<Stmt>) {
    let mut out: Vec<Stmt> = Vec::with_capacity(stmts.len());
    for stmt in stmts.drain(..) {
        let merged = match (&stmt.kind, out.last_mut()) {
            (StmtKind::Fn(def), Some(prev)) => match &mut prev.kind {
                StmtKind::Fn(prev_def)
                    if prev_def.name.is_some()
                        && prev_def.name == def.name
                        && prev_def.params.len() == def.params.len()
                        && prev_def.params.iter().all(|p| p.kind == ParamKind::Normal)
                        && def.params.iter().all(|p| p.kind == ParamKind::Normal)
                        && (has_clause_patterns(prev_def)
                            || has_clause_patterns(def)
                            || is_merged_clause(prev_def)) =>
                {
                    let combined = combine_clauses(prev_def, def);
                    *prev_def = Rc::new(combined);
                    prev.span = prev.span.to(stmt.span);
                    true
                }
                _ => false,
            },
            _ => false,
        };
        if !merged {
            out.push(stmt);
        }
    }
    *stmts = out;
}

const CLAUSE_ARG: &str = "__clause_arg";

fn has_clause_patterns(def: &FnDef) -> bool {
    def.params
        .iter()
        .any(|p| !matches!(p.pattern, Pattern::Bind(_) | Pattern::Wildcard))
}

fn is_merged_clause(def: &FnDef) -> bool {
    def.params
        .first()
        .and_then(Param::name)
        .is_some_and(|n| n.as_str().starts_with(CLAUSE_ARG))
}

fn clause_arm(def: &FnDef) -> MatchArm {
    MatchArm {
        pattern: Pattern::Tuple(def.params.iter().map(|p| p.pattern.clone()).collect()),
        guard: None,
        body: def.body.clone(),
    }
}

fn combine_clauses(first: &FnDef, next: &FnDef) -> FnDef {
    // If `first` is already a merged clause function, append an arm.
    if let Some(Stmt {
        kind:
            StmtKind::Expr(Expr {
                kind: ExprKind::Match { arms, .. },
                ..
            }),
        ..
    }) = first.body.stmts.first()
        && first
            .params
            .first()
            .and_then(Param::name)
            .is_some_and(|n| n.as_str().starts_with(CLAUSE_ARG))
    {
        let mut def = first.clone();
        let mut arms = arms.clone();
        arms.push(clause_arm(next));
        if let Some(Stmt {
            kind:
                StmtKind::Expr(Expr {
                    kind: ExprKind::Match { arms: slot, .. },
                    ..
                }),
            ..
        }) = def.body.stmts.first_mut()
        {
            *slot = arms;
        }
        def.span = first.span.to(next.span);
        return def;
    }
    let span = first.span.to(next.span);
    let names: Vec<Sym> = (0..first.params.len())
        .map(|i| Sym::new(&format!("{CLAUSE_ARG}{i}")))
        .collect();
    let subject = Expr::new(
        ExprKind::Tuple(
            names
                .iter()
                .map(|n| Expr::new(ExprKind::Name(*n), span))
                .collect(),
        ),
        span,
    );
    let matcher = Expr::new(
        ExprKind::Match {
            subject: Some(Box::new(subject)),
            arms: vec![clause_arm(first), clause_arm(next)],
        },
        span,
    );
    FnDef {
        name: first.name,
        params: names.into_iter().map(simple_param).collect(),
        body: Block::of_expr(matcher),
        span,
        file: first.file,
        ret: first.ret.clone(),
        receiver: None,
        is_static: first.is_static,
        implicit_params: false,
        decorators: Vec::new(),
        bare: false,
        yields: first.yields.iter().chain(&next.yields).copied().collect(),
    }
}
