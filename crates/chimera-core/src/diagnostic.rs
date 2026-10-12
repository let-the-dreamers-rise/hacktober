//! Source spans and rustc-style diagnostics.
//!
//! Both the Chimera language and the SQL front end report errors through
//! [`Diagnostic`], so every error in the system looks the same:
//!
//! ```text
//! error[E0308]: mismatched types
//!  --> demo.chi:2:14
//!   |
//! 2 | let x: Int = "hello"
//!   |              ^^^^^^^ expected Int, found String
//!   |
//!   = help: convert with `parse_int`
//! ```

use std::fmt;
use std::ops::Range;

/// A half-open byte range `[start, end)` into a source text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Span {
    /// First byte of the span.
    pub start: usize,
    /// One past the last byte of the span.
    pub end: usize,
}

impl Span {
    /// Creates a span; `end` is clamped so it is never before `start`.
    pub const fn new(start: usize, end: usize) -> Self {
        let end = if end < start { start } else { end };
        Span { start, end }
    }

    /// An empty span at `offset`.
    pub const fn point(offset: usize) -> Self {
        Span {
            start: offset,
            end: offset,
        }
    }

    /// The smallest span covering both `self` and `other`.
    pub fn to(self, other: Span) -> Span {
        Span::new(self.start.min(other.start), self.end.max(other.end))
    }

    /// Length in bytes.
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the span covers no bytes.
    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// The span as a `Range`, for slicing.
    pub const fn range(&self) -> Range<usize> {
        self.start..self.end
    }
}

impl From<Range<usize>> for Span {
    fn from(r: Range<usize>) -> Self {
        Span::new(r.start, r.end)
    }
}

/// A named source text with a precomputed line index.
#[derive(Clone, Debug)]
pub struct SourceFile {
    name: String,
    text: String,
    line_starts: Vec<usize>,
}

impl SourceFile {
    /// Indexes `text` for line/column lookups.
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        let line_starts = std::iter::once(0)
            .chain(text.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        SourceFile {
            name: name.into(),
            text,
            line_starts,
        }
    }

    /// The file name shown in diagnostics.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The full source text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Number of lines (a trailing newline does not start an extra line).
    pub fn line_count(&self) -> usize {
        if self.text.ends_with('\n') {
            self.line_starts.len() - 1
        } else {
            self.line_starts.len()
        }
        .max(1)
    }

    /// Zero-based line containing byte `offset` (clamped to the text).
    pub fn line_index(&self, offset: usize) -> usize {
        let offset = offset.min(self.text.len());
        let idx = self.line_starts.partition_point(|&start| start <= offset) - 1;
        idx.min(self.line_count() - 1)
    }

    /// One-based `(line, column)` of byte `offset`; columns count characters.
    pub fn line_col(&self, offset: usize) -> (usize, usize) {
        let offset = self.floor_char_boundary(offset.min(self.text.len()));
        let line = self.line_index(offset);
        let start = self.line_starts[line];
        let col = self.text[start..offset].chars().count();
        (line + 1, col + 1)
    }

    /// Text of zero-based line `idx`, without its line terminator.
    pub fn line(&self, idx: usize) -> &str {
        let Some(&start) = self.line_starts.get(idx) else {
            return "";
        };
        let end = self
            .line_starts
            .get(idx + 1)
            .map_or(self.text.len(), |&next| next - 1);
        self.text[start..end].trim_end_matches('\r')
    }

    /// Byte offset where zero-based line `idx` starts.
    pub fn line_start(&self, idx: usize) -> usize {
        self.line_starts
            .get(idx)
            .copied()
            .unwrap_or(self.text.len())
    }

    fn floor_char_boundary(&self, mut offset: usize) -> usize {
        while !self.text.is_char_boundary(offset) {
            offset -= 1;
        }
        offset
    }
}

/// How serious a diagnostic is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Severity {
    /// Compilation or execution cannot continue.
    Error,
    /// Suspicious but allowed.
    Warning,
    /// Extra information attached to another diagnostic.
    Note,
    /// A suggestion.
    Help,
}

impl Severity {
    fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
            Severity::Help => "help",
        }
    }

    fn color(self) -> &'static str {
        match self {
            Severity::Error => "\x1b[1;31m",
            Severity::Warning => "\x1b[1;33m",
            Severity::Note => "\x1b[1;36m",
            Severity::Help => "\x1b[1;32m",
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A span annotated with a message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    /// The annotated source range.
    pub span: Span,
    /// Text printed next to the underline (may be empty).
    pub message: String,
    /// Primary labels are underlined with `^`, secondary ones with `-`.
    pub primary: bool,
}

/// An error, warning or note about some source text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    /// Severity of the diagnostic.
    pub severity: Severity,
    /// Optional stable error code such as `E0308`.
    pub code: Option<String>,
    /// The headline.
    pub message: String,
    /// Annotated spans, in the order they were added.
    pub labels: Vec<Label>,
    /// `= note:` lines.
    pub notes: Vec<String>,
    /// `= help:` lines.
    pub help: Vec<String>,
}

impl Diagnostic {
    /// Creates a diagnostic with no labels.
    pub fn new(severity: Severity, message: impl Into<String>) -> Self {
        Diagnostic {
            severity,
            code: None,
            message: message.into(),
            labels: Vec::new(),
            notes: Vec::new(),
            help: Vec::new(),
        }
    }

    /// Shorthand for an error.
    pub fn error(message: impl Into<String>) -> Self {
        Diagnostic::new(Severity::Error, message)
    }

    /// Shorthand for a warning.
    pub fn warning(message: impl Into<String>) -> Self {
        Diagnostic::new(Severity::Warning, message)
    }

    /// Attaches an error code.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Adds a primary (`^^^`) label.
    pub fn with_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(Label {
            span,
            message: message.into(),
            primary: true,
        });
        self
    }

    /// Adds a secondary (`---`) label.
    pub fn with_secondary(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push(Label {
            span,
            message: message.into(),
            primary: false,
        });
        self
    }

    /// Adds a `= note:` line.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Adds a `= help:` line.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help.push(help.into());
        self
    }

    /// The first primary label's span, falling back to the first label.
    pub fn primary_span(&self) -> Option<Span> {
        self.labels
            .iter()
            .find(|l| l.primary)
            .or_else(|| self.labels.first())
            .map(|l| l.span)
    }

    /// Renders the diagnostic against its source, optionally with ANSI color.
    pub fn render(&self, source: &SourceFile, color: bool) -> String {
        Renderer { source, color }.render(self)
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.severity)?;
        if let Some(code) = &self.code {
            write!(f, "[{code}]")?;
        }
        write!(f, ": {}", self.message)
    }
}

impl std::error::Error for Diagnostic {}

struct Renderer<'a> {
    source: &'a SourceFile,
    color: bool,
}

const TAB_WIDTH: usize = 4;
const BLUE: &str = "\x1b[1;34m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

impl Renderer<'_> {
    fn paint(&self, style: &str, text: &str) -> String {
        if self.color && !text.is_empty() {
            format!("{style}{text}{RESET}")
        } else {
            text.to_owned()
        }
    }

    /// Display column (0-based) of `offset` within its line, expanding tabs.
    fn display_col(&self, line: usize, offset: usize) -> usize {
        let start = self.source.line_start(line);
        let text = self.source.line(line);
        let upto = offset.saturating_sub(start).min(text.len());
        let mut col = 0;
        for (i, ch) in text.char_indices() {
            if i >= upto {
                break;
            }
            col += if ch == '\t' { TAB_WIDTH } else { 1 };
        }
        col
    }

    fn render(&self, diag: &Diagnostic) -> String {
        let mut out = String::new();
        let sev_color = diag.severity.color();
        let mut header = diag.severity.as_str().to_owned();
        if let Some(code) = &diag.code {
            header.push_str(&format!("[{code}]"));
        }
        out.push_str(&self.paint(sev_color, &header));
        out.push_str(&self.paint(BOLD, &format!(": {}", diag.message)));
        out.push('\n');

        // Group labels by the line they start on.
        let mut lines: Vec<(usize, Vec<&Label>)> = Vec::new();
        for label in &diag.labels {
            let line = self.source.line_index(label.span.start);
            match lines.iter_mut().find(|(l, _)| *l == line) {
                Some((_, group)) => group.push(label),
                None => lines.push((line, vec![label])),
            }
        }
        lines.sort_by_key(|(line, _)| *line);

        let max_line = lines.last().map_or(1, |(l, _)| l + 1);
        let width = max_line.to_string().len();
        let pad = " ".repeat(width);
        let bar = self.paint(BLUE, "|");

        if let Some(span) = diag.primary_span() {
            let (line, col) = self.source.line_col(span.start);
            out.push_str(&format!(
                "{pad}{} {}:{line}:{col}\n",
                self.paint(BLUE, "-->"),
                self.source.name()
            ));
        }

        if !lines.is_empty() {
            out.push_str(&format!("{pad} {bar}\n"));
            let mut prev: Option<usize> = None;
            for (line, labels) in &lines {
                if let Some(p) = prev
                    && line - p > 1
                {
                    out.push_str(&format!("{}\n", self.paint(BLUE, "...")));
                }
                prev = Some(*line);
                let text = self
                    .source
                    .line(*line)
                    .replace('\t', &" ".repeat(TAB_WIDTH));
                let number = format!("{:>width$}", line + 1);
                out.push_str(&format!("{} {bar} {text}\n", self.paint(BLUE, &number)));

                for label in labels {
                    let start_col = self.display_col(*line, label.span.start);
                    let line_end = self.source.line_start(*line) + self.source.line(*line).len();
                    let end_col = self.display_col(*line, label.span.end.min(line_end));
                    let len = end_col.saturating_sub(start_col).max(1);
                    let (mark, style) = if label.primary {
                        ('^', sev_color)
                    } else {
                        ('-', BLUE)
                    };
                    let mut underline = mark.to_string().repeat(len);
                    if label.span.end > line_end {
                        underline.push_str("...");
                    }
                    if !label.message.is_empty() {
                        underline.push(' ');
                        underline.push_str(&label.message);
                    }
                    out.push_str(&format!(
                        "{pad} {bar} {}{}\n",
                        " ".repeat(start_col),
                        self.paint(style, &underline)
                    ));
                }
            }
        }

        if !diag.notes.is_empty() || !diag.help.is_empty() {
            if !lines.is_empty() {
                out.push_str(&format!("{pad} {bar}\n"));
            }
            for note in &diag.notes {
                out.push_str(&format!("{pad} = {}: {note}\n", self.paint(BOLD, "note")));
            }
            for help in &diag.help {
                out.push_str(&format!("{pad} = {}: {help}\n", self.paint(BOLD, "help")));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_helpers() {
        let a = Span::new(3, 7);
        let b = Span::new(5, 12);
        assert_eq!(a.to(b), Span::new(3, 12));
        assert_eq!(a.len(), 4);
        assert!(Span::point(4).is_empty());
        assert_eq!(Span::new(9, 2), Span::new(9, 9));
        assert_eq!(Span::from(1..4).range(), 1..4);
    }

    #[test]
    fn line_and_column_lookup() {
        let src = SourceFile::new("t", "ab\ncdé\n\nxyz");
        assert_eq!(src.line_count(), 4);
        assert_eq!(src.line_col(0), (1, 1));
        assert_eq!(src.line_col(2), (1, 3));
        assert_eq!(src.line_col(3), (2, 1));
        // 'é' is two bytes; the offset after it is column 4.
        assert_eq!(src.line_col(7), (2, 4));
        assert_eq!(src.line_col(9), (4, 1));
        assert_eq!(src.line_col(1_000), (4, 4));
        assert_eq!(src.line(1), "cdé");
        assert_eq!(src.line(2), "");
        assert_eq!(src.line(99), "");
    }

    #[test]
    fn trailing_newline_and_crlf() {
        let src = SourceFile::new("t", "one\r\ntwo\n");
        assert_eq!(src.line_count(), 2);
        assert_eq!(src.line(0), "one");
        assert_eq!(src.line_index(src.text().len()), 1);
    }

    #[test]
    fn renders_like_rustc() {
        let src = SourceFile::new("demo.chi", "let x = 1\nlet y: Int = \"hello\"\n");
        let start = src.text().find('"').unwrap();
        let diag = Diagnostic::error("mismatched types")
            .with_code("E0308")
            .with_label(Span::new(start, start + 7), "expected Int, found String")
            .with_help("convert with `parse_int`");
        let rendered = diag.render(&src, false);
        let expected = "\
error[E0308]: mismatched types
 --> demo.chi:2:14
  |
2 | let y: Int = \"hello\"
  |              ^^^^^^^ expected Int, found String
  |
  = help: convert with `parse_int`
";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn renders_secondary_labels_gaps_and_multiline_spans() {
        let text = "fn f() {\n  a\n  b\n  c\n  d\n}\n";
        let src = SourceFile::new("m.chi", text);
        let open = text.find('{').unwrap();
        let close = text.rfind('}').unwrap();
        let diag = Diagnostic::error("unclosed thing")
            .with_secondary(Span::new(open, close + 1), "opened here")
            .with_label(Span::point(close), "");
        let rendered = diag.render(&src, false);
        assert!(rendered.contains("1 | fn f() {"));
        assert!(rendered.contains("-..."), "{rendered}");
        assert!(rendered.contains("..."));
        assert!(rendered.contains("6 | }"));
        assert!(rendered.contains("--> m.chi:6:1"));
    }

    #[test]
    fn color_output_contains_escape_codes() {
        let src = SourceFile::new("c", "boom");
        let diag = Diagnostic::warning("loud").with_label(Span::new(0, 4), "here");
        let colored = diag.render(&src, true);
        assert!(colored.contains("\x1b[1;33m"));
        assert!(!diag.render(&src, false).contains('\x1b'));
    }

    #[test]
    fn diagnostic_without_labels() {
        let src = SourceFile::new("x", "");
        let diag = Diagnostic::error("no input").with_note("nothing to do");
        assert_eq!(
            diag.render(&src, false),
            "error: no input\n  = note: nothing to do\n"
        );
        assert_eq!(diag.to_string(), "error: no input");
    }

    #[test]
    fn tabs_are_expanded_consistently() {
        let src = SourceFile::new("t", "\tbad");
        let diag = Diagnostic::error("e").with_label(Span::new(1, 4), "x");
        let rendered = diag.render(&src, false);
        assert!(rendered.contains("1 |     bad"));
        assert!(rendered.contains("  |     ^^^ x"));
    }
}
