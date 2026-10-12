//! Box-drawn text tables for the SQL shell and CLI reports.
//!
//! ```
//! use chimera_core::table::{Align, Table};
//!
//! let mut t = Table::new(["id", "name"]);
//! t.set_align(0, Align::Right);
//! t.add_row(["1", "ada"]);
//! t.add_row(["42", "grace"]);
//! assert_eq!(t.render(), "\
//! ┌────┬───────┐
//! │ id │ name  │
//! ├────┼───────┤
//! │  1 │ ada   │
//! │ 42 │ grace │
//! └────┴───────┘
//! ");
//! ```

/// Horizontal alignment of a column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Align {
    /// Pad on the right.
    #[default]
    Left,
    /// Pad on the left (numbers).
    Right,
}

/// Which characters to draw borders with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Style {
    /// Unicode box-drawing characters.
    #[default]
    Unicode,
    /// Plain `+`, `-` and `|`.
    Ascii,
}

/// A table of strings rendered with aligned columns.
#[derive(Clone, Debug, Default)]
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    align: Vec<Align>,
    style: Style,
}

impl Table {
    /// Creates a table with the given column headers.
    pub fn new<I, S>(headers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let headers: Vec<String> = headers.into_iter().map(Into::into).collect();
        let align = vec![Align::Left; headers.len()];
        Table {
            headers,
            rows: Vec::new(),
            align,
            style: Style::Unicode,
        }
    }

    /// Sets the alignment of column `col` (ignored if out of range).
    pub fn set_align(&mut self, col: usize, align: Align) {
        if let Some(a) = self.align.get_mut(col) {
            *a = align;
        }
    }

    /// Chooses the border style.
    pub fn set_style(&mut self, style: Style) {
        self.style = style;
    }

    /// Appends a row. Short rows are padded with empty cells; extra cells are
    /// dropped.
    pub fn add_row<I, S>(&mut self, cells: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut row: Vec<String> = cells.into_iter().map(Into::into).collect();
        row.resize(self.headers.len(), String::new());
        self.rows.push(row);
    }

    /// Number of data rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table has no data rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Renders the table, one line per row, with a trailing newline.
    pub fn render(&self) -> String {
        let width = |s: &str| s.chars().count();
        let widths: Vec<usize> = (0..self.headers.len())
            .map(|c| {
                self.rows
                    .iter()
                    .map(|r| width(&r[c]))
                    .chain(std::iter::once(width(&self.headers[c])))
                    .max()
                    .unwrap_or(0)
            })
            .collect();

        let [tl, tm, tr, ml, mm, mr, bl, bm, br, h, v] = match self.style {
            Style::Unicode => ["┌", "┬", "┐", "├", "┼", "┤", "└", "┴", "┘", "─", "│"],
            Style::Ascii => ["+", "+", "+", "+", "+", "+", "+", "+", "+", "-", "|"],
        };

        let rule = |left: &str, mid: &str, right: &str| {
            let segments: Vec<String> = widths.iter().map(|w| h.repeat(w + 2)).collect();
            format!("{left}{}{right}\n", segments.join(mid))
        };
        let line = |cells: &[String], aligns: &[Align]| {
            let mut out = String::from(v);
            for ((cell, w), align) in cells.iter().zip(&widths).zip(aligns) {
                let pad = " ".repeat(w - width(cell));
                match align {
                    Align::Left => out.push_str(&format!(" {cell}{pad} {v}")),
                    Align::Right => out.push_str(&format!(" {pad}{cell} {v}")),
                }
            }
            out.push('\n');
            out
        };

        let mut out = rule(tl, tm, tr);
        out.push_str(&line(&self.headers, &vec![Align::Left; widths.len()]));
        out.push_str(&rule(ml, mm, mr));
        for row in &self.rows {
            out.push_str(&line(row, &self.align));
        }
        out.push_str(&rule(bl, bm, br));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_table_still_has_headers() {
        let t = Table::new(["a"]);
        assert!(t.is_empty());
        assert_eq!(t.render(), "┌───┐\n│ a │\n├───┤\n└───┘\n");
    }

    #[test]
    fn ascii_style_and_padding_rows() {
        let mut t = Table::new(["k", "value"]);
        t.set_style(Style::Ascii);
        t.add_row(["x"]);
        t.add_row(["y", "z", "dropped"]);
        assert_eq!(t.len(), 2);
        assert_eq!(
            t.render(),
            "+---+-------+\n| k | value |\n+---+-------+\n| x |       |\n| y | z     |\n+---+-------+\n"
        );
    }

    #[test]
    fn unicode_cells_measure_in_chars() {
        let mut t = Table::new(["név"]);
        t.add_row(["🐉"]);
        let rendered = t.render();
        assert!(rendered.contains("│ 🐉   │"));
    }
}
