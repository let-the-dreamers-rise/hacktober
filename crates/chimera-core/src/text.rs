//! Text utilities: edit distance, suggestions and human-readable units.

use std::time::Duration;

/// Levenshtein edit distance between two strings, counted in characters.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let substitution = prev[j] + usize::from(ca != cb);
            cur[j + 1] = substitution.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Picks the candidate closest to `name`, if any is close enough to be a
/// plausible typo (distance at most one third of the name's length, min 1).
///
/// Ties are broken by candidate order, so callers get stable suggestions.
///
/// ```
/// use chimera_core::text::suggest;
/// assert_eq!(suggest("lenght", ["len", "length", "list"]), Some("length"));
/// assert_eq!(suggest("zzz", ["length"]), None);
/// ```
pub fn suggest<'a>(name: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let limit = (name.chars().count() / 3).max(1);
    let lowered = name.to_lowercase();
    candidates
        .into_iter()
        .filter(|c| *c != name)
        .map(|c| {
            // A pure case mismatch is the most likely typo of all.
            let d = if c.to_lowercase() == lowered {
                0
            } else {
                levenshtein(name, c)
            };
            (d, c)
        })
        .filter(|(d, _)| *d <= limit)
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c)
}

/// Formats a byte count with binary units, e.g. `1.5 KiB`.
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

/// Formats a duration compactly, e.g. `850µs`, `12.3ms`, `4.20s`, `3m07s`.
pub fn format_duration(d: Duration) -> String {
    let micros = d.as_micros();
    if micros < 1_000 {
        format!("{micros}µs")
    } else if micros < 1_000_000 {
        format!("{:.1}ms", micros as f64 / 1_000.0)
    } else if d.as_secs() < 60 {
        format!("{:.2}s", d.as_secs_f64())
    } else {
        format!("{}m{:02}s", d.as_secs() / 60, d.as_secs() % 60)
    }
}

/// Quotes and escapes a string the way Chimera and SQL literals print it.
pub fn escape_debug(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            c if c.is_control() => out.push_str(&format!("\\u{{{:x}}}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("flaw", "lawn"), 2);
        assert_eq!(levenshtein("héllo", "hello"), 1);
    }

    #[test]
    fn suggestions() {
        let names = ["print", "println", "parse_int", "push"];
        assert_eq!(suggest("prnt", names), Some("print"));
        assert_eq!(suggest("pussh", names), Some("push"));
        assert_eq!(suggest("PRINT", names), Some("print"));
        assert_eq!(suggest("completely_off", names), None);
        // An exact match is not a suggestion.
        assert_eq!(suggest("print", ["print"]), None);
    }

    #[test]
    fn bytes_and_durations() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.5 KiB");
        assert_eq!(format_bytes(3 * 1024 * 1024), "3.0 MiB");
        assert_eq!(format_duration(Duration::from_micros(850)), "850µs");
        assert_eq!(format_duration(Duration::from_micros(12_300)), "12.3ms");
        assert_eq!(format_duration(Duration::from_millis(4_200)), "4.20s");
        assert_eq!(format_duration(Duration::from_secs(187)), "3m07s");
    }

    #[test]
    fn escaping() {
        assert_eq!(escape_debug("a\"b\\c\nd"), r#""a\"b\\c\nd""#);
        assert_eq!(escape_debug("\u{7}"), r#""\u{7}""#);
    }
}
