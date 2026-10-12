//! The offline assistant: a deterministic, rule-based stand-in for a model.
//!
//! It handles the prompts small programs most often send — arithmetic,
//! summaries, sentiment, keywords, extraction, classification, pig latin,
//! haiku, jokes — and answers everything else with an honest note about
//! being offline. Because it is deterministic, programs that use `ai "…"`
//! can be tested without network access.

use chimera_core::{Span, fnv1a64};

use super::text;
use crate::interp::Interp;
use crate::value::*;

/// Answers `prompt` without a model.
pub fn respond(interp: &mut Interp, prompt: &str, span: Span) -> R<String> {
    let p = prompt.trim();
    let lower = p.to_lowercase();
    let starts = |words: &[&str]| words.iter().any(|w| lower.starts_with(w));

    if starts(&["hello", "hi ", "hi!", "hey", "greetings"]) || lower == "hi" {
        return Ok(
            "Hello! I'm Chimera's offline assistant. Set ANTHROPIC_API_KEY to talk to Claude."
                .into(),
        );
    }
    if starts(&["summarize", "summarise", "tl;dr", "tldr", "sum up"]) {
        return Ok(text::summarize(
            &payload(p, &["summarize", "summarise", "tl;dr", "tldr", "sum up"]),
            2,
        ));
    }
    if lower.contains("sentiment")
        || starts(&["is this positive", "is this negative", "how does this feel"])
    {
        let s = text::sentiment(&payload(
            p,
            &[
                "sentiment of",
                "sentiment",
                "is this positive",
                "is this negative",
            ],
        ));
        return Ok(format!("{} ({:+.2})", s.label, s.score));
    }
    if starts(&[
        "keywords",
        "key words",
        "topics",
        "extract keywords",
        "list keywords",
    ]) {
        return Ok(text::keywords(&payload(p, &["keywords", "key words", "topics"]), 5).join(", "));
    }
    if lower.contains("pig latin") {
        let body = match p.find(':') {
            Some(i) => p[i + 1..].trim().to_owned(),
            None => {
                let start = lower.find("translate").map_or(0, |i| i + "translate".len());
                let end = lower
                    .find(" to pig latin")
                    .or_else(|| lower.find(" into pig latin"))
                    .unwrap_or(p.len());
                p[start.min(end)..end]
                    .trim()
                    .trim_matches(['"', '\''])
                    .to_owned()
            }
        };
        return Ok(text::pig_latin(&body));
    }
    if let Some(kind) = lower
        .strip_prefix("extract ")
        .and_then(|r| r.split_whitespace().next())
    {
        let body = payload(p, &["from"]);
        let found = text::extract(&body, kind);
        return Ok(if found.is_empty() {
            "(none found)".into()
        } else {
            found.join(", ")
        });
    }
    if lower.starts_with("classify ") {
        // "classify <text> as a, b or c"
        if let Some(i) = lower.rfind(" as ") {
            let body = p["classify ".len()..i].trim().trim_matches(['"', '\'']);
            let labels: Vec<(String, Vec<String>)> = p[i + 4..]
                .split([',', '/'])
                .flat_map(|s| s.split(" or "))
                .map(|s| s.trim().trim_end_matches(['?', '.']).to_owned())
                .filter(|s| !s.is_empty())
                .map(|s| (s, Vec::new()))
                .collect();
            if !labels.is_empty() {
                return Ok(text::classify(body, &labels)[0].0.clone());
            }
        }
    }
    if starts(&["reverse"]) {
        return Ok(payload(p, &["reverse"]).chars().rev().collect());
    }
    if starts(&["uppercase", "upper case", "shout"]) {
        return Ok(payload(p, &["uppercase", "upper case", "shout"]).to_uppercase());
    }
    if starts(&["lowercase", "lower case", "whisper"]) {
        return Ok(payload(p, &["lowercase", "lower case", "whisper"]).to_lowercase());
    }
    if starts(&[
        "count the words",
        "count words",
        "how many words",
        "word count",
    ]) {
        let body = payload(
            p,
            &[
                "in",
                "count the words",
                "count words",
                "how many words",
                "word count",
            ],
        );
        return Ok(text::tokenize(&body).len().to_string());
    }
    if lower.contains("haiku") {
        let topic = lower
            .split_once(" about ")
            .map_or("code", |(_, t)| t)
            .trim_end_matches(['.', '?', '!'])
            .trim();
        return Ok(haiku(topic));
    }
    if lower.contains("joke") {
        return Ok(pick(JOKES, p).to_owned());
    }
    if let Some(answer) = arithmetic(interp, p, span)? {
        return Ok(answer);
    }
    Ok(format!(
        "I'm Chimera's offline assistant, so I can't answer that. Set ANTHROPIC_API_KEY to send prompts to Claude. (prompt: {})",
        chimera_core::text::escape_debug(p)
    ))
}

/// The text a command applies to: after the first `:` if there is one,
/// otherwise after the command word.
fn payload(p: &str, words: &[&str]) -> String {
    if let Some(i) = p.find(':') {
        return p[i + 1..].trim().trim_matches(['"', '\'']).to_owned();
    }
    let lower = p.to_lowercase();
    for w in words {
        if let Some(i) = lower.find(w) {
            let rest = p[i + w.len()..].trim_start();
            let rest = ["of ", "this ", "the ", "in ", "for "]
                .iter()
                .fold(rest, |r, pre| {
                    if r.to_lowercase().starts_with(pre) {
                        &r[pre.len()..]
                    } else {
                        r
                    }
                });
            return rest.trim().trim_matches(['"', '\'']).to_owned();
        }
    }
    p.to_owned()
}

/// `what is 2 + 2`, `calculate 3 * (4 + 5)`, `17 ** 3?`.
fn arithmetic(interp: &mut Interp, p: &str, span: Span) -> R<Option<String>> {
    let lower = p.to_lowercase();
    let mut expr = lower.as_str();
    for pre in [
        "what is",
        "what's",
        "calculate",
        "compute",
        "evaluate",
        "solve",
        "how much is",
    ] {
        if let Some(rest) = expr.strip_prefix(pre) {
            expr = rest;
            break;
        }
    }
    let expr = expr.trim().trim_end_matches(['?', '.', '!', '=']).trim();
    let allowed = |c: char| c.is_ascii_digit() || " +-*/%^().,_e".contains(c) || c.is_alphabetic();
    if expr.is_empty() || !expr.chars().any(|c| c.is_ascii_digit()) || !expr.chars().all(allowed) {
        return Ok(None);
    }
    // Spell out the common English operators.
    let expr = expr
        .replace(" plus ", " + ")
        .replace(" minus ", " - ")
        .replace(" times ", " * ")
        .replace(" multiplied by ", " * ")
        .replace(" divided by ", " / ")
        .replace(" to the power of ", " ** ")
        .replace('^', " ** ");
    // Only letters that spell known math functions may remain.
    let ok_words = [
        "sqrt",
        "sin",
        "cos",
        "tan",
        "log",
        "exp",
        "abs",
        "pi",
        "e",
        "floor",
        "ceil",
        "factorial",
        "of",
        "the",
        "square",
        "root",
    ];
    if expr
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .any(|w| !ok_words.contains(&w))
    {
        return Ok(None);
    }
    let file = interp.add_file("<ai>", &expr);
    let (parsed, errors) = crate::parser::parse_expr_range(&expr, file, 0, expr.len());
    let Some(e) = parsed else { return Ok(None) };
    if !errors.is_empty() {
        return Ok(None);
    }
    let env = crate::interp::Env::child(&interp.globals.clone(), crate::interp::EnvKind::Block);
    let _ = span;
    match interp.eval(&e, &env) {
        Ok(v) => Ok(Some(interp.to_str(&v)?)),
        Err(_) => Ok(None),
    }
}

fn pick<'a>(options: &[&'a str], seed: &str) -> &'a str {
    options[(fnv1a64(seed.as_bytes()) % options.len() as u64) as usize]
}

const JOKES: &[&str] = &[
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "There are 10 kinds of people: those who understand binary and those who don't.",
    "A SQL query walks into a bar, walks up to two tables and asks: may I join you?",
    "I would tell you a UDP joke, but you might not get it.",
    "Chimera speaks fourteen languages and still can't explain a regex.",
];

/// A 5-7-5 haiku built from templates (syllables are counted, and the
/// topic only goes where it fits).
fn haiku(topic: &str) -> String {
    let n = text::tokenize(topic)
        .iter()
        .map(|w| text::syllables(w))
        .sum::<usize>()
        .clamp(1, 5);
    // First lines: topic + filler to reach five syllables.
    let fill5 = ["", "o", "the", "quiet", "softly the", "autumn wind and"];
    let opener = match n {
        5 => topic.to_string(),
        _ => format!("{} {topic}", fill5[5 - n]).trim().to_owned(),
    };
    let middles = [
        "the compiler hums along",
        "a thousand tokens drifting",
        "leaves fall through the stack frames",
        "every branch returns at last",
    ];
    let closers = [
        "nothing left to fix",
        "the tests are green now",
        "silence after rain",
        "one more pull request",
    ];
    format!(
        "{opener}\n{}\n{}",
        pick(&middles, topic),
        pick(&closers, &format!("{topic}!"))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_extraction() {
        assert_eq!(payload("summarize: a. b.", &["summarize"]), "a. b.");
        assert_eq!(
            payload("sentiment of I love it", &["sentiment"]),
            "I love it"
        );
        assert_eq!(payload("reverse \"abc\"", &["reverse"]), "abc");
    }

    #[test]
    fn haiku_has_three_lines() {
        let h = haiku("rust");
        assert_eq!(h.lines().count(), 3);
        assert!(h.lines().next().unwrap().ends_with("rust"));
    }
}
