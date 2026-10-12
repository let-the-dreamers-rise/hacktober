//! `please …`: English as a statement.
//!
//! ```text
//! please sort scores descending
//! please add 42 to scores and then print scores
//! please keep the even numbers in xs
//! please plot sales
//! total = please the sum of prices
//! ```
//!
//! Instructions are matched against a table of patterns first, so common
//! requests run instantly, offline and deterministically. Anything the
//! table doesn't cover is sent to the AI backend, which translates it into
//! Chimera code that then runs in the current scope (with the offline
//! backend, that's an error explaining what is understood).

use std::rc::Rc;

use chimera_core::Span;

use super::{Ask, Backend};
use crate::ast::BinOp;
use crate::interp::{Env, Flow, Interp, sym};
use crate::value::*;

/// The system prompt used when an instruction is translated into code.
pub const CODEGEN_SYSTEM: &str = "You translate instructions into Chimera code. Chimera is a polyglot \
language: it accepts Python, JavaScript, Ruby, Lua, Go, Rust, Kotlin and Haskell syntax, freely mixed. \
Prefer simple Python-style code. Reply with only the code: no explanations and no Markdown fences.";

/// A pattern element.
#[derive(Debug)]
enum Pat {
    /// One of these words (the matched word is recorded).
    Word(Vec<&'static str>),
    /// Optionally one of these words.
    Opt(Vec<&'static str>),
    /// One or more words, captured.
    Cap,
}

fn compile(pattern: &'static str) -> Vec<Pat> {
    pattern
        .split_whitespace()
        .map(|t| {
            if t.starts_with('{') {
                Pat::Cap
            } else if let Some(inner) = t.strip_prefix('[').and_then(|t| t.strip_suffix(']')) {
                Pat::Opt(inner.split('|').collect())
            } else {
                Pat::Word(t.split('|').collect())
            }
        })
        .collect()
}

/// What a successful match produced.
#[derive(Debug, Default, Clone)]
struct Match {
    caps: Vec<String>,
    words: Vec<String>,
}

fn match_at(pat: &[Pat], words: &[String], m: &mut Match) -> bool {
    let Some((first, rest)) = pat.split_first() else {
        return words.is_empty();
    };
    match first {
        Pat::Word(alts) => {
            let Some(w) = words.first() else { return false };
            if !alts.contains(&w.to_lowercase().as_str()) {
                return false;
            }
            m.words.push(w.to_lowercase());
            if match_at(rest, &words[1..], m) {
                return true;
            }
            m.words.pop();
            false
        }
        Pat::Opt(alts) => {
            if let Some(w) = words.first()
                && alts.contains(&w.to_lowercase().as_str())
            {
                m.words.push(w.to_lowercase());
                if match_at(rest, &words[1..], m) {
                    return true;
                }
                m.words.pop();
            }
            match_at(rest, words, m)
        }
        Pat::Cap => {
            // Shortest capture first, so later literal words anchor it.
            let max = words.len();
            let min = 1;
            let range: Box<dyn Iterator<Item = usize>> = if rest.is_empty() {
                Box::new(max..=max)
            } else {
                Box::new(min..=max)
            };
            for n in range {
                if n == 0 {
                    continue;
                }
                m.caps.push(words[..n].join(" "));
                if match_at(rest, &words[n..], m) {
                    return true;
                }
                m.caps.pop();
            }
            false
        }
    }
}

/// The instruction table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    SortByDesc,
    SortBy,
    SortDesc,
    Sort,
    Reverse,
    Shuffle,
    Print,
    Push,
    PushFront,
    FilterKeep,
    FilterDrop,
    Unique,
    Remove,
    Set,
    IncBy,
    Inc,
    DecBy,
    Dec,
    MapOp,
    Sum,
    Mean,
    Max,
    Min,
    Count,
    Swap,
    StrOp,
    Join,
    CountTo,
    CountDown,
    Clear,
    Describe,
    Plot,
    Eval,
}

const NUMBERS: &str =
    "[number|numbers|item|items|element|elements|value|values|ones|entries|elements]";

fn rules() -> Vec<(Vec<Pat>, Act)> {
    let filt = "even|odd|positive|negative|nonzero|non-zero|zero|empty|truthy|falsy|nil|null";
    let table: Vec<(String, Act)> = vec![
        ("sort {x} by {k} descending|decreasing|reversed|desc".into(), Act::SortByDesc),
        ("sort {x} by {k}".into(), Act::SortBy),
        ("sort {x} [in] descending|decreasing|reverse|desc|backwards [order]".into(), Act::SortDesc),
        ("sort {x} [in] ascending|increasing|asc [order]".into(), Act::Sort),
        ("sort {x}".into(), Act::Sort),
        ("reverse {x}".into(), Act::Reverse),
        ("shuffle {x}".into(), Act::Shuffle),
        (format!("keep|filter|select|retain [only] [the] [all] {filt} {NUMBERS} in|from|of {{x}}"), Act::FilterKeep),
        (format!("keep|filter|select|retain [only] [the] [all] {filt} {NUMBERS} {{x}}"), Act::FilterKeep),
        (format!("remove|drop|delete|discard [all] [the] {filt} {NUMBERS} from|in|of {{x}}"), Act::FilterDrop),
        ("remove|drop|delete [the] [all] duplicates from|in {x}".into(), Act::Unique),
        ("unique|dedupe|deduplicate|uniq|uniquify {x}".into(), Act::Unique),
        ("remove|delete|drop {v} from {x}".into(), Act::Remove),
        ("print|show|display|say|output|echo|log {x}".into(), Act::Print),
        ("add|append|push {v} to|onto {x}".into(), Act::Push),
        ("prepend|unshift {v} to|onto {x}".into(), Act::PushFront),
        ("insert {v} at [the] start|front|beginning|head of {x}".into(), Act::PushFront),
        ("set|make {x} to|equal|= [to] {v}".into(), Act::Set),
        ("let {x} be|= {v}".into(), Act::Set),
        ("assign {v} to {x}".into(), Act::Set),
        ("increment|increase|bump|raise {x} by {v}".into(), Act::IncBy),
        ("increment|increase|bump {x}".into(), Act::Inc),
        ("decrement|decrease|reduce|lower {x} by {v}".into(), Act::DecBy),
        ("decrement|decrease {x}".into(), Act::Dec),
        (
            format!("double|triple|square|cube|negate|halve|increment|decrement|abs|round [every|each|all] [of] [the] {NUMBERS} in|of {{x}}"),
            Act::MapOp,
        ),
        ("double|triple|square|cube|negate|halve|abs|round [every|each|all] {x}".into(), Act::MapOp),
        ("count [up] to {v}".into(), Act::CountTo),
        ("count down from {v}".into(), Act::CountDown),
        ("[what|what's] [is] [the] sum|total [of] {x}".into(), Act::Sum),
        ("add up {x}".into(), Act::Sum),
        ("[what|what's] [is] [the] average|mean [of] {x}".into(), Act::Mean),
        ("[what|what's] [is] [the] max|maximum|largest|biggest|highest [value] [of|in] {x}".into(), Act::Max),
        ("[what|what's] [is] [the] min|minimum|smallest|lowest [value] [of|in] {x}".into(), Act::Min),
        ("[what|what's] [is] [the] count|length|size|len [of] {x}".into(), Act::Count),
        (format!("how many {NUMBERS} [are] [there] in {{x}}"), Act::Count),
        ("how many things|words|characters|chars|letters|keys [are] [there] in {x}".into(), Act::Count),
        ("swap {a} and|with {b}".into(), Act::Swap),
        ("uppercase|upcase|capitalize|capitalise|lowercase|downcase|titlecase|title-case|trim|strip {x}".into(), Act::StrOp),
        ("make {x} uppercase|lowercase|capitalized|upper|lower".into(), Act::StrOp),
        ("join {x} with|using|by {v}".into(), Act::Join),
        ("clear|empty|reset {x}".into(), Act::Clear),
        ("describe|inspect|summarize|summarise|analyze|analyse {x}".into(), Act::Describe),
        ("tell me about {x}".into(), Act::Describe),
        ("plot|chart|graph|visualize|visualise|draw {x}".into(), Act::Plot),
        ("what is|what's {x}".into(), Act::Eval),
        ("calculate|compute|evaluate|eval|return|give me {x}".into(), Act::Eval),
    ];
    table
        .into_iter()
        .map(|(p, a)| (compile(Box::leak(p.into_boxed_str())), a))
        .collect()
}

thread_local! {
    static RULES: Vec<(Vec<Pat>, Act)> = rules();
}

/// Splits an instruction into words, keeping quoted strings and bracketed
/// expressions together.
fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    for c in text.chars() {
        match quote {
            Some(q) => {
                cur.push(c);
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' if cur.is_empty() || depth > 0 => {
                    quote = Some(c);
                    cur.push(c);
                }
                '(' | '[' | '{' => {
                    depth += 1;
                    cur.push(c);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    cur.push(c);
                }
                c if c.is_whitespace() && depth <= 0 => {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                }
                c => cur.push(c),
            },
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    // Trailing punctuation on the last word: "please print xs."
    if let Some(last) = out.last_mut() {
        while last.len() > 1 && last.ends_with(['.', '!', '?']) && !last.ends_with("..") {
            last.pop();
        }
    }
    out
}

/// Splits "do this and then that, then the other" into steps.
fn steps(text: &str) -> Vec<String> {
    let mut parts = vec![text.to_owned()];
    for sep in [
        ", and then ",
        " and then ",
        ", then ",
        "; then ",
        " then ",
        "; ",
    ] {
        parts = parts
            .iter()
            .flat_map(|p| p.split(sep).map(str::to_owned).collect::<Vec<_>>())
            .collect();
    }
    parts
        .into_iter()
        .map(|p| p.trim().to_owned())
        .filter(|p| !p.is_empty())
        .collect()
}

/// Runs a natural-language instruction in `env`.
pub fn please(interp: &mut Interp, text: &str, env: &Rc<Env>, span: Span) -> R {
    // `please("sort xs")`, `please "sort xs"`, `please instruction_var`.
    let t = text.trim();
    if t.starts_with(['(', '"', '\''])
        || (!t.contains(' ') && env.get(sym(t)).is_some_and(|v| matches!(v, Value::Str(_))))
    {
        if let Ok(Value::Str(s)) = eval_text(interp, t, env, span) {
            return please(interp, &s, env, span);
        }
    }
    let mut last = Value::Nil;
    let mut it: Option<String> = None;
    for step in steps(text) {
        last = run_step(interp, &step, env, span, &mut it)?;
    }
    Ok(last)
}

fn run_step(
    interp: &mut Interp,
    step: &str,
    env: &Rc<Env>,
    span: Span,
    it: &mut Option<String>,
) -> R {
    let mut ws = words(step);
    // Politeness is free.
    while ws.first().is_some_and(|w| {
        matches!(
            w.to_lowercase().as_str(),
            "please" | "kindly" | "just" | "now" | "then" | "also"
        )
    }) {
        ws.remove(0);
    }
    if ws.last().is_some_and(|w| w.eq_ignore_ascii_case("please")) {
        ws.pop();
    }
    if ws.is_empty() {
        return Ok(Value::Nil);
    }
    // "<instruction> N times"
    if ws.len() >= 3 && ws[ws.len() - 1].eq_ignore_ascii_case("times") {
        let n_text = ws[ws.len() - 2].clone();
        if let Some(n) = number_word(&n_text).or_else(|| n_text.parse().ok()) {
            let inner = ws[..ws.len() - 2].join(" ");
            let mut last = Value::Nil;
            for _ in 0..n.max(0) {
                last = run_step(interp, &inner, env, span, it)?;
            }
            return Ok(last);
        }
    }
    // Resolve pronouns to the previous step's subject.
    if let Some(prev) = it.as_ref() {
        for w in &mut ws {
            if matches!(w.to_lowercase().as_str(), "it" | "them" | "that" | "this") {
                *w = prev.clone();
            }
        }
    }
    let found = RULES.with(|rules| {
        for (pat, act) in rules {
            let mut m = Match::default();
            if match_at(pat, &ws, &mut m) {
                return Some((*act, m));
            }
        }
        None
    });
    match found {
        Some((act, m)) => {
            if let Some(subject) = subject_of(act, &m) {
                *it = Some(subject);
            }
            match perform(interp, act, &m, env, span) {
                // The pattern matched but its pieces weren't code (`add pears
                // to the inventory, seven of them`): let the model try.
                Err(Flow::Throw(t))
                    if interp.ai.backend != Backend::Offline
                        && matches!(&t.value, Value::Instance(i) if i.class.has_ancestor_named("NameError") || i.class.has_ancestor_named("SyntaxError")) =>
                {
                    fallback(interp, step, env, span)
                }
                other => other,
            }
        }
        None => fallback(interp, step, env, span),
    }
}

fn subject_of(act: Act, m: &Match) -> Option<String> {
    let i = match act {
        Act::Push | Act::PushFront | Act::Remove => 1,
        Act::Set => {
            // "assign v to x" captures (v, x); the others capture (x, v).
            if m.words.first().is_some_and(|w| w == "assign") {
                1
            } else {
                0
            }
        }
        Act::CountTo | Act::CountDown | Act::Eval => return None,
        _ => 0,
    };
    m.caps.get(i).cloned()
}

fn number_word(w: &str) -> Option<i64> {
    const W: &[&str] = &[
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ];
    let lower = w.to_lowercase();
    if let Some(i) = W.iter().position(|x| *x == lower) {
        return Some(i as i64);
    }
    match lower.as_str() {
        "once" => Some(1),
        "twice" => Some(2),
        "thrice" => Some(3),
        "a" | "an" => Some(1),
        "hundred" | "a hundred" => Some(100),
        "thousand" | "a thousand" => Some(1000),
        _ => None,
    }
}

/// Evaluates a captured phrase as a Chimera expression.
fn eval_text(interp: &mut Interp, text: &str, env: &Rc<Env>, span: Span) -> R {
    let t = text.trim();
    if let Some(n) = number_word(t) {
        return Ok(Value::Int(n));
    }
    let t = t.strip_prefix("the ").unwrap_or(t);
    let t = t.strip_prefix("list ").unwrap_or(t);
    let file = interp.add_file("<please>", t);
    let (parsed, errors) = crate::parser::parse_expr_range(t, file, 0, t.len());
    match parsed {
        Some(e) if errors.is_empty() => {
            let saved = interp.cur_file;
            interp.cur_file = file;
            let r = interp.eval(&e, env);
            interp.cur_file = saved;
            r
        }
        _ => Err(interp.error(
            "SyntaxError",
            format!("please: could not understand `{t}`"),
            span,
        )),
    }
}

/// Assigns to a captured phrase (`xs`, `obj.field`, `xs[0]`).
fn assign_text(interp: &mut Interp, text: &str, value: Value, env: &Rc<Env>, span: Span) -> R<()> {
    let t = text.trim();
    let t = t.strip_prefix("the ").unwrap_or(t);
    let file = interp.add_file("<please>", t);
    let (parsed, errors) = crate::parser::parse_expr_range(t, file, 0, t.len());
    match parsed {
        Some(e) if errors.is_empty() => interp.assign(&e, value, env),
        _ => Err(interp.error(
            "SyntaxError",
            format!("please: cannot assign to `{t}`"),
            span,
        )),
    }
}

/// Replaces the contents of the collection named by `target` (in place for
/// lists, by reassignment otherwise).
fn replace(
    interp: &mut Interp,
    target: &str,
    old: &Value,
    items: Vec<Value>,
    env: &Rc<Env>,
    span: Span,
) -> R {
    match old {
        Value::List(l) => {
            *l.borrow_mut() = items;
            Ok(old.clone())
        }
        Value::Tuple(_) => {
            let v = Value::tuple(items);
            assign_text(interp, target, v.clone(), env, span)?;
            Ok(v)
        }
        Value::Str(_) => {
            let mut s = String::new();
            for x in &items {
                s.push_str(&interp.to_str(x)?);
            }
            let v = Value::str(&s);
            assign_text(interp, target, v.clone(), env, span)?;
            Ok(v)
        }
        _ => {
            let v = Value::list(items);
            assign_text(interp, target, v.clone(), env, span)?;
            Ok(v)
        }
    }
}

fn items_of(interp: &mut Interp, v: &Value, span: Span) -> R<Vec<Value>> {
    match v {
        Value::Str(s) => Ok(s.chars().map(|c| Value::str(&c.to_string())).collect()),
        Value::Map(m) => Ok(m.borrow().values()),
        other => interp.collect(other, span),
    }
}

fn perform(interp: &mut Interp, act: Act, m: &Match, env: &Rc<Env>, span: Span) -> R {
    let cap = |i: usize| m.caps.get(i).cloned().unwrap_or_default();
    match act {
        Act::Sort | Act::SortDesc | Act::SortBy | Act::SortByDesc => {
            let target = cap(0);
            let v = eval_text(interp, &target, env, span)?;
            let items = items_of(interp, &v, span)?;
            let key = if matches!(act, Act::SortBy | Act::SortByDesc) {
                let field = cap(1);
                let field = field
                    .trim_start_matches("their ")
                    .trim_start_matches("the ")
                    .trim_start_matches("its ")
                    .to_owned();
                Some(Value::native("please.key", move |interp, args| {
                    let item = args.get(0).cloned().unwrap_or(Value::Nil);
                    match &item {
                        Value::Map(d) => Ok(d.borrow().get_str(&field).unwrap_or(Value::Nil)),
                        _ => interp.get_attr(&item, sym(&field), args.span),
                    }
                }))
            } else {
                None
            };
            let desc = matches!(act, Act::SortDesc | Act::SortByDesc);
            let sorted =
                crate::builtins::core::sort_values(interp, items, key.as_ref(), desc, span)?;
            replace(interp, &target, &v, sorted, env, span)
        }
        Act::Reverse => {
            let target = cap(0);
            let v = eval_text(interp, &target, env, span)?;
            let mut items = items_of(interp, &v, span)?;
            items.reverse();
            replace(interp, &target, &v, items, env, span)
        }
        Act::Shuffle => {
            let target = cap(0);
            let v = eval_text(interp, &target, env, span)?;
            let mut items = items_of(interp, &v, span)?;
            interp.rng.shuffle(&mut items);
            replace(interp, &target, &v, items, env, span)
        }
        Act::Print => {
            // `please say hello world` prints the words when they aren't an
            // expression.
            let text = cap(0);
            let s = match eval_text(interp, &text, env, span) {
                Ok(v) => interp.to_str(&v)?,
                Err(_) => text.trim_matches(['"', '\'']).to_owned(),
            };
            interp.write_out(&s);
            interp.write_out("\n");
            Ok(Value::Nil)
        }
        Act::Push | Act::PushFront => {
            let item = eval_text(interp, &cap(0), env, span)?;
            let target = cap(1);
            let v = eval_text(interp, &target, env, span)?;
            match &v {
                Value::Set(s) => {
                    s.borrow_mut().insert(item, Value::Nil);
                    Ok(v.clone())
                }
                Value::Str(s) => {
                    let piece = interp.to_str(&item)?;
                    let joined = if act == Act::Push {
                        format!("{s}{piece}")
                    } else {
                        format!("{piece}{s}")
                    };
                    let nv = Value::str(&joined);
                    assign_text(interp, &target, nv.clone(), env, span)?;
                    Ok(nv)
                }
                Value::Int(_) | Value::Float(_) | Value::Big(_) => {
                    let nv = interp.binary_op(BinOp::Add, v.clone(), item, span)?;
                    assign_text(interp, &target, nv.clone(), env, span)?;
                    Ok(nv)
                }
                _ => {
                    let mut items = items_of(interp, &v, span)?;
                    if act == Act::Push {
                        items.push(item);
                    } else {
                        items.insert(0, item);
                    }
                    replace(interp, &target, &v, items, env, span)
                }
            }
        }
        Act::Remove => {
            let item = eval_text(interp, &cap(0), env, span)?;
            let target = cap(1);
            let v = eval_text(interp, &target, env, span)?;
            if let Value::Map(d) = &v {
                d.borrow_mut().remove(&item);
                return Ok(v.clone());
            }
            if let Value::Set(d) = &v {
                d.borrow_mut().remove(&item);
                return Ok(v.clone());
            }
            let mut items = items_of(interp, &v, span)?;
            let mut pos = None;
            for (i, x) in items.iter().enumerate() {
                if interp.values_equal(x, &item)? {
                    pos = Some(i);
                    break;
                }
            }
            match pos {
                Some(i) => {
                    items.remove(i);
                    replace(interp, &target, &v, items, env, span)
                }
                None => {
                    let s = interp.repr(&item)?;
                    Err(interp.error(
                        "ValueError",
                        format!("please: {s} is not in {target}"),
                        span,
                    ))
                }
            }
        }
        Act::FilterKeep | Act::FilterDrop => {
            let which = m.words.iter().find(|w| {
                matches!(
                    w.as_str(),
                    "even"
                        | "odd"
                        | "positive"
                        | "negative"
                        | "nonzero"
                        | "non-zero"
                        | "zero"
                        | "empty"
                        | "truthy"
                        | "falsy"
                        | "nil"
                        | "null"
                )
            });
            let which = which.cloned().unwrap_or_default();
            let target = cap(0);
            let v = eval_text(interp, &target, env, span)?;
            let items = items_of(interp, &v, span)?;
            let keep = act == Act::FilterKeep;
            let kept: Vec<Value> = items
                .into_iter()
                .filter(|x| {
                    let f = x.as_f64();
                    let ok = match which.as_str() {
                        "even" => x.as_int().is_some_and(|i| i % 2 == 0),
                        "odd" => x.as_int().is_some_and(|i| i % 2 != 0),
                        "positive" => f.is_some_and(|f| f > 0.0),
                        "negative" => f.is_some_and(|f| f < 0.0),
                        "nonzero" | "non-zero" => f.is_some_and(|f| f != 0.0),
                        "zero" => f == Some(0.0),
                        "empty" | "falsy" => !x.truthy(),
                        "nil" | "null" => matches!(x, Value::Nil),
                        _ => x.truthy(),
                    };
                    ok == keep
                })
                .collect();
            replace(interp, &target, &v, kept, env, span)
        }
        Act::Unique => {
            let target = cap(0);
            let v = eval_text(interp, &target, env, span)?;
            let items = items_of(interp, &v, span)?;
            let mut out: Vec<Value> = Vec::new();
            for x in items {
                let mut dup = false;
                for y in &out {
                    if interp.values_equal(&x, y)? {
                        dup = true;
                        break;
                    }
                }
                if !dup {
                    out.push(x);
                }
            }
            replace(interp, &target, &v, out, env, span)
        }
        Act::Set => {
            let (target, value) = if m.words.first().is_some_and(|w| w == "assign") {
                (cap(1), cap(0))
            } else {
                (cap(0), cap(1))
            };
            let v = eval_text(interp, &value, env, span)?;
            assign_text(interp, &target, v.clone(), env, span)?;
            Ok(v)
        }
        Act::Inc | Act::IncBy | Act::Dec | Act::DecBy => {
            let target = cap(0);
            let by = match act {
                Act::IncBy | Act::DecBy => eval_text(interp, &cap(1), env, span)?,
                _ => Value::Int(1),
            };
            let v = eval_text(interp, &target, env, span)?;
            let op = if matches!(act, Act::Inc | Act::IncBy) {
                BinOp::Add
            } else {
                BinOp::Sub
            };
            let nv = interp.binary_op(op, v, by, span)?;
            assign_text(interp, &target, nv.clone(), env, span)?;
            Ok(nv)
        }
        Act::MapOp => {
            let op = m.words.first().cloned().unwrap_or_default();
            let target = cap(0);
            let v = eval_text(interp, &target, env, span)?;
            let items = items_of(interp, &v, span)?;
            let mut out = Vec::with_capacity(items.len());
            for x in items {
                out.push(map_op(interp, &op, x, span)?);
            }
            replace(interp, &target, &v, out, env, span)
        }
        Act::Sum | Act::Mean | Act::Max | Act::Min | Act::Count => {
            let v = eval_text(interp, &cap(0), env, span)?;
            if act == Act::Count {
                let n = match &v {
                    Value::Str(s) if m.words.iter().any(|w| w == "words") => {
                        s.split_whitespace().count()
                    }
                    Value::Str(s) => s.chars().count(),
                    other => items_of(interp, other, span)?.len(),
                };
                return Ok(Value::Int(n as i64));
            }
            let items = items_of(interp, &v, span)?;
            if items.is_empty() {
                return match act {
                    Act::Sum => Ok(Value::Int(0)),
                    _ => interp.value_err("please: the collection is empty", span),
                };
            }
            match act {
                Act::Sum | Act::Mean => {
                    let n = items.len();
                    let mut acc = Value::Int(0);
                    for x in items {
                        acc = interp.binary_op(BinOp::Add, acc, x, span)?;
                    }
                    if act == Act::Mean {
                        interp.binary_op(BinOp::Div, acc, Value::Int(n as i64), span)
                    } else {
                        Ok(acc)
                    }
                }
                _ => {
                    let mut best = items[0].clone();
                    for x in items.into_iter().skip(1) {
                        let ord = interp.compare_values(&x, &best, span)?;
                        let better = if act == Act::Max {
                            ord.is_gt()
                        } else {
                            ord.is_lt()
                        };
                        if better {
                            best = x;
                        }
                    }
                    Ok(best)
                }
            }
        }
        Act::Swap => {
            let (a, b) = (cap(0), cap(1));
            let va = eval_text(interp, &a, env, span)?;
            let vb = eval_text(interp, &b, env, span)?;
            assign_text(interp, &a, vb, env, span)?;
            assign_text(interp, &b, va, env, span)?;
            Ok(Value::Nil)
        }
        Act::StrOp => {
            let op = m
                .words
                .iter()
                .find(|w| *w != "make")
                .cloned()
                .unwrap_or_default();
            let target = cap(0);
            let v = eval_text(interp, &target, env, span)?;
            let s = interp.to_str(&v)?;
            let out = match op.as_str() {
                "uppercase" | "upcase" | "upper" => s.to_uppercase(),
                "lowercase" | "downcase" | "lower" => s.to_lowercase(),
                "trim" | "strip" => s.trim().to_owned(),
                "capitalize" | "capitalise" | "capitalized" => {
                    let mut c = s.chars();
                    match c.next() {
                        Some(f) => {
                            f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase()
                        }
                        None => String::new(),
                    }
                }
                _ => s
                    .split(' ')
                    .map(|w| {
                        let mut c = w.chars();
                        match c.next() {
                            Some(f) => {
                                f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase()
                            }
                            None => String::new(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            };
            let nv = Value::str(&out);
            if matches!(v, Value::Str(_)) {
                assign_text(interp, &target, nv.clone(), env, span)?;
            }
            Ok(nv)
        }
        Act::Join => {
            let v = eval_text(interp, &cap(0), env, span)?;
            let sep_v = eval_text(interp, &cap(1), env, span)?;
            let sep = match cap(1).as_str() {
                "commas" | "comma" => ", ".to_owned(),
                "spaces" | "space" => " ".to_owned(),
                "dashes" | "dash" => "-".to_owned(),
                "newlines" | "newline" | "lines" => "\n".to_owned(),
                _ => interp.to_str(&sep_v)?,
            };
            let mut parts = Vec::new();
            for x in items_of(interp, &v, span)? {
                parts.push(interp.to_str(&x)?);
            }
            Ok(Value::str(&parts.join(&sep)))
        }
        Act::CountTo | Act::CountDown => {
            let n = eval_text(interp, &cap(0), env, span)?.as_int().unwrap_or(0);
            let seq: Vec<String> = if act == Act::CountTo {
                (1..=n.min(10_000)).map(|i| i.to_string()).collect()
            } else {
                (0..=n.min(10_000)).rev().map(|i| i.to_string()).collect()
            };
            interp.write_out(&seq.join(" "));
            interp.write_out("\n");
            Ok(Value::Nil)
        }
        Act::Clear => {
            let target = cap(0);
            let v = eval_text(interp, &target, env, span)?;
            match &v {
                Value::Map(d) | Value::Set(d) => {
                    d.borrow_mut().clear();
                    Ok(v.clone())
                }
                Value::Int(_) | Value::Float(_) => {
                    assign_text(interp, &target, Value::Int(0), env, span)?;
                    Ok(Value::Int(0))
                }
                _ => replace(interp, &target, &v, Vec::new(), env, span),
            }
        }
        Act::Describe => {
            let target = cap(0);
            let v = eval_text(interp, &target, env, span)?;
            let line = describe(interp, &target, &v, span)?;
            interp.write_out(&line);
            interp.write_out("\n");
            Ok(Value::Nil)
        }
        Act::Plot => {
            let target = cap(0);
            let v = eval_text(interp, &target, env, span)?;
            let chart = plot(interp, &v, span)?;
            interp.write_out(&chart);
            Ok(Value::Nil)
        }
        Act::Eval => {
            let text = cap(0);
            match eval_text(interp, &text, env, span) {
                Ok(v) => Ok(v),
                Err(_) => fallback(interp, &format!("what is {text}"), env, span),
            }
        }
    }
}

fn map_op(interp: &mut Interp, op: &str, x: Value, span: Span) -> R {
    match op {
        "double" => interp.binary_op(BinOp::Mul, x, Value::Int(2), span),
        "triple" => interp.binary_op(BinOp::Mul, x, Value::Int(3), span),
        "square" => interp.binary_op(BinOp::Mul, x.clone(), x, span),
        "cube" => {
            let sq = interp.binary_op(BinOp::Mul, x.clone(), x.clone(), span)?;
            interp.binary_op(BinOp::Mul, sq, x, span)
        }
        "negate" => interp.binary_op(BinOp::Sub, Value::Int(0), x, span),
        "halve" => interp.binary_op(BinOp::Div, x, Value::Int(2), span),
        "increment" => interp.binary_op(BinOp::Add, x, Value::Int(1), span),
        "decrement" => interp.binary_op(BinOp::Sub, x, Value::Int(1), span),
        "abs" => interp.math_fn("abs", x, span),
        "round" => interp.math_fn("round", x, span),
        _ => Ok(x),
    }
}

fn describe(interp: &mut Interp, name: &str, v: &Value, span: Span) -> R<String> {
    let t = v.type_name();
    Ok(match v {
        Value::Str(s) => format!(
            "{name} is a string of {} characters and {} words",
            s.chars().count(),
            s.split_whitespace().count()
        ),
        Value::Map(d) => {
            let keys: Vec<String> = d
                .borrow()
                .keys()
                .iter()
                .map(|k| crate::interp::format::display_plain(k, false))
                .collect();
            format!(
                "{name} is a {t} with {} keys: {}",
                keys.len(),
                keys.join(", ")
            )
        }
        Value::List(_) | Value::Tuple(_) | Value::Set(_) | Value::Range(_) | Value::Tensor(_) => {
            let items = items_of(interp, v, span)?;
            let nums: Vec<f64> = items.iter().filter_map(Value::as_f64).collect();
            if !items.is_empty() && nums.len() == items.len() {
                let n = nums.len() as f64;
                let sum: f64 = nums.iter().sum();
                let mean = sum / n;
                let min = nums.iter().copied().fold(f64::INFINITY, f64::min);
                let max = nums.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let var = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
                format!(
                    "{name} is a {t} of {} numbers: min {}, max {}, mean {}, std {}, sum {}",
                    items.len(),
                    num(min),
                    num(max),
                    num((mean * 1000.0).round() / 1000.0),
                    num((var.sqrt() * 1000.0).round() / 1000.0),
                    num(sum)
                )
            } else {
                format!("{name} is a {t} of {} items", items.len())
            }
        }
        other => {
            let r = interp.repr(other)?;
            format!("{name} is a {t}: {r}")
        }
    })
}

/// A horizontal bar chart.
fn plot(interp: &mut Interp, v: &Value, span: Span) -> R<String> {
    let mut rows: Vec<(String, f64)> = Vec::new();
    match v {
        Value::Map(d) => {
            let entries: Vec<(Value, Value)> = d
                .borrow()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            for (k, x) in entries {
                rows.push((interp.to_str(&k)?, x.as_f64().unwrap_or(0.0)));
            }
        }
        other => {
            for (i, x) in items_of(interp, other, span)?.into_iter().enumerate() {
                rows.push((i.to_string(), x.as_f64().unwrap_or(0.0)));
            }
        }
    }
    let label_w = rows
        .iter()
        .map(|(l, _)| l.chars().count())
        .max()
        .unwrap_or(1);
    let max = rows.iter().map(|(_, x)| x.abs()).fold(0.0, f64::max);
    let width = 40.0;
    let mut out = String::new();
    for (label, x) in rows {
        let n = if max > 0.0 {
            (x.abs() / max * width).round() as usize
        } else {
            0
        };
        let bar = if x < 0.0 {
            "░".repeat(n)
        } else {
            "█".repeat(n)
        };
        let pad = label_w - label.chars().count();
        out.push_str(&format!("{}{label} │{bar} {}\n", " ".repeat(pad), num(x)));
    }
    Ok(out)
}

fn num(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        crate::interp::format::fmt_float(x)
    }
}

/// Instructions the table doesn't know go to the AI backend, which writes
/// Chimera code that runs in the current scope.
fn fallback(interp: &mut Interp, text: &str, env: &Rc<Env>, span: Span) -> R {
    if interp.ai.backend == Backend::Offline {
        return Err(interp.error(
            "AiError",
            format!(
                "please: I don't understand {} offline. I know: sort/reverse/shuffle X, add V to X, remove V from X, \
                 keep the even numbers in X, double every number in X, set X to V, increment X, the sum/average/max/min of X, \
                 describe X, plot X, print X, count to N. Set ANTHROPIC_API_KEY to let Claude handle anything else.",
                chimera_core::text::escape_debug(text)
            ),
            span,
        ));
    }
    let mut scope = String::new();
    let names = env.all_names();
    for n in names.iter().take(40) {
        if let Some(v) = env.get(*n) {
            if matches!(v, Value::Native(_) | Value::Module(_)) {
                continue;
            }
            let mut r = crate::interp::format::repr_plain(&v);
            if r.chars().count() > 80 {
                r = r.chars().take(77).collect::<String>() + "...";
            }
            scope.push_str(&format!("  {} = {r}  # {}\n", n.as_str(), v.type_name()));
        }
    }
    let prompt = format!(
        "Variables in scope:\n{}\nInstruction: {text}\n\nWrite Chimera code that carries out the instruction using these variables. \
         The value of the last expression is the result.",
        if scope.is_empty() {
            "  (none)\n".to_owned()
        } else {
            scope
        }
    );
    let opts = Ask {
        system: Some(CODEGEN_SYSTEM.into()),
        ..Ask::default()
    };
    let reply = super::complete(interp, &prompt, &opts, span)?;
    let code = super::strip_fences(&reply);
    let file = interp.add_file("<please: generated>", &code);
    let (stmts, errors) = crate::parser::parse(&code, file);
    if let Some(e) = errors.first() {
        return Err(interp.error(
            "AiError",
            format!(
                "please: the generated code did not parse ({}):\n{code}",
                e.message
            ),
            span,
        ));
    }
    interp.exec_snippet(&stmts, file, env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_words_and_steps() {
        assert_eq!(
            words("add [1, 2] to xs."),
            vec!["add", "[1, 2]", "to", "xs"]
        );
        assert_eq!(words("print \"a b\""), vec!["print", "\"a b\""]);
        assert_eq!(
            steps("sort xs and then print xs, then reverse it"),
            vec!["sort xs", "print xs", "reverse it"]
        );
    }

    #[test]
    fn patterns_capture() {
        let pat = compile("add|append {v} to {x}");
        let mut m = Match::default();
        let ws = words("append 3 to my_list");
        assert!(match_at(&pat, &ws, &mut m));
        assert_eq!(m.caps, vec!["3", "my_list"]);
        assert_eq!(m.words, vec!["append", "to"]);
        let pat = compile("sort {x} [in] descending [order]");
        let mut m = Match::default();
        assert!(match_at(
            &pat,
            &words("sort scores in descending order"),
            &mut m
        ));
        assert_eq!(m.caps, vec!["scores"]);
    }
}
