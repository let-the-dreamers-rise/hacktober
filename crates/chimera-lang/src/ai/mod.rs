//! AI in the syntax.
//!
//! * `ai "prompt"`, `ask "…"`, `claude("…", system: "…")` send a prompt to
//!   the configured backend and return the reply as a string.
//! * `please sort xs descending` runs an English instruction (see [`nl`]).
//! * `embed`, `similarity`, `sentiment`, `summarize`, `keywords`,
//!   `classify`, `extract`, `tokenize` and `markov` are deterministic,
//!   local text tools (see [`text`]).
//! * `import anthropic` gives a client shaped like the official SDKs:
//!   `anthropic.Anthropic().messages.create(model=…, messages=[…])`.
//!
//! Backends: **offline** (rule-based, deterministic; the default without an
//! API key), **claude** (the Messages API, used automatically when
//! `ANTHROPIC_API_KEY` is set) and **mock** (scripted replies for tests).
//! `CHIMERA_AI=offline|claude|mock` and `CHIMERA_AI_MODEL` override the
//! defaults; programs can also set `ai.backend` and `ai.model`.

pub mod claude;
pub mod nl;
pub mod offline;
pub mod text;

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use chimera_core::Span;

use crate::builtins::objects::module as make_module;
use crate::interp::{Args, Interp, sym};
use crate::ml::tensor::Tensor;
use crate::value::*;

/// Which engine answers prompts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Deterministic rule-based answers; no network.
    Offline,
    /// The Anthropic Messages API.
    Claude,
    /// Scripted replies queued with `ai.mock([...])`.
    Mock,
}

impl Backend {
    /// The backend's name as shown to programs.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Offline => "offline",
            Backend::Claude => "claude",
            Backend::Mock => "mock",
        }
    }

    /// Parses a backend name.
    pub fn from_name(s: &str) -> Option<Backend> {
        match s.trim().to_lowercase().as_str() {
            "offline" | "local" | "none" => Some(Backend::Offline),
            "claude" | "anthropic" | "online" | "api" => Some(Backend::Claude),
            "mock" | "fake" | "test" => Some(Backend::Mock),
            _ => None,
        }
    }
}

/// AI settings and state, owned by the interpreter.
#[derive(Debug)]
pub struct AiConfig {
    /// The active backend.
    pub backend: Backend,
    /// Model id for the Claude backend.
    pub model: String,
    /// Output token ceiling per request.
    pub max_tokens: u32,
    /// Optional effort level for the Claude backend.
    pub effort: Option<String>,
    /// Default system prompt.
    pub system: Option<String>,
    /// Number of prompts answered.
    pub calls: u64,
    cache: HashMap<String, String>,
    mock: VecDeque<String>,
    history: Vec<(String, String)>,
}

impl AiConfig {
    /// Settings from the environment (`CHIMERA_AI`, `CHIMERA_AI_MODEL`,
    /// `ANTHROPIC_API_KEY`).
    pub fn from_env() -> AiConfig {
        let backend = std::env::var("CHIMERA_AI")
            .ok()
            .and_then(|b| Backend::from_name(&b))
            .unwrap_or(if claude::api_key().is_some() {
                Backend::Claude
            } else {
                Backend::Offline
            });
        let model = std::env::var("CHIMERA_AI_MODEL")
            .ok()
            .filter(|m| !m.trim().is_empty())
            .unwrap_or_else(|| claude::DEFAULT_MODEL.to_owned());
        AiConfig {
            backend,
            model,
            max_tokens: 16_000,
            effort: None,
            system: None,
            calls: 0,
            cache: HashMap::new(),
            mock: VecDeque::new(),
            history: Vec::new(),
        }
    }

    /// Deterministic offline settings (tests, `--offline`).
    pub fn offline() -> AiConfig {
        AiConfig {
            backend: Backend::Offline,
            ..AiConfig::from_env()
        }
    }

    /// Queues scripted replies and switches to the mock backend.
    pub fn mock(&mut self, replies: impl IntoIterator<Item = String>) {
        self.mock.extend(replies);
        self.backend = Backend::Mock;
    }
}

/// Options for one prompt.
#[derive(Debug, Default)]
pub struct Ask {
    /// System prompt (overrides `ai.system`).
    pub system: Option<String>,
    /// Model (overrides `ai.model`).
    pub model: Option<String>,
    /// Output token ceiling.
    pub max_tokens: Option<u32>,
    /// Effort level.
    pub effort: Option<String>,
    /// Continue the running conversation (`ai.chat`).
    pub chat: bool,
}

/// Sends `prompt` to the active backend.
pub fn complete(interp: &mut Interp, prompt: &str, opts: &Ask, span: Span) -> R<String> {
    interp.ai.calls += 1;
    let reply = match interp.ai.backend {
        Backend::Mock => interp
            .ai
            .mock
            .pop_front()
            .unwrap_or_else(|| format!("mock: {prompt}")),
        Backend::Offline => offline::respond(interp, prompt, span)?,
        Backend::Claude => {
            let cfg = &interp.ai;
            let model = opts.model.clone().unwrap_or_else(|| cfg.model.clone());
            let system = opts.system.clone().or_else(|| cfg.system.clone());
            let effort = opts.effort.clone().or_else(|| cfg.effort.clone());
            let mut turns = if opts.chat {
                cfg.history.clone()
            } else {
                Vec::new()
            };
            turns.push(("user".to_owned(), prompt.to_owned()));
            let key = format!(
                "{model}\u{0}{}\u{0}{prompt}",
                system.as_deref().unwrap_or("")
            );
            match cfg.cache.get(&key).filter(|_| !opts.chat) {
                Some(hit) => hit.clone(),
                None => {
                    let req = claude::Request {
                        model: &model,
                        system: system.as_deref(),
                        messages: &turns,
                        max_tokens: opts.max_tokens.unwrap_or(cfg.max_tokens),
                        effort: effort.as_deref(),
                    };
                    match claude::complete(&req) {
                        Ok(text) => {
                            if !opts.chat {
                                interp.ai.cache.insert(key, text.clone());
                            }
                            text
                        }
                        Err(msg) => return Err(interp.error("AiError", msg, span)),
                    }
                }
            }
        }
    };
    if opts.chat {
        interp
            .ai
            .history
            .push(("user".to_owned(), prompt.to_owned()));
        interp
            .ai
            .history
            .push(("assistant".to_owned(), reply.clone()));
    }
    Ok(reply)
}

/// Removes a surrounding Markdown code fence, if any.
pub fn strip_fences(s: &str) -> String {
    let t = s.trim();
    if let Some(rest) = t.strip_prefix("```") {
        let body = rest.split_once('\n').map_or("", |(_, b)| b);
        return body
            .trim_end()
            .trim_end_matches("```")
            .trim_end()
            .to_owned();
    }
    t.to_owned()
}

fn options(interp: &mut Interp, args: &Args) -> R<Ask> {
    let mut opts = Ask::default();
    for (k, v) in &args.named {
        let name = k.as_str();
        match &*name {
            "system" => opts.system = Some(interp.to_str(v)?),
            "model" => opts.model = Some(interp.to_str(v)?),
            "effort" => opts.effort = Some(interp.to_str(v)?),
            "max_tokens" | "maxTokens" => {
                opts.max_tokens = v.as_int().map(|n| n.clamp(1, 128_000) as u32)
            }
            _ => {}
        }
    }
    Ok(opts)
}

/// `ai(prompt, …)`: extra positional arguments are appended to the prompt
/// (`ai("Summarize this:", article)`); `json: true` parses the reply.
fn ask(interp: &mut Interp, args: Args, chat: bool) -> R {
    let span = args.span;
    if args.pos.is_empty() {
        return interp.type_err("`ai` needs a prompt", span);
    }
    let mut parts = Vec::with_capacity(args.pos.len());
    for v in &args.pos {
        parts.push(interp.to_str(v)?);
    }
    let mut prompt = parts.join("\n\n");
    let mut opts = options(interp, &args)?;
    opts.chat = chat;
    let want_json = args.named("json").is_some_and(|v| v.truthy());
    if want_json {
        prompt.push_str("\n\nRespond with only valid JSON: no prose and no code fences.");
    }
    let reply = complete(interp, &prompt, &opts, span)?;
    if want_json {
        let body = strip_fences(&reply);
        return crate::builtins::json::parse(&body).map_err(|e| {
            interp.error(
                "AiError",
                format!("the reply was not valid JSON ({e}): {body}"),
                span,
            )
        });
    }
    Ok(Value::str(&reply))
}

// ----- the `ai` object ----------------------------------------------------------

/// The `ai` value: callable, with text tools as methods and settings as
/// attributes.
#[derive(Debug)]
pub struct AiObject;

impl Object for AiObject {
    fn type_name(&self) -> &str {
        "ai"
    }

    fn call(&self, interp: &mut Interp, args: Args, _this: &Value) -> Option<R> {
        Some(ask(interp, args, false))
    }

    fn get_attr(&self, interp: &mut Interp, name: &str) -> Option<R> {
        let cfg = &interp.ai;
        Some(Ok(match name {
            "backend" => Value::str(cfg.backend.name()),
            "model" => Value::str(&cfg.model),
            "calls" => Value::Int(cfg.calls as i64),
            "max_tokens" => Value::Int(i64::from(cfg.max_tokens)),
            "effort" => cfg.effort.as_deref().map_or(Value::Nil, Value::str),
            "system" => cfg.system.as_deref().map_or(Value::Nil, Value::str),
            "available" | "online" => Value::Bool(claude::api_key().is_some()),
            "history" => Value::list(
                cfg.history
                    .iter()
                    .map(|(role, content)| {
                        Value::map_from(vec![
                            ("role", Value::str(role)),
                            ("content", Value::str(content)),
                        ])
                    })
                    .collect(),
            ),
            _ => return None,
        }))
    }

    fn set_attr(&self, interp: &mut Interp, name: &str, value: Value) -> bool {
        let text = match &value {
            Value::Nil => None,
            Value::Str(s) => Some(s.to_string()),
            other => Some(crate::interp::format::display_plain(other, false)),
        };
        let cfg = &mut interp.ai;
        match name {
            "backend" => match text.as_deref().and_then(Backend::from_name) {
                Some(b) => cfg.backend = b,
                None => return false,
            },
            "model" => cfg.model = text.unwrap_or_else(|| claude::DEFAULT_MODEL.to_owned()),
            "system" => cfg.system = text,
            "effort" => cfg.effort = text,
            "max_tokens" => match value.as_int() {
                Some(n) => cfg.max_tokens = n.clamp(1, 128_000) as u32,
                None => return false,
            },
            _ => return false,
        }
        true
    }

    fn call_method(&self, interp: &mut Interp, name: &str, args: Args, _this: &Value) -> Option<R> {
        Some(match name {
            "ask" | "complete" | "prompt" | "generate" | "query" | "run_prompt" => {
                ask(interp, args, false)
            }
            "chat" | "say" | "send" => ask(interp, args, true),
            "reset" | "clear" | "forget" => {
                interp.ai.history.clear();
                Ok(Value::Nil)
            }
            "mock" => {
                let span = args.span;
                let replies = match args.get(0) {
                    Some(v) => match interp.collect(&v.clone(), span) {
                        Ok(items) => items,
                        Err(e) => return Some(Err(e)),
                    },
                    None => Vec::new(),
                };
                let mut texts = Vec::with_capacity(replies.len());
                for r in &replies {
                    match interp.to_str(r) {
                        Ok(s) => texts.push(s),
                        Err(e) => return Some(Err(e)),
                    }
                }
                interp.ai.mock(texts);
                Ok(Value::Nil)
            }
            "use" | "use_backend" | "set_backend" => {
                let span = args.span;
                match args
                    .get(0)
                    .map(|v| crate::interp::format::display_plain(v, false))
                {
                    Some(b) => match Backend::from_name(&b) {
                        Some(b) => {
                            interp.ai.backend = b;
                            Ok(Value::Nil)
                        }
                        None => interp.value_err(
                            format!("unknown AI backend `{b}` (try offline, claude or mock)"),
                            span,
                        ),
                    },
                    None => interp.type_err("ai.use needs a backend name", span),
                }
            }
            "translate" => translate(interp, args),
            "code" | "write_code" | "codegen" => code(interp, args),
            other => return tool(other).map(|f| f(interp, args)),
        })
    }

    fn display(&self) -> String {
        "<ai>".into()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

type Tool = fn(&mut Interp, Args) -> R;

/// The local text tools, by name.
fn tool(name: &str) -> Option<Tool> {
    Some(match name {
        "embed" | "embedding" | "embeddings" | "encode" => embed_fn,
        "similarity" | "cosine" | "cosine_similarity" | "similar" => similarity_fn,
        "sentiment" | "analyze_sentiment" | "sentiment_analysis" => sentiment_fn,
        "summarize" | "summarise" | "summary" | "tldr" => summarize_fn,
        "keywords" | "key_words" | "topics" => keywords_fn,
        "classify" | "categorize" | "label" | "zero_shot" => classify_fn,
        "extract" | "entities" | "find_all" => extract_fn,
        "tokenize" | "tokens" | "words" => tokenize_fn,
        "markov" | "markov_chain" | "language_model" => markov_fn,
        "pig_latin" | "piglatin" => pig_latin_fn,
        _ => return None,
    })
}

fn text_arg(interp: &mut Interp, args: &Args, i: usize, fname: &str) -> R<String> {
    let v = crate::builtins::arg(interp, args, i, fname)?;
    interp.to_str(&v)
}

fn vector_of(interp: &mut Interp, v: &Value, span: Span) -> R<Vec<f64>> {
    match v {
        Value::Str(s) => Ok(text::embed(s, 256)),
        other => Ok(interp.to_tensor(other, span)?.data()),
    }
}

fn embed_fn(interp: &mut Interp, args: Args) -> R {
    let s = text_arg(interp, &args, 0, "embed")?;
    let dim = args
        .arg(1, "dim")
        .and_then(|d| d.as_int())
        .unwrap_or(256)
        .clamp(8, 1 << 16) as usize;
    let v = text::embed(&s, dim);
    Ok(Value::Tensor(
        Tensor::new(v, vec![dim]).map_err(|e| interp.error("ValueError", e, args.span))?,
    ))
}

fn similarity_fn(interp: &mut Interp, args: Args) -> R {
    let span = args.span;
    let a = crate::builtins::arg(interp, &args, 0, "similarity")?;
    let b = crate::builtins::arg(interp, &args, 1, "similarity")?;
    let (va, vb) = (vector_of(interp, &a, span)?, vector_of(interp, &b, span)?);
    if va.len() != vb.len() {
        return interp.value_err(
            format!(
                "cannot compare vectors of length {} and {}",
                va.len(),
                vb.len()
            ),
            span,
        );
    }
    Ok(Value::Float(text::cosine(&va, &vb)))
}

fn sentiment_fn(interp: &mut Interp, args: Args) -> R {
    let s = text_arg(interp, &args, 0, "sentiment")?;
    let r = text::sentiment(&s);
    let words = |ws: Vec<String>| Value::list(ws.iter().map(|w| Value::str(w)).collect());
    Ok(Value::map_from(vec![
        ("label", Value::str(r.label)),
        ("score", Value::Float((r.score * 1000.0).round() / 1000.0)),
        ("positive", words(r.positive)),
        ("negative", words(r.negative)),
    ]))
}

fn summarize_fn(interp: &mut Interp, args: Args) -> R {
    let s = text_arg(interp, &args, 0, "summarize")?;
    let n = args
        .arg(1, "sentences")
        .and_then(|n| n.as_int())
        .unwrap_or(2)
        .max(1) as usize;
    Ok(Value::str(&text::summarize(&s, n)))
}

fn keywords_fn(interp: &mut Interp, args: Args) -> R {
    let s = text_arg(interp, &args, 0, "keywords")?;
    let n = args
        .arg(1, "n")
        .and_then(|n| n.as_int())
        .unwrap_or(5)
        .max(1) as usize;
    Ok(Value::list(
        text::keywords(&s, n)
            .iter()
            .map(|w| Value::str(w))
            .collect(),
    ))
}

/// `classify(text, ["a", "b"])` or `classify(text, {"a": [examples], …})`.
fn classify_fn(interp: &mut Interp, args: Args) -> R {
    let span = args.span;
    let s = text_arg(interp, &args, 0, "classify")?;
    let spec = crate::builtins::arg(interp, &args, 1, "classify")?;
    let mut labels = Vec::new();
    match &spec {
        Value::Map(m) => {
            let entries: Vec<(Value, Value)> = m
                .borrow()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            for (k, v) in entries {
                let label = interp.to_str(&k)?;
                let mut examples = Vec::new();
                for e in interp.collect(&v, span)? {
                    examples.push(interp.to_str(&e)?);
                }
                labels.push((label, examples));
            }
        }
        other => {
            for l in interp.collect(other, span)? {
                labels.push((interp.to_str(&l)?, Vec::new()));
            }
        }
    }
    if labels.is_empty() {
        return interp.value_err("classify needs at least one label", span);
    }
    let ranked = text::classify(&s, &labels);
    if args.named("scores").is_some_and(|v| v.truthy()) {
        let mut d = Dict::new();
        for (label, p) in ranked {
            d.insert(
                Value::str(&label),
                Value::Float((p * 1000.0).round() / 1000.0),
            );
        }
        return Ok(Value::Map(Rc::new(RefCell::new(d))));
    }
    Ok(Value::str(&ranked[0].0))
}

fn extract_fn(interp: &mut Interp, args: Args) -> R {
    let s = text_arg(interp, &args, 0, "extract")?;
    let kind = text_arg(interp, &args, 1, "extract")?;
    Ok(Value::list(
        text::extract(&s, &kind)
            .iter()
            .map(|w| Value::str(w))
            .collect(),
    ))
}

fn tokenize_fn(interp: &mut Interp, args: Args) -> R {
    let s = text_arg(interp, &args, 0, "tokenize")?;
    Ok(Value::list(
        text::tokenize(&s).iter().map(|w| Value::str(w)).collect(),
    ))
}

fn pig_latin_fn(interp: &mut Interp, args: Args) -> R {
    let s = text_arg(interp, &args, 0, "pig_latin")?;
    Ok(Value::str(&text::pig_latin(&s)))
}

fn markov_fn(interp: &mut Interp, args: Args) -> R {
    let s = text_arg(interp, &args, 0, "markov")?;
    let order = args
        .arg(1, "order")
        .and_then(|n| n.as_int())
        .unwrap_or(1)
        .clamp(1, 8) as usize;
    Ok(Value::Object(Rc::new(MarkovObject {
        chain: RefCell::new(text::Markov::train(&s, order)),
        corpus: RefCell::new(s),
        order,
        generated: Cell::new(0),
    })))
}

/// `ai.translate(text, "French")`.
fn translate(interp: &mut Interp, args: Args) -> R {
    let span = args.span;
    let s = text_arg(interp, &args, 0, "translate")?;
    let lang = match args.arg(1, "to") {
        Some(v) => interp.to_str(&v)?,
        None => "English".into(),
    };
    let l = lang.to_lowercase();
    if l.contains("pig") {
        return Ok(Value::str(&text::pig_latin(&s)));
    }
    if interp.ai.backend == Backend::Offline && (l == "reverse" || l == "backwards") {
        return Ok(Value::str(&s.chars().rev().collect::<String>()));
    }
    let opts = Ask {
        system: Some("You are a translator. Reply with only the translation.".into()),
        ..Ask::default()
    };
    let out = complete(interp, &format!("Translate to {lang}:\n\n{s}"), &opts, span)?;
    Ok(Value::str(out.trim()))
}

/// `ai.code("a function that reverses a string")` → source text.
fn code(interp: &mut Interp, args: Args) -> R {
    let span = args.span;
    let s = text_arg(interp, &args, 0, "ai.code")?;
    let opts = Ask {
        system: Some(nl::CODEGEN_SYSTEM.into()),
        ..Ask::default()
    };
    let out = complete(interp, &s, &opts, span)?;
    Ok(Value::str(&strip_fences(&out)))
}

/// A trained Markov chain: `m.generate(20)`, `m.train(more_text)`.
#[derive(Debug)]
pub struct MarkovObject {
    chain: RefCell<text::Markov>,
    corpus: RefCell<String>,
    order: usize,
    generated: Cell<u64>,
}

impl Object for MarkovObject {
    fn type_name(&self) -> &str {
        "MarkovChain"
    }

    fn get_attr(&self, _interp: &mut Interp, name: &str) -> Option<R> {
        Some(Ok(match name {
            "order" => Value::Int(self.order as i64),
            "states" => Value::Int(self.chain.borrow().states() as i64),
            "generated" => Value::Int(self.generated.get() as i64),
            _ => return None,
        }))
    }

    fn call_method(&self, interp: &mut Interp, name: &str, args: Args, this: &Value) -> Option<R> {
        Some(match name {
            "generate" | "sample" | "babble" | "speak" => {
                let n = args
                    .arg(0, "words")
                    .and_then(|n| n.as_int())
                    .unwrap_or(20)
                    .clamp(1, 10_000) as usize;
                let start = match args.arg(1, "start") {
                    Some(Value::Str(s)) => Some(s.to_string()),
                    _ => None,
                };
                self.generated.set(self.generated.get() + 1);
                let chain = self.chain.borrow();
                let rng = &mut interp.rng;
                let out = chain.generate(n, start.as_deref(), |k| rng.index(k));
                Ok(Value::str(&out))
            }
            "train" | "feed" | "learn" => {
                let s = match text_arg(interp, &args, 0, "train") {
                    Ok(s) => s,
                    Err(e) => return Some(Err(e)),
                };
                let mut corpus = self.corpus.borrow_mut();
                corpus.push(' ');
                corpus.push_str(&s);
                *self.chain.borrow_mut() = text::Markov::train(&corpus, self.order);
                Ok(this.clone())
            }
            _ => return None,
        })
    }

    fn display(&self) -> String {
        format!(
            "<MarkovChain order={} states={}>",
            self.order,
            self.chain.borrow().states()
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// An `anthropic.Anthropic()` client (Python / TypeScript SDK shape).
#[derive(Debug)]
struct ClientObject;

/// `client.messages`.
#[derive(Debug)]
struct MessagesApi;

impl Object for ClientObject {
    fn type_name(&self) -> &str {
        "Anthropic"
    }

    fn get_attr(&self, _interp: &mut Interp, name: &str) -> Option<R> {
        match name {
            "messages" => Some(Ok(Value::Object(Rc::new(MessagesApi)))),
            _ => None,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Object for MessagesApi {
    fn type_name(&self) -> &str {
        "Messages"
    }

    fn call_method(&self, interp: &mut Interp, name: &str, args: Args, _this: &Value) -> Option<R> {
        match name {
            "create" | "Create" | "New" | "new" => Some(messages_create(interp, args)),
            _ => None,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `client.messages.create(model=…, max_tokens=…, system=…, messages=[…])`,
/// or the same as a single map (`create({model: …, messages: […]})`).
fn messages_create(interp: &mut Interp, args: Args) -> R {
    let span = args.span;
    let mut params: Vec<(String, Value)> = args
        .named
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.clone()))
        .collect();
    if let Some(Value::Map(m)) = args.get(0) {
        for (k, v) in m.borrow().iter() {
            params.push((crate::interp::format::display_plain(k, false), v.clone()));
        }
    }
    let get = |name: &str| {
        params
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
    };
    let mut opts = Ask::default();
    if let Some(v) = get("model") {
        opts.model = Some(interp.to_str(&v)?);
    }
    if let Some(v) = get("system") {
        opts.system = Some(interp.to_str(&v)?);
    }
    opts.max_tokens = get("max_tokens")
        .or_else(|| get("maxTokens"))
        .and_then(|v| v.as_int())
        .map(|n| n.clamp(1, 128_000) as u32);
    let Some(messages) = get("messages") else {
        return interp.type_err("messages.create needs `messages`", span);
    };
    // Flatten the conversation into one prompt for non-Claude backends; the
    // Claude backend gets the real turns.
    let mut turns = Vec::new();
    for m in interp.collect(&messages, span)? {
        let (role, content) = match &m {
            Value::Map(d) => {
                let d = d.borrow();
                (
                    d.get_str("role").unwrap_or(Value::str("user")),
                    d.get_str("content").unwrap_or(Value::Nil),
                )
            }
            other => (Value::str("user"), other.clone()),
        };
        let role = interp.to_str(&role)?;
        let text = match &content {
            Value::List(blocks) => {
                let mut out = String::new();
                for b in blocks.borrow().clone() {
                    let t = match &b {
                        Value::Map(bm) => bm.borrow().get_str("text").unwrap_or(Value::Nil),
                        other => other.clone(),
                    };
                    out.push_str(&interp.to_str(&t)?);
                }
                out
            }
            other => interp.to_str(other)?,
        };
        turns.push((role, text));
    }
    let Some((_, last)) = turns.pop() else {
        return interp.value_err("messages.create needs at least one message", span);
    };
    let saved = std::mem::replace(&mut interp.ai.history, turns);
    opts.chat = true;
    let result = complete(interp, &last, &opts, span);
    interp.ai.history = saved;
    let reply = result?;
    let model = opts.model.unwrap_or_else(|| interp.ai.model.clone());
    let id = format!("msg_chimera_{:04}", interp.ai.calls);
    Ok(Value::map_from(vec![
        ("id", Value::str(&id)),
        ("type", Value::str("message")),
        ("role", Value::str("assistant")),
        ("model", Value::str(&model)),
        (
            "content",
            Value::list(vec![Value::map_from(vec![
                ("type", Value::str("text")),
                ("text", Value::str(&reply)),
            ])]),
        ),
        ("stop_reason", Value::str("end_turn")),
    ]))
}

// ----- installation ---------------------------------------------------------------

/// Registers `ai`, `ask`, `claude` and the text tools as globals.
pub fn install(interp: &mut Interp) {
    let ai = Value::Object(Rc::new(AiObject));
    for n in ["ai", "AI", "ask", "claude", "Claude", "llm", "LLM"] {
        interp.builtins.insert(sym(n), ai.clone());
    }
    let tools: &[(&[&str], Tool)] = &[
        (&["embed", "embedding"], embed_fn),
        (
            &["similarity", "cosine_similarity", "cosineSimilarity"],
            similarity_fn,
        ),
        (
            &["sentiment", "analyze_sentiment", "analyzeSentiment"],
            sentiment_fn,
        ),
        (&["summarize", "summarise"], summarize_fn),
        (&["keywords"], keywords_fn),
        (&["classify", "zero_shot"], classify_fn),
        (&["extract_entities", "extractEntities"], extract_fn),
        (&["tokenize", "word_tokenize"], tokenize_fn),
        (&["markov", "MarkovChain", "markov_chain"], markov_fn),
        (&["pig_latin", "pigLatin"], pig_latin_fn),
    ];
    for (names, f) in tools {
        crate::builtins::def(interp, names, *f);
    }
}

/// Importable AI modules: `ai`/`llm`/`nlp` and `anthropic`.
pub fn module(interp: &mut Interp, name: &str) -> Option<Value> {
    match name {
        "ai" | "llm" | "nlp" | "claude" | "chimera.ai" => {
            let ai = Value::Object(Rc::new(AiObject));
            let mut members: Vec<(&str, Value)> =
                vec![("ai", ai.clone()), ("ask", ai.clone()), ("claude", ai)];
            for n in [
                "embed",
                "similarity",
                "sentiment",
                "summarize",
                "keywords",
                "classify",
                "extract",
                "tokenize",
                "markov",
                "pig_latin",
            ] {
                if let Some(f) = tool(n) {
                    members.push((n, Value::native(n, f)));
                }
            }
            let _ = interp;
            Some(make_module(name, members))
        }
        "anthropic" | "@anthropic-ai/sdk" | "anthropic-sdk" => {
            let client = Value::native("Anthropic", |_interp, _args| {
                Ok(Value::Object(Rc::new(ClientObject)))
            });
            Some(make_module(
                "anthropic",
                vec![
                    ("Anthropic", client.clone()),
                    ("Client", client.clone()),
                    ("default", client),
                    ("DEFAULT_MODEL", Value::str(claude::DEFAULT_MODEL)),
                ],
            ))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_names_round_trip() {
        for b in [Backend::Offline, Backend::Claude, Backend::Mock] {
            assert_eq!(Backend::from_name(b.name()), Some(b));
        }
        assert_eq!(Backend::from_name("nope"), None);
    }

    #[test]
    fn strips_code_fences() {
        assert_eq!(strip_fences("```python\nprint(1)\n```"), "print(1)");
        assert_eq!(strip_fences("  x = 1  "), "x = 1");
    }
}
