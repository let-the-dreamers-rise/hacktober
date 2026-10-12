//! The Claude backend: the Anthropic Messages API, spoken over a `curl`
//! subprocess so Chimera stays free of HTTP and TLS dependencies.
//!
//! The API key travels to curl on stdin (`-H @-`), never on the command
//! line, so it doesn't show up in `ps`. The request body goes through a
//! temporary file that is removed as soon as curl exits.
//!
//! Requests opt into server-side fallbacks (`fallbacks: "default"`), so a
//! request declined by a safety classifier is retried on Anthropic's
//! recommended fallback model instead of failing outright. A refusal from
//! the whole chain still comes back as `stop_reason: "refusal"`, which is
//! surfaced as an `AiError`.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::builtins::json::{self, quote};
use crate::value::Value;

/// The model used unless `CHIMERA_AI_MODEL` or `ai.model = …` says otherwise.
pub const DEFAULT_MODEL: &str = "claude-opus-5-5";

/// Messages API endpoint.
const API_URL: &str = "https://api.anthropic.com/v1/messages";

/// API version header value.
const API_VERSION: &str = "2023-06-01";

/// Beta header enabling `fallbacks: "default"`.
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

/// One request to the Messages API.
#[derive(Debug)]
pub struct Request<'a> {
    /// Model id, e.g. `claude-opus-5-5`.
    pub model: &'a str,
    /// Optional system prompt.
    pub system: Option<&'a str>,
    /// Conversation turns as `(role, text)`; the last should be `"user"`.
    pub messages: &'a [(String, String)],
    /// Output token ceiling.
    pub max_tokens: u32,
    /// Optional effort level (`low` … `max`).
    pub effort: Option<&'a str>,
}

impl Request<'_> {
    /// The JSON request body.
    pub fn body(&self) -> String {
        let mut b = String::from("{");
        b.push_str(&format!("\"model\":{},", quote(self.model)));
        b.push_str(&format!("\"max_tokens\":{},", self.max_tokens));
        b.push_str("\"fallbacks\":\"default\",");
        if let Some(system) = self.system {
            b.push_str(&format!("\"system\":{},", quote(system)));
        }
        if let Some(effort) = self.effort {
            b.push_str(&format!(
                "\"output_config\":{{\"effort\":{}}},",
                quote(effort)
            ));
        }
        b.push_str("\"messages\":[");
        for (i, (role, text)) in self.messages.iter().enumerate() {
            if i > 0 {
                b.push(',');
            }
            b.push_str(&format!(
                "{{\"role\":{},\"content\":{}}}",
                quote(role),
                quote(text)
            ));
        }
        b.push_str("]}");
        b
    }
}

/// The API key from the environment, if any.
pub fn api_key() -> Option<String> {
    std::env::var("ANTHROPIC_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty())
}

/// Sends a request and returns the reply text.
pub fn complete(req: &Request<'_>) -> Result<String, String> {
    let Some(key) = api_key() else {
        return Err("the Claude backend needs ANTHROPIC_API_KEY to be set".into());
    };
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("chimera-ai-{}-{n}.json", std::process::id()));
    std::fs::write(&path, req.body())
        .map_err(|e| format!("could not write the request body: {e}"))?;
    let result = run_curl(&key, &path);
    let _ = std::fs::remove_file(&path);
    parse_response(&result?)
}

fn run_curl(key: &str, body_path: &std::path::Path) -> Result<String, String> {
    let mut child = Command::new("curl")
        .args(["-sS", "--max-time", "600", "-H", "@-", "--data-binary"])
        .arg(format!("@{}", body_path.display()))
        .arg(API_URL)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run curl ({e}); install curl to use the Claude backend"))?;
    if let Some(mut stdin) = child.stdin.take() {
        let headers = format!(
            "content-type: application/json\nx-api-key: {key}\nanthropic-version: {API_VERSION}\nanthropic-beta: {FALLBACK_BETA}\n"
        );
        stdin
            .write_all(headers.as_bytes())
            .map_err(|e| format!("could not send headers to curl: {e}"))?;
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("curl failed: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("curl failed: {}", err.trim()));
    }
    String::from_utf8(out.stdout).map_err(|_| "the API returned invalid UTF-8".into())
}

/// Extracts the reply text from a Messages API response body.
pub fn parse_response(body: &str) -> Result<String, String> {
    let v = json::parse(body).map_err(|e| format!("unexpected response from the API ({e})"))?;
    let Value::Map(m) = v else {
        return Err("unexpected response from the API".into());
    };
    let m = m.borrow();
    let as_str = |v: Option<Value>| match v {
        Some(Value::Str(s)) => Some(s.to_string()),
        _ => None,
    };
    if as_str(m.get_str("type")).as_deref() == Some("error") {
        let msg = match m.get_str("error") {
            Some(Value::Map(e)) => {
                let e = e.borrow();
                let kind = as_str(e.get_str("type")).unwrap_or_default();
                let msg = as_str(e.get_str("message")).unwrap_or_default();
                format!("{kind}: {msg}")
            }
            _ => "unknown error".into(),
        };
        return Err(format!("the Claude API returned an error ({msg})"));
    }
    // Check the stop reason before reading content: a refusal can arrive
    // with no content at all.
    if as_str(m.get_str("stop_reason")).as_deref() == Some("refusal") {
        return Err("Claude declined this request (stop_reason: refusal)".into());
    }
    let mut text = String::new();
    if let Some(Value::List(blocks)) = m.get_str("content") {
        for block in blocks.borrow().iter() {
            if let Value::Map(b) = block {
                let b = b.borrow();
                if as_str(b.get_str("type")).as_deref() == Some("text")
                    && let Some(t) = as_str(b.get_str("text"))
                {
                    text.push_str(&t);
                }
            }
        }
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_request_bodies() {
        let turns = vec![("user".to_string(), "Say \"hi\"\n".to_string())];
        let req = Request {
            model: DEFAULT_MODEL,
            system: Some("Be brief."),
            messages: &turns,
            max_tokens: 1024,
            effort: Some("low"),
        };
        let body = req.body();
        assert!(json::parse(&body).is_ok(), "{body}");
        assert!(body.contains(r#""content":"Say \"hi\"\n""#), "{body}");
        assert!(body.contains(r#""fallbacks":"default""#));
        assert!(body.contains(r#""output_config":{"effort":"low"}"#));
    }

    #[test]
    fn parses_replies_errors_and_refusals() {
        let ok = r#"{"type":"message","content":[{"type":"text","text":"Hello"},{"type":"text","text":" there"}],"stop_reason":"end_turn"}"#;
        assert_eq!(parse_response(ok).unwrap(), "Hello there");
        let err = r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#;
        assert!(
            parse_response(err)
                .unwrap_err()
                .contains("authentication_error")
        );
        let refusal = r#"{"type":"message","content":[],"stop_reason":"refusal"}"#;
        assert!(parse_response(refusal).unwrap_err().contains("refusal"));
    }
}
