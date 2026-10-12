//! Offline text intelligence: tokenising, hashed embeddings, sentiment,
//! extractive summaries, keywords, nearest-centroid classification, entity
//! extraction and Markov-chain text generation.
//!
//! Everything here is deterministic and dependency-free, so programs using
//! `sentiment(...)` or `summarize(...)` behave the same on every machine
//! and in CI. With the Claude backend enabled, `ai(...)` answers free-form
//! prompts; these helpers stay local either way.

use std::collections::HashMap;

use chimera_core::fnv1a64;

/// Words that carry little meaning on their own.
const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "even", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "may", "me", "might",
    "more", "most", "much", "must", "my", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "us", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "it's", "i'm", "don't", "isn't", "can't", "won't",
    "let", "lets", "get", "got", "like", "make", "made",
];

/// Sentiment lexicon: word → valence in [-3, 3].
const LEXICON: &[(&str, f64)] = &[
    ("amazing", 3.0),
    ("awesome", 3.0),
    ("brilliant", 3.0),
    ("excellent", 3.0),
    ("fantastic", 3.0),
    ("incredible", 3.0),
    ("love", 3.0),
    ("loved", 3.0),
    ("loves", 3.0),
    ("outstanding", 3.0),
    ("perfect", 3.0),
    ("superb", 3.0),
    ("wonderful", 3.0),
    ("best", 2.5),
    ("delightful", 2.5),
    ("thrilled", 2.5),
    ("beautiful", 2.0),
    ("enjoy", 2.0),
    ("enjoyed", 2.0),
    ("great", 2.0),
    ("happy", 2.0),
    ("impressive", 2.0),
    ("joy", 2.0),
    ("lovely", 2.0),
    ("pleased", 2.0),
    ("recommend", 2.0),
    ("fun", 1.5),
    ("glad", 1.5),
    ("good", 1.5),
    ("helpful", 1.5),
    ("like", 1.0),
    ("liked", 1.5),
    ("nice", 1.5),
    ("powerful", 1.5),
    ("smooth", 1.5),
    ("thanks", 1.5),
    ("win", 1.5),
    ("works", 1.0),
    ("clean", 1.0),
    ("easy", 1.0),
    ("fast", 1.0),
    ("fine", 0.5),
    ("ok", 0.5),
    ("okay", 0.5),
    ("worth", 1.0),
    ("abysmal", -3.0),
    ("awful", -3.0),
    ("disgusting", -3.0),
    ("hate", -3.0),
    ("hated", -3.0),
    ("horrible", -3.0),
    ("terrible", -3.0),
    ("worst", -3.0),
    ("disaster", -2.5),
    ("useless", -2.5),
    ("angry", -2.0),
    ("bad", -2.0),
    ("broken", -2.0),
    ("crash", -2.0),
    ("crashes", -2.0),
    ("disappointed", -2.0),
    ("disappointing", -2.0),
    ("fail", -2.0),
    ("failed", -2.0),
    ("fails", -2.0),
    ("painful", -2.0),
    ("poor", -2.0),
    ("sad", -2.0),
    ("ugly", -2.0),
    ("annoying", -1.5),
    ("boring", -1.5),
    ("bug", -1.5),
    ("buggy", -2.0),
    ("confusing", -1.5),
    ("difficult", -1.0),
    ("hard", -0.5),
    ("problem", -1.5),
    ("slow", -1.5),
    ("wrong", -1.5),
    ("meh", -1.0),
    ("mediocre", -1.5),
    ("messy", -1.5),
    ("worse", -2.0),
];

const NEGATORS: &[&str] = &[
    "not", "no", "never", "n't", "isn't", "wasn't", "don't", "doesn't", "didn't", "can't", "won't",
    "hardly", "nothing",
];
const INTENSIFIERS: &[(&str, f64)] = &[
    ("very", 1.5),
    ("really", 1.4),
    ("extremely", 1.8),
    ("so", 1.3),
    ("super", 1.5),
    ("incredibly", 1.8),
    ("absolutely", 1.6),
    ("totally", 1.4),
    ("slightly", 0.5),
    ("somewhat", 0.6),
    ("kinda", 0.6),
];

/// Lower-cased word tokens (letters, digits, apostrophes).
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() || (c == '\'' && !cur.is_empty()) {
            cur.extend(c.to_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur).trim_end_matches('\'').to_owned());
        }
    }
    if !cur.is_empty() {
        out.push(cur.trim_end_matches('\'').to_owned());
    }
    out
}

/// Whether `w` is a stopword.
pub fn is_stopword(w: &str) -> bool {
    STOPWORDS.contains(&w)
}

/// Splits text into sentences.
pub fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let end = matches!(c, '.' | '!' | '?' | '\n');
        let next_is_space = chars.get(i + 1).is_none_or(|n| n.is_whitespace());
        if end && next_is_space {
            let s = cur.trim().to_owned();
            if !s.is_empty() {
                out.push(s);
            }
            cur.clear();
        }
    }
    let s = cur.trim().to_owned();
    if !s.is_empty() {
        out.push(s);
    }
    out
}

/// A feature-hashed embedding of word unigrams, bigrams and character
/// trigrams, L2-normalised. Similar texts get nearby vectors.
pub fn embed(text: &str, dim: usize) -> Vec<f64> {
    let dim = dim.max(8);
    let mut v = vec![0.0; dim];
    let mut add = |feature: &str, weight: f64| {
        let h = fnv1a64(feature.as_bytes());
        let idx = (h % dim as u64) as usize;
        let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
        v[idx] += sign * weight;
    };
    let words = tokenize(text);
    for w in &words {
        let weight = if is_stopword(w) { 0.25 } else { 1.0 };
        add(&format!("w:{w}"), weight);
        let padded: Vec<char> = format!("#{w}#").chars().collect();
        for tri in padded.windows(3) {
            let t: String = tri.iter().collect();
            add(&format!("c:{t}"), 0.35 * weight);
        }
    }
    for pair in words.windows(2) {
        add(&format!("b:{} {}", pair[0], pair[1]), 0.5);
    }
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Cosine similarity of two vectors (0 for zero vectors).
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let dot: f64 = (0..n).map(|i| a[i] * b[i]).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Sentiment analysis result.
#[derive(Clone, Debug, PartialEq)]
pub struct Sentiment {
    /// Score in [-1, 1].
    pub score: f64,
    /// `"positive"`, `"negative"` or `"neutral"`.
    pub label: &'static str,
    /// Words that pushed the score up.
    pub positive: Vec<String>,
    /// Words that pushed the score down.
    pub negative: Vec<String>,
}

/// Lexicon-based sentiment with negation ("not good"), intensifiers
/// ("very good") and exclamation emphasis.
pub fn sentiment(text: &str) -> Sentiment {
    let words = tokenize(text);
    let mut total = 0.0;
    let mut positive = Vec::new();
    let mut negative = Vec::new();
    for (i, w) in words.iter().enumerate() {
        let Some(&(_, base)) = LEXICON.iter().find(|(k, _)| k == w) else {
            continue;
        };
        let mut val = base;
        // Look back up to three words for negators and intensifiers.
        let lo = i.saturating_sub(3);
        let mut negated = false;
        for prev in &words[lo..i] {
            if NEGATORS.contains(&prev.as_str()) || prev.ends_with("n't") {
                negated = !negated;
            }
            if let Some(&(_, k)) = INTENSIFIERS.iter().find(|(n, _)| n == prev) {
                val *= k;
            }
        }
        if negated {
            val = -val * 0.75;
        }
        if val > 0.0 {
            positive.push(w.clone());
        } else if val < 0.0 {
            negative.push(w.clone());
        }
        total += val;
    }
    let bangs = text.matches('!').count().min(3) as f64;
    total *= 1.0 + 0.1 * bangs;
    // Normalise like VADER: x / sqrt(x² + α).
    let score = total / (total * total + 15.0).sqrt();
    let label = if score >= 0.05 {
        "positive"
    } else if score <= -0.05 {
        "negative"
    } else {
        "neutral"
    };
    Sentiment {
        score,
        label,
        positive,
        negative,
    }
}

fn word_frequencies(text: &str) -> HashMap<String, f64> {
    let mut freq: HashMap<String, f64> = HashMap::new();
    for w in tokenize(text) {
        if w.len() > 2 && !is_stopword(&w) && !w.chars().all(|c| c.is_ascii_digit()) {
            *freq.entry(w).or_default() += 1.0;
        }
    }
    freq
}

/// The `n` most significant words, most frequent first (ties keep their
/// order of first appearance).
pub fn keywords(text: &str, n: usize) -> Vec<String> {
    let freq = word_frequencies(text);
    let mut order: Vec<String> = Vec::new();
    for w in tokenize(text) {
        if freq.contains_key(&w) && !order.contains(&w) {
            order.push(w);
        }
    }
    let mut ranked: Vec<(usize, String)> = order.into_iter().enumerate().collect();
    ranked.sort_by(|(ia, a), (ib, b)| {
        let (fa, fb) = (freq[a], freq[b]);
        fb.partial_cmp(&fa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ia.cmp(ib))
    });
    ranked.into_iter().take(n).map(|(_, w)| w).collect()
}

/// An extractive summary: the `n` highest-scoring sentences, in their
/// original order. Sentences score by the frequency of their content words,
/// with a bonus for leading sentences.
pub fn summarize(text: &str, n: usize) -> String {
    let sents = sentences(text);
    if sents.len() <= n {
        return sents.join(" ");
    }
    let freq = word_frequencies(text);
    let max = freq.values().copied().fold(1.0, f64::max);
    let mut scored: Vec<(usize, f64)> = sents
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let words = tokenize(s);
            let content: f64 = words
                .iter()
                .filter_map(|w| freq.get(w))
                .map(|f| f / max)
                .sum();
            let len = (words.len().max(1) as f64).sqrt();
            let position = if i == 0 { 1.25 } else { 1.0 };
            (i, content / len * position)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let mut keep: Vec<usize> = scored.into_iter().take(n.max(1)).map(|(i, _)| i).collect();
    keep.sort_unstable();
    keep.into_iter()
        .map(|i| sents[i].clone())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Seed vocabulary for common topic labels, so zero-shot classification
/// works offline: a label like "sports" brings its own examples.
const TOPICS: &[(&[&str], &str)] = &[
    (
        &["sports", "sport", "athletics"],
        "game match team goal score player striker coach league season win tournament ball",
    ),
    (
        &["cooking", "food", "recipe", "recipes", "cuisine"],
        "recipe oven bake cook eggs whisk flour pan dish ingredients taste kitchen",
    ),
    (
        &["politics", "political", "government"],
        "election vote government minister policy parliament senate law president campaign party",
    ),
    (
        &["technology", "tech", "programming", "software", "computers"],
        "code software computer program bug compiler server app developer data algorithm",
    ),
    (
        &["business", "finance", "economy", "money"],
        "market stock price revenue profit company sales investors bank economy shares",
    ),
    (
        &["science", "research"],
        "experiment study research scientists theory data physics biology chemistry lab",
    ),
    (
        &["health", "medicine", "medical"],
        "doctor patient hospital disease treatment health medicine symptoms vaccine",
    ),
    (
        &["music", "songs"],
        "song album band music concert guitar singer melody lyrics",
    ),
    (
        &["movies", "film", "films", "cinema", "entertainment"],
        "movie film actor director cinema scene trailer premiere",
    ),
    (
        &["travel", "tourism"],
        "flight hotel trip travel beach vacation tourist passport",
    ),
    (
        &["weather", "climate"],
        "rain sunny storm temperature forecast weather wind snow cloudy",
    ),
];

fn topic_examples(label: &str) -> Option<&'static str> {
    let l = label.to_lowercase();
    TOPICS
        .iter()
        .find(|(names, _)| names.contains(&l.as_str()))
        .map(|(_, words)| *words)
}

/// Zero/few-shot classification by nearest centroid in embedding space.
///
/// Each label comes with example texts (the label itself always counts as
/// one). Returns `(label, confidence)` pairs sorted best first, with
/// confidences summing to 1 (softmax over similarities).
pub fn classify(text: &str, labels: &[(String, Vec<String>)]) -> Vec<(String, f64)> {
    const DIM: usize = 256;
    let x = embed(text, DIM);
    let words: Vec<String> = tokenize(text);
    let mut sims: Vec<(String, f64)> = labels
        .iter()
        .map(|(label, examples)| {
            let mut centroid = embed(label, DIM);
            for e in examples {
                for (c, v) in centroid.iter_mut().zip(embed(e, DIM)) {
                    *c += v;
                }
            }
            let mut sim = cosine(&x, &centroid);
            // Built-in topics: count the text's words found in the topic's
            // vocabulary.
            if examples.is_empty()
                && let Some(vocab) = topic_examples(label)
            {
                let vocab: Vec<&str> = vocab.split_whitespace().collect();
                let hits = words.iter().filter(|w| vocab.contains(&w.as_str())).count();
                sim += 0.4 * hits as f64;
            }
            // Literal mentions of the label are strong evidence.
            for lw in tokenize(label) {
                if words.contains(&lw) {
                    sim += 0.3;
                }
            }
            // Sentiment-shaped labels get help from the sentiment model.
            let s = sentiment(text);
            match label.to_lowercase().as_str() {
                "positive" | "good" | "happy" | "praise" => sim += s.score.max(0.0),
                "negative" | "bad" | "sad" | "complaint" => sim += (-s.score).max(0.0),
                "neutral" => sim += 0.3 * (1.0 - s.score.abs()),
                _ => {}
            }
            (label.clone(), sim)
        })
        .collect();
    let temp = 0.1;
    let max = sims
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = sims.iter().map(|(_, s)| ((s - max) / temp).exp()).sum();
    for (_, s) in &mut sims {
        *s = ((*s - max) / temp).exp() / total;
    }
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sims
}

/// Simple entity extraction: emails, URLs, numbers, hashtags, @mentions,
/// dates (`2024-10-01`, `10/01/2024`) and capitalised names.
pub fn extract(text: &str, kind: &str) -> Vec<String> {
    let raw: Vec<&str> = text.split_whitespace().collect();
    let trim = |w: &str| {
        w.trim_matches(|c: char| ",.;:!?()[]{}\"'".contains(c))
            .to_owned()
    };
    let kind = kind.to_lowercase();
    let kind = kind.trim_end_matches('s');
    let mut out = Vec::new();
    match kind {
        "email" => {
            for w in &raw {
                let w = trim(w);
                if let Some((user, host)) = w.split_once('@')
                    && !user.is_empty()
                    && host.contains('.')
                {
                    out.push(w);
                }
            }
        }
        "url" | "link" => {
            for w in &raw {
                let w = trim(w);
                if w.starts_with("http://") || w.starts_with("https://") || w.starts_with("www.") {
                    out.push(w);
                }
            }
        }
        "number" | "amount" | "price" => {
            for w in &raw {
                let w = trim(w).trim_start_matches(['$', '€', '£']).replace(',', "");
                if !w.is_empty() && w.parse::<f64>().is_ok() {
                    out.push(w);
                }
            }
        }
        "hashtag" | "tag" => {
            for w in &raw {
                let w = trim(w);
                if w.len() > 1 && w.starts_with('#') {
                    out.push(w);
                }
            }
        }
        "mention" | "handle" => {
            for w in &raw {
                let w = trim(w);
                if w.len() > 1 && w.starts_with('@') {
                    out.push(w);
                }
            }
        }
        "date" => {
            for w in &raw {
                let w = trim(w);
                let parts: Vec<&str> = w.split(['-', '/']).collect();
                if parts.len() == 3
                    && parts
                        .iter()
                        .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
                {
                    out.push(w);
                }
            }
        }
        "name" | "person" | "people" | "entitie" | "entity" | "proper_noun" => {
            // Runs of capitalised words. A sentence's first word only counts
            // when it is followed by another capitalised word and isn't a
            // common sentence opener ("Yesterday Ada met …").
            const OPENERS: &[&str] = &[
                "yesterday",
                "today",
                "tomorrow",
                "however",
                "meanwhile",
                "later",
                "finally",
                "first",
                "next",
            ];
            let words: Vec<String> = raw.iter().map(|w| trim(w)).collect();
            let is_cap = |t: &str| {
                t.len() > 1
                    && t.chars().next().is_some_and(char::is_uppercase)
                    && t.chars().skip(1).all(|c| c.is_lowercase() || c == '\'')
            };
            let mut run: Vec<String> = Vec::new();
            let mut sentence_start = true;
            for (i, t) in words.iter().enumerate() {
                let lower = t.to_lowercase();
                let opener = is_stopword(&lower) || OPENERS.contains(&lower.as_str());
                let next_cap = words.get(i + 1).is_some_and(|n| is_cap(n));
                let counts = is_cap(t) && !(sentence_start && (opener || !next_cap));
                if counts {
                    run.push(t.clone());
                } else if !run.is_empty() {
                    out.push(run.join(" "));
                    run.clear();
                }
                sentence_start = raw[i].ends_with(['.', '!', '?']);
                if sentence_start && !run.is_empty() {
                    out.push(run.join(" "));
                    run.clear();
                }
            }
            if !run.is_empty() {
                out.push(run.join(" "));
            }
        }
        _ => {}
    }
    let mut seen = Vec::new();
    out.retain(|x| {
        if seen.contains(x) {
            false
        } else {
            seen.push(x.clone());
            true
        }
    });
    out
}

/// Translates a sentence to Pig Latin (the one language the offline
/// backend speaks fluently).
pub fn pig_latin(text: &str) -> String {
    let mut out = String::new();
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String| {
        if word.is_empty() {
            return;
        }
        let lower = word.to_lowercase();
        let cap = word.chars().next().is_some_and(char::is_uppercase);
        let vowels = "aeiou";
        let split = lower.find(|c: char| vowels.contains(c)).unwrap_or(0);
        let mut pig = if split == 0 {
            format!("{lower}way")
        } else {
            let (head, tail) = lower.split_at(split);
            // Keep "qu" together: "queen" → "eenquay".
            if head.ends_with('q') && tail.starts_with('u') {
                format!("{}{head}uay", &tail[1..])
            } else {
                format!("{tail}{head}ay")
            }
        };
        if cap && let Some(first) = pig.get(..1) {
            pig = first.to_uppercase() + &pig[1..];
        }
        out.push_str(&pig);
        word.clear();
    };
    for c in text.chars() {
        if c.is_alphabetic() {
            word.push(c);
        } else {
            flush(&mut word, &mut out);
            out.push(c);
        }
    }
    flush(&mut word, &mut out);
    out
}

/// Counts syllables in an English word (heuristic).
pub fn syllables(word: &str) -> usize {
    let w: Vec<char> = word
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphabetic())
        .collect();
    if w.is_empty() {
        return 0;
    }
    let vowel = |c: char| "aeiouy".contains(c);
    let mut count = 0;
    let mut prev = false;
    for &c in &w {
        let v = vowel(c);
        if v && !prev {
            count += 1;
        }
        prev = v;
    }
    if w.ends_with(&['e']) && !w.ends_with(&['l', 'e']) && count > 1 {
        count -= 1;
    }
    count.max(1)
}

/// A word-level Markov chain for generating text in the style of a corpus.
#[derive(Debug, Clone)]
pub struct Markov {
    order: usize,
    table: HashMap<Vec<String>, Vec<String>>,
    starts: Vec<Vec<String>>,
}

impl Markov {
    /// Trains a chain of the given order on `corpus`.
    pub fn train(corpus: &str, order: usize) -> Markov {
        let order = order.max(1);
        let mut table: HashMap<Vec<String>, Vec<String>> = HashMap::new();
        let mut starts = Vec::new();
        for sentence in sentences(corpus) {
            let words: Vec<String> = sentence.split_whitespace().map(str::to_owned).collect();
            if words.len() <= order {
                continue;
            }
            starts.push(words[..order].to_vec());
            for win in words.windows(order + 1) {
                table
                    .entry(win[..order].to_vec())
                    .or_default()
                    .push(win[order].clone());
            }
            // An end marker so generation can stop at sentence ends.
            table
                .entry(words[words.len() - order..].to_vec())
                .or_default()
                .push(String::new());
        }
        Markov {
            order,
            table,
            starts,
        }
    }

    /// Number of distinct states.
    pub fn states(&self) -> usize {
        self.table.len()
    }

    /// Generates up to `max_words` words, using `pick(n)` to choose an index
    /// below `n` (so the caller controls randomness).
    pub fn generate(
        &self,
        max_words: usize,
        start: Option<&str>,
        mut pick: impl FnMut(usize) -> usize,
    ) -> String {
        if self.starts.is_empty() {
            return String::new();
        }
        let mut state: Vec<String> = match start {
            Some(s) => {
                let want: Vec<String> = s.split_whitespace().map(str::to_owned).collect();
                self.starts
                    .iter()
                    .chain(self.table.keys())
                    .find(|k| {
                        k.first().map(|w| w.to_lowercase())
                            == want.first().map(|w| w.to_lowercase())
                    })
                    .cloned()
                    .unwrap_or_else(|| self.starts[pick(self.starts.len())].clone())
            }
            None => self.starts[pick(self.starts.len())].clone(),
        };
        let mut out = state.clone();
        while out.len() < max_words {
            let Some(nexts) = self.table.get(&state) else {
                break;
            };
            let next = &nexts[pick(nexts.len())];
            if next.is_empty() {
                if out.len() >= max_words / 2 {
                    break;
                }
                // Hop to a fresh sentence start.
                state = self.starts[pick(self.starts.len())].clone();
                out.extend(state.iter().cloned());
                continue;
            }
            out.push(next.clone());
            state = out[out.len() - self.order..].to_vec();
        }
        out.truncate(max_words);
        out.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_and_splits() {
        assert_eq!(
            tokenize("Hello, World! It's 2024."),
            vec!["hello", "world", "it's", "2024"]
        );
        assert_eq!(
            sentences("One. Two! Three? 3.5 is a number"),
            vec!["One.", "Two!", "Three?", "3.5 is a number"]
        );
    }

    #[test]
    fn sentiment_handles_negation_and_intensity() {
        assert_eq!(sentiment("I love this language").label, "positive");
        assert_eq!(sentiment("this is not good").label, "negative");
        assert_eq!(sentiment("the build is broken and slow").label, "negative");
        assert_eq!(sentiment("the sky is blue").label, "neutral");
        assert!(sentiment("very good").score > sentiment("good").score);
    }

    #[test]
    fn embeddings_rank_similar_text_higher() {
        let a = embed("the cat sat on the mat", 128);
        let b = embed("a cat sitting on a mat", 128);
        let c = embed("quarterly revenue grew by ten percent", 128);
        assert!(cosine(&a, &b) > cosine(&a, &c));
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn summaries_keywords_and_classes() {
        let text = "Rust is a systems language. Rust has ownership and borrowing. \
                    The weather was nice. Ownership makes Rust memory safe.";
        let s = summarize(text, 2);
        assert!(s.contains("Rust"), "{s}");
        assert!(!s.contains("weather"), "{s}");
        assert_eq!(keywords(text, 2), vec!["rust", "ownership"]);
        let labels = vec![
            (
                "sports".to_string(),
                vec![
                    "football match goal".to_string(),
                    "the team won the game".to_string(),
                ],
            ),
            (
                "programming".to_string(),
                vec![
                    "compiler code bug".to_string(),
                    "rust python javascript".to_string(),
                ],
            ),
        ];
        assert_eq!(
            classify("my code has a bug in the compiler", &labels)[0].0,
            "programming"
        );
        assert_eq!(
            classify("what a goal in the final match", &labels)[0].0,
            "sports"
        );
    }

    #[test]
    fn extraction_and_pig_latin() {
        let t = "Mail ada@example.com or see https://chimera.dev on 2024-10-01, cost $1,200 #rust @ferris";
        assert_eq!(extract(t, "emails"), vec!["ada@example.com"]);
        assert_eq!(extract(t, "urls"), vec!["https://chimera.dev"]);
        assert_eq!(extract(t, "dates"), vec!["2024-10-01"]);
        assert_eq!(extract(t, "numbers"), vec!["1200"]);
        assert_eq!(extract(t, "hashtags"), vec!["#rust"]);
        assert_eq!(
            extract("Yesterday Ada Lovelace met Charles Babbage.", "names"),
            vec!["Ada Lovelace", "Charles Babbage"]
        );
        assert_eq!(
            pig_latin("Hello world, eat queens"),
            "Ellohay orldway, eatway eensquay"
        );
        assert_eq!(syllables("banana"), 3);
        assert_eq!(syllables("code"), 1);
    }

    #[test]
    fn markov_generates_from_corpus() {
        let m = Markov::train("the cat sat. the cat ran. the dog sat.", 1);
        let mut i = 0;
        let s = m.generate(6, Some("the"), |n| {
            i += 1;
            i % n
        });
        assert!(s.starts_with("the"), "{s}");
        assert!(m.states() > 3);
    }
}
