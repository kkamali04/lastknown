// scrub.rs — layered secret / personal-data scrubbing for what recall hands back (and, opt-in,
// what it stores).
//
// Three independent layers, each ON by default except `custom` (empty until the user adds terms):
//   secrets  API keys, tokens, private keys, passwords in URLs, .env-style KEY=VALUE
//   pii      emails, phone numbers, street addresses, card numbers (Luhn), SSNs, IP addresses
//   custom   the user's own words / `re:<regex>` entries
//
// Settings live in `~/.recall/settings.json` under one `scrub` object:
//   { "scrub": { "secrets": true, "pii": true, "custom": ["Project Falcon", "re:ACME-\\d{4}"],
//                "store_redacted": false } }
// A missing file or key means the default; a bad value warns on stderr and keeps the default.
//
// OUTPUT time (always, when a layer is on): `scrub_output` runs on every snippet and title that
// `recall query` prints or the daemon returns. INDEX time (`store_redacted`; switched on for a
// profile with nothing indexed yet, see `init_new_install`; off when the key is absent): `scrub_stored`
// runs in `read_markdown_text`, the one reader behind every ingest path, so FTS5, the content hash
// and the embeddings all see the redacted text. Trade-off: a redacted value can no longer be found
// by searching for it, because it is not in the index.
//
// Patterns are ported from the existing scrubbers, not invented: vendor-key shapes and the
// Bearer / assignment guards from an earlier internal redactor; email / SSN / phone / adversarial
// patterns from recall's own `ingest/scrub_turns.py`. Deliberately NOT ported: the bare hex /
// entropy tiers. They eat git SHAs, UUIDs and base64 fixtures, which is exactly what a search
// result must keep. Cost: an unprefixed random token with no key-ish name next to it is not caught.
//
// Replacements are typed placeholders (`[EMAIL]`, `[API_KEY:openai]`), never partial values, and
// are idempotent: no pattern matches a placeholder, so scrubbing twice equals scrubbing once.

use regex::{Captures, Regex};
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub secrets: bool,
    pub pii: bool,
    /// Raw entries: a literal word/phrase, or `re:<regex>`.
    pub custom: Vec<String>,
    pub store_redacted: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self { secrets: true, pii: true, custom: Vec::new(), store_redacted: false }
    }
}

impl Config {
    pub fn load() -> Self {
        Self::load_with_source().0
    }

    /// The settings in force and which rule decided `store_redacted`, from ONE read of
    /// settings.json: two reads could disagree if another command saved the key in between.
    fn load_with_source() -> (Self, &'static str) {
        let text = std::fs::read_to_string(crate::settings::settings_path()).ok();
        Self::resolve(text.as_deref(), REDACT_THIS_RUN.load(std::sync::atomic::Ordering::Relaxed), env_store_redacted())
    }

    /// Pure rule, strongest first: the env pin, a readable key in settings.json (either value),
    /// the database's own record that this index was started redacted (`marker`), then the legacy
    /// default (off). `text` is the file's content; None = no file, or one that cannot be read.
    fn resolve(text: Option<&str>, marker: bool, env: Option<bool>) -> (Self, &'static str) {
        let mut c = text.map(Self::from_json).unwrap_or_default();
        let stated = text.is_some_and(|t| store_redacted_stated(t) == Some(true));
        let source = match (env, stated, marker) {
            (Some(v), _, _) => {
                c.store_redacted = v;
                "env RECALL_SCRUB_STORE_REDACTED"
            }
            (None, true, _) => "settings.json",
            (None, false, true) => {
                c.store_redacted = true;
                "database (this index was started redacted)"
            }
            (None, false, false) => "default (existing index, no stated choice)",
        };
        (c, source)
    }

    /// What `recall status --json` reports under `scrub`, and what a published benchmark should
    /// print: the settings in force and where `store_redacted` came from.
    pub fn status_summary() -> serde_json::Value {
        let (c, source) = Self::load_with_source();
        serde_json::json!({
            "secrets": c.secrets,
            "pii": c.pii,
            "custom_terms": c.custom.len(),
            "store_redacted": c.store_redacted,
            "store_redacted_source": source,
        })
    }

    /// Never fails and never creates anything: missing file = defaults.
    pub fn load_from(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(t) => Self::from_json(&t),
            Err(_) => Self::default(),
        }
    }

    pub fn from_json(text: &str) -> Self {
        let mut c = Self::default();
        let Ok(serde_json::Value::Object(root)) = serde_json::from_str::<serde_json::Value>(text) else {
            return c;
        };
        let Some(serde_json::Value::Object(s)) = root.get("scrub") else { return c };
        for (key, slot) in [("secrets", &mut c.secrets), ("pii", &mut c.pii), ("store_redacted", &mut c.store_redacted)] {
            match s.get(key) {
                None => {}
                Some(serde_json::Value::Bool(b)) => *slot = *b,
                Some(serde_json::Value::String(v)) if matches!(v.to_lowercase().as_str(), "on" | "true") => *slot = true,
                Some(serde_json::Value::String(v)) if matches!(v.to_lowercase().as_str(), "off" | "false") => *slot = false,
                Some(_) => eprintln!("recall: settings.json scrub.{key}: expected true/false; using the default"),
            }
        }
        match s.get("custom") {
            None => {}
            Some(serde_json::Value::Array(a)) => {
                c.custom = a.iter().filter_map(|v| v.as_str()).map(str::to_string).filter(|x| !x.is_empty()).collect()
            }
            Some(_) => eprintln!("recall: settings.json scrub.custom: expected a list of strings; ignoring"),
        }
        c
    }
}

struct Rule {
    re: Lazy,
    kind: Kind,
}

enum Kind {
    /// Fixed placeholder for the whole match.
    Fixed(&'static str),
    /// Keep capture 1 (the NAME / scheme), replace the rest.
    KeepPrefix(&'static str),
    /// Decide in code; `None` keeps the original text.
    Check(fn(&Captures) -> Option<String>),
}

/// A pattern compiled on first USE, and only when its cheap `gate` passes. A CLI query is a fresh
/// process, so compiling all ~35 patterns up front cost ~14 ms on every query; a snippet with no
/// `@` never pays for the email pattern, one with no `sk-` never pays for the OpenAI one.
///
/// Compiled with ASCII-only shorthand classes: Unicode digit/word/space/boundary classes blow the
/// automaton up, and every credential and PII shape here is ASCII.
struct Lazy {
    pat: String,
    /// Literal the text must contain for the pattern to possibly match (auto-derived for vendor
    /// keys from the pattern's leading literal).
    needle: Option<String>,
    gate: Option<fn(&str) -> bool>,
    re: OnceLock<Regex>,
}

impl Lazy {
    fn gate(mut self, g: fn(&str) -> bool) -> Self {
        self.gate = Some(g);
        self
    }

    fn regex(&self) -> &Regex {
        self.re.get_or_init(|| {
            let ascii = self
                .pat
                .replace(r"\d", "[0-9]")
                .replace(r"\s", r"[ \t\r\n]")
                .replace(r"\w", "[A-Za-z0-9_]")
                .replace(r"\b", r"(?-u:\b)");
            Regex::new(&ascii).unwrap_or_else(|e| panic!("scrub pattern {}: {e}", self.pat))
        })
    }

    fn is_match(&self, text: &str) -> bool {
        if let Some(n) = &self.needle {
            if !text.contains(n.as_str()) {
                return false;
            }
        }
        if let Some(g) = self.gate {
            if !g(text) {
                return false;
            }
        }
        self.regex().is_match(text)
    }

    fn replace_all<'t, R: regex::Replacer>(&self, text: &'t str, rep: R) -> Cow<'t, str> {
        self.regex().replace_all(text, rep)
    }
}

/// Leading literal of a pattern, e.g. `\bsk-ant-[A-Za-z]{20,}` -> `sk-ant-`. None when the pattern
/// starts with a group/class, is case-insensitive, or has a top-level alternation.
fn leading_literal(p: &str) -> Option<String> {
    let mut depth = 0i32;
    let mut prev = ' ';
    for c in p.chars() {
        match c {
            '(' if prev != '\\' => depth += 1,
            ')' if prev != '\\' => depth -= 1,
            '|' if depth == 0 => return None,
            _ => {}
        }
        prev = c;
    }
    if p.starts_with("(?i)") {
        return None;
    }
    let p = p.strip_prefix("\\b").unwrap_or(p);
    let p = p.strip_prefix("(?s)").unwrap_or(p);
    let lit: String = p
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ':' | '/'))
        .collect();
    (lit.len() >= 2).then_some(lit)
}

fn rx(p: &str) -> Lazy {
    Lazy { pat: p.to_string(), needle: leading_literal(p), gate: None, re: OnceLock::new() }
}

/// Cheap precondition for the phone pattern: `(ddd)`, `+d`, or `ddd<sep>ddd<sep>dddd`.
fn phone_shape(s: &str) -> bool {
    let b = s.as_bytes();
    let d = |i: usize| b.get(i).is_some_and(u8::is_ascii_digit);
    let sep = |i: usize| b.get(i).is_some_and(|c| matches!(c, b' ' | b'.' | b'-'));
    (0..b.len()).any(|i| {
        (b[i] == b'+' && d(i + 1))
            || (b[i] == b'(' && d(i + 1) && d(i + 2) && d(i + 3) && b.get(i + 4) == Some(&b')'))
            || (d(i) && d(i + 1) && d(i + 2) && sep(i + 3) && d(i + 4) && d(i + 5) && d(i + 6) && sep(i + 7))
    })
}

fn has_digit(s: &str) -> bool {
    s.bytes().any(|b| b.is_ascii_digit())
}

/// ASCII case-insensitive "contains any of".
fn ci_any(s: &str, needles: &[&str]) -> bool {
    let b = s.as_bytes();
    needles.iter().any(|n| {
        let n = n.as_bytes();
        b.len() >= n.len() && b.windows(n.len()).any(|w| w.eq_ignore_ascii_case(n))
    })
}

/// 13+ digits in one run, allowing single space/dash separators (the card pattern's precondition).
fn long_digit_run(s: &str) -> bool {
    // up to two separators (space/dash) between digits, matching the `[ \-]{1,2}` in the patterns
    let (mut run, mut seps) = (0usize, 0u8);
    for b in s.bytes() {
        if b.is_ascii_digit() {
            run += 1;
            seps = 0;
            if run >= 13 {
                return true;
            }
        } else if matches!(b, b' ' | b'-') && seps < 2 && run > 0 {
            seps += 1;
        } else {
            run = 0;
            seps = 0;
        }
    }
    false
}

fn vendor(re: &str, label: &'static str) -> Rule {
    Rule { re: rx(re), kind: Kind::Fixed(label) }
}

fn gated(re: &str, label: &'static str, g: fn(&str) -> bool) -> Rule {
    Rule { re: rx(re).gate(g), kind: Kind::Fixed(label) }
}

fn secret_rules() -> &'static Vec<Rule> {
    static R: OnceLock<Vec<Rule>> = OnceLock::new();
    R.get_or_init(|| {
        vec![
            // Whole PEM private key block, then a stray BEGIN header with no END (truncated chunk).
            vendor(r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", "[PRIVATE_KEY]"),
            // Truncated chunk (BEGIN with no END): the key body that follows is still key material.
            vendor(r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*", "[PRIVATE_KEY]"),
            // scheme://user:password@host — keep the scheme, drop the credentials.
            Rule {
                re: rx(r"(?i)\b([a-z][a-z0-9+.\-]{1,20}://)[^\s:/@]+:[^\s@/]+@").gate(|s| s.contains("://")),
                kind: Kind::KeepPrefix("[URL_CREDENTIALS]@"),
            },
            // Vendor-prefixed keys. Order matters: specific `sk-` flavours before generic.
            Rule { re: rx(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"), kind: Kind::Fixed("[API_KEY:anthropic]") },
            Rule { re: rx(r"\bsk-or-[A-Za-z0-9_\-]{20,}"), kind: Kind::Fixed("[API_KEY:openrouter]") },
            Rule { re: rx(r"\bsk-lf-[A-Za-z0-9_\-]{20,}"), kind: Kind::Fixed("[API_KEY:langfuse]") },
            Rule {
                // `sk-` plus 20+ key chars; a real key always carries a digit, prose like
                // `sk-learn-...-documentation` does not.
                re: rx(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}"),
                kind: Kind::Check(|c| c[0].chars().any(|x| x.is_ascii_digit()).then(|| "[API_KEY:openai]".into())),
            },
            gated(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{10,}", "[API_KEY:stripe]", |s| s.contains("_live_") || s.contains("_test_")),
            gated(r"\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}", "[API_KEY:github]", |s| s.contains("gh") && s.contains('_')),
            vendor(r"\bglpat-[A-Za-z0-9_\-]{16,}", "[API_KEY:gitlab]"),
            vendor(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}", "[API_KEY:slack]"),
            gated(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", "[API_KEY:aws]", |s| s.contains("KIA") || s.contains("SIA")),
            vendor(r"\bAIza[0-9A-Za-z_\-]{35}", "[API_KEY:google]"),
            vendor(r"\bya29\.[A-Za-z0-9_\-]{20,}", "[API_KEY:google_oauth]"),
            vendor(r"\bgsk_[A-Za-z0-9]{20,}", "[API_KEY:groq]"),
            vendor(r"\bxai-[A-Za-z0-9]{20,}", "[API_KEY:xai]"),
            vendor(r"\btvly-[A-Za-z0-9_\-]{16,}", "[API_KEY:tavily]"),
            vendor(r"\bfc-[A-Za-z0-9]{20,}", "[API_KEY:firecrawl]"),
            vendor(r"\bapify_api_[A-Za-z0-9]{20,}", "[API_KEY:apify]"),
            vendor(r"\bhf_[A-Za-z0-9]{30,}", "[API_KEY:huggingface]"),
            vendor(r"\br8_[A-Za-z0-9]{20,}", "[API_KEY:replicate]"),
            vendor(r"\bnpm_[A-Za-z0-9]{30,}", "[API_KEY:npm]"),
            vendor(r"\bsbp_[A-Za-z0-9]{30,}", "[API_KEY:supabase]"),
            vendor(r"\bSG\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}", "[API_KEY:sendgrid]"),
            gated(r"\b[MN][A-Za-z0-9_\-]{23,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{20,}", "[API_KEY:discord]", |s| s.matches('.').count() >= 2),
            vendor(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{4,}", "[JWT]"),
            // `Bearer <tok>`: the scheme word survives. A token must look like one (digit or
            // base64 punctuation, or 24+ solid letters) or "bearer or basic authentication" would
            // lose its next English word.
            Rule {
                re: rx(r"(?i)\b(?:(authorization)[ \t]*[:=][ \t]*)?(bearer|basic)\s+([A-Za-z0-9._~+/=\-]+)").gate(|s| ci_any(s, &["bearer", "basic"])),
                kind: Kind::Check(|c| {
                    // Inside an `Authorization:` header anything after the scheme is the credential.
                    // Bare prose ("basic helix-loop-helix", "basic research.") needs a token shape:
                    // letters+digits and 12+ long, or base64 `=` padding, or 24+ solid letters.
                    let t = &c[3];
                    if c.get(1).is_none() && t.len() < 8 {
                        return None;
                    }
                    let mixed = t.chars().any(|x| x.is_ascii_digit()) && t.chars().any(|x| x.is_ascii_alphabetic());
                    let solid = t.len() >= 24 && t.chars().all(|x| x.is_ascii_alphanumeric());
                    let tokenish = c.get(1).is_some() || (t.len() >= 12 && mixed) || t.ends_with('=') || solid;
                    tokenish.then(|| format!("{}[TOKEN]", &c[0][..c[0].len() - t.len()]))
                }),
            },
            // `password = "hunter2xyz"`, `"api_key": "abc123..."`, `pw = 'two words'` — quoted
            // literal next to a secret-ish name; the value runs to the matching quote. Unquoted
            // code (`token = self.next()`) is not touched.
            Rule {
                re: rx(r#"(?i)\b([\w\-]*(?:api[-_]?key|secret|token|passw(?:or)?d|credential|private[-_]?key)[\w\-]*["']?[ \t]*[:=][ \t]*)(?:"([^"\r\n]{8,})"|'([^'\r\n]{8,})')"#).gate(|s| ci_any(s, &["key", "secret", "token", "passw", "credential"])),
                kind: Kind::Check(quoted_secret),
            },
            // .env style: `OPENAI_API_KEY=...`, `export DB_PASSWORD=...` (UPPER_CASE name that
            // names a secret; may sit mid-line, snippets are not line-aligned). Paths, ${VARS},
            // placeholders and pure numbers (`MAX_TOKENS=4096`) are left alone.
            Rule {
                re: rx(r#"\b((?:export[ \t]+)?[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE)[A-Z0-9_]*[ \t]*=[ \t]*)(["']?)([^\s"'#]{4,})"#).gate(|s| s.contains('=')),
                kind: Kind::Check(|c| {
                    let v = &c[3];
                    (!not_a_secret_value(v)).then(|| format!("{}{}[SECRET]", &c[1], &c[2]))
                }),
            },
        ]
    })
}

/// Values that are references or placeholders, not secrets. Whole-value forms only: a substring
/// such as `xxx` or a leading `$` inside a longer string does not exempt it.
fn not_a_secret_value(v: &str) -> bool {
    let l = v.to_lowercase();
    let first = v.chars().next().unwrap_or('x');
    // `$VAR`, `${VAR}`, `%VAR%`, `<your-key>`, `{{ var }}`, `[REDACTED]`, `(...)`
    let reference = v.starts_with("${") && v.ends_with('}')
        || v.starts_with('$') && v.len() > 1 && v[1..].chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        || v.starts_with('%') && v.ends_with('%')
        || v.starts_with('<') && v.ends_with('>')
        || v.starts_with("{{")
        || v.starts_with('[') && v.ends_with(']');
    let path = v.starts_with(['/', '.', '~']) || (v.starts_with("http") && !v.contains('@'));
    // A short pure number is a limit (`MAX_TOKENS=4096`); 8+ digits can be a real PIN-style secret.
    let small_number = v.len() < 8 && v.chars().all(|c| c.is_ascii_digit());
    let placeholder = matches!(l.as_str(), "changeme" | "placeholder" | "example" | "redacted" | "null" | "none" | "true" | "false")
        || ["your", "example", "placeholder", "dummy", "sample"].iter().any(|w| l.strip_prefix(w).is_some_and(|r| r.starts_with(['_', '-'])))
        || v.chars().all(|c| c == first);
    reference || path || small_number || placeholder
}

fn quoted_secret(c: &Captures) -> Option<String> {
    let (v, q) = match c.get(2) {
        Some(m) => (m.as_str(), '"'),
        None => (c.get(3)?.as_str(), '\''),
    };
    // `token_type = "identifier"`: a lone lowercase word under a non-password name is a code
    // value, not a credential. A password-ish name always redacts.
    let name = c[1].to_lowercase();
    let word = v.chars().all(|ch| ch.is_ascii_lowercase() || ch == '_' || ch == '-');
    if word && !name.contains("pass") && !name.contains("secret_key") && !name.contains("api_key") && !name.contains("apikey") {
        return None;
    }
    (!not_a_secret_value(v)).then(|| format!("{}{q}[SECRET]{q}", &c[1]))
}

fn pii_rules() -> &'static Vec<Rule> {
    static R: OnceLock<Vec<Rule>> = OnceLock::new();
    R.get_or_init(|| {
        vec![
            Rule {
                re: rx(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.([A-Za-z]{2,})\b").gate(|s| s.contains('@')),
                // `icon@2x.png`, `name@1.svg` are asset names, not mailboxes.
                kind: Kind::Check(|c| {
                    let tld = c[1].to_lowercase();
                    let asset = ["png", "jpg", "jpeg", "gif", "svg", "webp", "css", "js", "ts", "json", "md"];
                    (!asset.contains(&tld.as_str())).then(|| "[EMAIL]".into())
                }),
            },
            Rule { re: rx(r"\b(\d{3})-(\d{2})-(\d{4})\b").gate(|s| s.contains('-') && has_digit(s)), kind: Kind::Check(ssn) },
            // `card=4111...`, `"pan": "4111 ..."`: a payment field name makes the number a card even
            // where a bare `=`-glued number is treated as an id.
            Rule {
                re: rx(r#"(?i)\b((?:card|pan|cc|credit)[_ \-]?(?:num(?:ber)?|no)?["']?[ \t]*[:=][ \t]*["']?)((?:\d{4}[ \-]{1,2}\d{4}[ \-]{1,2}\d{4}[ \-]{1,2}\d{1,7}|\d{4}[ \-]{1,2}\d{6}[ \-]{1,2}\d{4,5}|\d{13,19}))\b"#).gate(long_digit_run),
                kind: Kind::Check(card_field),
            },
            Rule { re: rx(r"(^|[^A-Za-z0-9/_=#.\-])((?:\d{4}[ \-]{1,2}\d{4}[ \-]{1,2}\d{4}[ \-]{1,2}\d{1,7}|\d{4}[ \-]{1,2}\d{6}[ \-]{1,2}\d{4,5}|\d{13,19}))\b").gate(long_digit_run), kind: Kind::Check(card) },
            // 3-3-4 WITH separators or parentheses, optional +CC; bare 10-digit runs (timestamps,
            // ids) are not phones. International: leading `+`.
            Rule {
                re: rx(r"(?:\+\d{1,3}[ .\-]?)?\(\d{3}\)[ .\-]?\d{3}[ .\-]?\d{4}\b|(?:\+\d{1,3}[ .\-])?\b\d{3}[ .\-]\d{3}[ .\-]\d{4}\b|\+\d{1,3}[ .\-]?\d{2,4}(?:[ .\-]\d{2,4}){2,3}\b|\+\d{10,14}\b").gate(phone_shape),
                kind: Kind::Fixed("[PHONE]"),
            },
            // Unambiguous street suffixes only: "Court", "Way", "Place" read as law reports and prose
            // ("1975 International Court of Justice"). Year-like house numbers are skipped.
            Rule {
                re: rx(r"\b(\d{1,6})\s+(?:[A-Z][a-z]+\.?\s+){1,4}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Terrace|Highway|Hwy)\b\.?(?:,?\s+(?:Apt|Suite|Ste|Unit|#)\.?\s*[\w\-]+)?")
                    .gate(|s| has_digit(s) && ci_any(s, &[" st", "street", " ave", "avenue", " rd", "road", "blvd", "boulevard", " dr", "drive", " ln", "lane", "terrace", "hwy", "highway"])),
                kind: Kind::Check(|c| {
                    let n: u32 = c[1].parse().ok()?;
                    (!(1800..=2099).contains(&n)).then(|| "[ADDRESS]".into())
                }),
            },
            Rule { re: rx(r"(?i)(?:\b(ec|version|ver|rev|release|build)[ \t]*)?\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b").gate(|s| has_digit(s) && s.matches('.').count() >= 3), kind: Kind::Check(ipv4) },
        ]
    })
}

/// `000-00-0000`, area 000 / 666 / 9xx, group 00 and serial 0000 are never issued: placeholders and
/// test data, not people.
fn ssn(c: &Captures) -> Option<String> {
    let area: u32 = c[1].parse().ok()?;
    (area != 0 && area != 666 && area < 900 && &c[2] != "00" && &c[3] != "0000").then(|| "[SSN]".into())
}

/// 13-19 digits, optionally grouped by spaces/dashes. Needs a real card prefix AND the Luhn check;
/// a long number without both (a millisecond timestamp, an id) is left alone.
fn card(c: &Captures) -> Option<String> {
    luhn_card(&c[2]).then(|| format!("{}[CARD]", &c[1]))
}

fn card_field(c: &Captures) -> Option<String> {
    luhn_card(&c[2]).then(|| format!("{}[CARD]", &c[1]))
}

fn luhn_card(s: &str) -> bool {
    let digits: Vec<u32> = s.chars().filter_map(|x| x.to_digit(10)).collect();
    let n = digits.len();
    if !(13..=19).contains(&n) || digits.windows(2).all(|w| w[0] == w[1]) {
        return false;
    }
    let p2 = digits[0] * 10 + digits[1];
    let prefix_ok = match digits[0] {
        4 => matches!(n, 13 | 16 | 19),
        5 => (51..=55).contains(&p2) && n == 16,
        2 => (22..=27).contains(&p2) && n == 16,
        3 => (matches!(p2, 34 | 37) && n == 15) || (p2 == 35 && n >= 16) || (matches!(p2, 30 | 36 | 38 | 39) && n == 14),
        6 => (p2 == 60 || p2 == 65 || p2 == 64) && (16..=19).contains(&n),
        _ => false,
    };
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| if i % 2 == 1 { let x = d * 2; if x > 9 { x - 9 } else { x } } else { d })
        .sum();
    (prefix_ok && sum % 10 == 0)
}

/// Valid dotted quad, excluding loopback / unspecified / broadcast (not personal) and `v1.2.3.4`
/// style versions (a leading `v` or word char glued to the quad is checked via the caller's
/// lookbehind-free guard: the pattern's own `\b` already rejects `a1.2.3.4`).
fn ipv4(c: &Captures) -> Option<String> {
    // "EC 1.14.16.2" (enzyme), "version 1.2.3.4": a numbered thing, not an address.
    let o: Vec<u32> = (2..=5).filter_map(|i| c[i].parse().ok()).collect();
    if let Some(w) = c.get(1) {
        if w.as_str().eq_ignore_ascii_case("ec") || o.first().is_some_and(|&a| a < 100) {
            return None;
        }
    }
    if o.len() != 4 || o.iter().any(|&x| x > 255) || (2..=5).any(|i| c[i].len() > 1 && c[i].starts_with('0')) {
        return None;
    }
    if o[0] == 127 || o == [0, 0, 0, 0] || o == [255, 255, 255, 255] {
        return None;
    }
    Some("[IP]".into())
}

/// Every valid entry joined into ONE alternation, so a term like `CUSTOM` cannot rewrite the
/// `[CUSTOM]` an earlier term just produced.
pub(crate) fn compile_custom(entries: &[String]) -> Option<Regex> {
    let pats: Vec<String> = entries
        .iter()
        .enumerate()
        .filter_map(|(idx, e)| {
            let pat = match e.strip_prefix("re:") {
                Some(p) => p.to_string(),
                None => {
                    let edge = |c: Option<char>| if c.is_some_and(|c| c.is_alphanumeric() || c == '_') { r"\b" } else { "" };
                    format!("(?i){}{}{}", edge(e.chars().next()), regex::escape(e), edge(e.chars().last()))
                }
            };
            match Regex::new(&pat) {
                Ok(_) => Some(format!("(?:{pat})")),
                // Loud, not silent: a custom term that is not being applied must be visible.
                // The entry itself is not echoed: it may be the sensitive term.
                Err(_) => {
                    eprintln!("recall: scrub.custom entry #{} is not a valid regex; NOT applied", idx + 1);
                    None
                }
            }
        })
        .collect();
    if pats.is_empty() {
        return None;
    }
    Regex::new(&pats.join("|")).ok()
}

fn apply<'a>(text: Cow<'a, str>, rules: &[Rule]) -> Cow<'a, str> {
    let mut cur = text;
    for r in rules {
        if !r.re.is_match(&cur) {
            continue;
        }
        let replaced = match &r.kind {
            Kind::Fixed(p) => r.re.replace_all(&cur, *p).into_owned(),
            Kind::KeepPrefix(p) => r.re.replace_all(&cur, |c: &Captures| format!("{}{p}", &c[1])).into_owned(),
            Kind::Check(f) => r.re.replace_all(&cur, |c: &Captures| f(c).unwrap_or_else(|| c[0].to_string())).into_owned(),
        };
        cur = Cow::Owned(replaced);
    }
    cur
}

// ── Hidden characters ────────────────────────────────────────────────────────────────────────
//
// A credential with a zero-width space inside it, a bidi override, a NUL or an ANSI colour code
// between its characters reads as the key to a person and to an LLM, and matches no pattern. So the
// rules run on a copy with those characters removed. Text that matches nothing is returned byte for
// byte as it came in (emoji ZWJ sequences and Persian/Indic ZWNJ stay intact); text that DID match is
// returned from the cleaned copy, so the hidden characters are gone from it.
//
// Ported from ai-memory (MIT, Copyright (c) 2026 Fabio Akita):
// crates/ai-memory-core/src/sanitize.rs `is_default_ignorable` and `ESCAPE_SEQUENCES`.

/// ANSI/VT escape sequences, removed whole so `ESC[31m` does not leave `[31m` behind as text.
const ESCAPE_SEQUENCES: &str = r"\x1b\[[0-?]*[ -/]*[@-~]|\x{9b}[0-?]*[ -/]*[@-~]|\x{9d}[^\x07\x{9c}]*(?:\x07|\x{9c})|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]";

fn escape_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(ESCAPE_SEQUENCES).expect("escape-sequence pattern"))
}

/// Unicode `Default_Ignorable_Code_Point`: characters that render as nothing (zero-width space and
/// joiners, soft hyphen, bidi marks and overrides, variation selectors, tag characters, BOM, ...).
const fn is_default_ignorable(c: char) -> bool {
    matches!(c,
        '\u{ad}' | '\u{34f}' | '\u{61c}' | '\u{115f}' | '\u{1160}' | '\u{17b4}' | '\u{17b5}'
        | '\u{180b}'..='\u{180f}' | '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}'
        | '\u{2060}'..='\u{206f}' | '\u{3164}' | '\u{fe00}'..='\u{fe0f}' | '\u{feff}'
        | '\u{ffa0}' | '\u{fff0}'..='\u{fff8}' | '\u{1bca0}'..='\u{1bca3}'
        | '\u{1d173}'..='\u{1d17a}' | '\u{e0000}'..='\u{e0fff}')
}

/// Invisible or control: everything but the whitespace a text file legitimately holds.
fn is_hidden(c: char) -> bool {
    (c.is_control() && !matches!(c, '\t' | '\n' | '\r')) || is_default_ignorable(c)
}

/// Byte ranges of the maximal runs of hidden characters and whole ANSI sequences in `text`.
fn hidden_runs(text: &str) -> Vec<(usize, usize)> {
    if !text.chars().any(is_hidden) {
        return Vec::new();
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let escapes: Vec<(usize, usize)> = if text.contains(['\x1b', '\u{9b}', '\u{9d}']) {
        escape_re().find_iter(text).map(|m| (m.start(), m.end())).collect()
    } else {
        Vec::new()
    };
    let mut esc = escapes.into_iter().peekable();
    let mut it = text.char_indices().peekable();
    while let Some(&(i, c)) = it.peek() {
        let end = if esc.peek().is_some_and(|&(st, _)| st == i) {
            esc.next().map(|(_, e)| e)
        } else if is_hidden(c) {
            Some(i + c.len_utf8())
        } else {
            None
        };
        match end {
            Some(e) => {
                match runs.last_mut() {
                    Some(last) if last.1 == i => last.1 = e,
                    _ => runs.push((i, e)),
                }
                while it.peek().is_some_and(|&(j, _)| j < e) {
                    it.next();
                }
            }
            None => {
                it.next();
            }
        }
    }
    // A joiner between two non-ASCII characters (emoji ZWJ sequences, Persian/Indic ZWNJ) cannot be
    // splitting an ASCII credential, so it is never touched.
    runs.retain(|&(st, en)| {
        let (prev, next) = (text[..st].chars().next_back(), text[en..].chars().next());
        !matches!((prev, next), (Some(a), Some(b)) if !a.is_ascii() && !b.is_ascii())
    });
    runs
}

/// `text` with run `i` replaced by a space where `spaced[i]`, else removed.
fn render_runs(text: &str, runs: &[(usize, usize)], spaced: &[bool]) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for (k, &(st, en)) in runs.iter().enumerate() {
        out.push_str(&text[last..st]);
        if spaced[k] {
            out.push(' ');
        }
        last = en;
    }
    out.push_str(&text[last..]);
    out
}

/// A compiled scrubber for one `Config`. Build once per command / per daemon request.
pub struct Scrubber {
    secrets: bool,
    pii: bool,
    custom: Option<Regex>,
}

impl Scrubber {
    pub fn new(cfg: &Config) -> Self {
        Self { secrets: cfg.secrets, pii: cfg.pii, custom: compile_custom(&cfg.custom) }
    }

    pub fn is_noop(&self) -> bool {
        !self.secrets && !self.pii && self.custom.is_none()
    }

    /// Custom terms run on the text BETWEEN typed placeholders only, so a term like `EMAIL` never
    /// rewrites `[EMAIL]` (the daemon scrubs, then the CLI scrubs again).
    fn apply_custom(&self, text: &str) -> String {
        let ph = placeholder_re();
        let (mut out, mut last) = (String::with_capacity(text.len()), 0);
        let re = self.custom.as_ref().expect("checked by caller");
        let seg = |s: &str, out: &mut String| out.push_str(&re.replace_all(s, "[CUSTOM]"));
        for m in ph.find_iter(text) {
            seg(&text[last..m.start()], &mut out);
            out.push_str(m.as_str());
            last = m.end();
        }
        seg(&text[last..], &mut out);
        out
    }

    pub fn scrub<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if self.is_noop() || hidden_runs(text).is_empty() {
            return self.scrub_clean(text);
        }
        // Repeat until nothing more is found, so a second credential that needs a different
        // reading than the first is still caught (the placeholders from pass 1 do not re-match).
        let mut cur = text.to_string();
        for _ in 0..3 {
            let next = self.scrub_hidden_pass(&cur);
            if next == cur {
                break;
            }
            cur = next;
        }
        // Nothing matched: hand back the original, hidden characters and all.
        if cur == text { Cow::Borrowed(text) } else { Cow::Owned(cur) }
    }

    /// One pass over text that hides characters. Three readings of the hidden runs; the one that
    /// redacts the most wins, and the original comes back if none redacts anything.
    ///   raw     hidden characters left where they are (a NUL can be the word boundary a key needs)
    ///   joined  all removed (`sk-<ZWSP>proj-...` is one key)
    ///   spaced  all replaced by a space
    /// "Most" = more placeholders, then fewer ASCII word characters left in the output (a key
    /// half-redacted leaves its tail behind, so the full redaction scores lower), then the earlier
    /// reading above, so hidden characters survive in text that only needed an unrelated redaction.
    ///
    /// Cost: three scrubs of the text per pass, so about 3x a plain scrub for text with hidden
    /// characters; text without any pays one scan.
    ///
    /// Known limits (accidental leaks are the goal, not an adversary who already holds the key): one
    /// text whose credentials need different readings, or one credential with both a boundary-
    /// supplying and a key-splitting hidden character, can keep part of a key.
    fn scrub_hidden_pass(&self, text: &str) -> String {
        let runs = hidden_runs(text);
        if runs.is_empty() {
            return text.to_string();
        }
        let score = |out: &str| {
            (
                placeholder_re().find_iter(out).count(),
                std::cmp::Reverse(out.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_').count()),
            )
        };
        let mut winner: Option<(_, String)> = None;
        for reading in [
            text.to_string(),
            render_runs(text, &runs, &vec![false; runs.len()]),
            render_runs(text, &runs, &vec![true; runs.len()]),
        ] {
            let out = self.scrub_clean(&reading);
            if out == reading {
                continue;
            }
            let sc = score(&out);
            if winner.as_ref().is_none_or(|(w, _)| sc > *w) {
                winner = Some((sc, out.into_owned()));
            }
        }
        winner.map_or_else(|| text.to_string(), |(_, out)| out)
    }

    fn scrub_clean<'a>(&self, text: &'a str) -> Cow<'a, str> {
        let mut cur = Cow::Borrowed(text);
        // Secrets first: a `postgres://user:pw@host` must lose its credentials before the email
        // pattern sees `pw@host`.
        if self.secrets {
            cur = apply(cur, secret_rules());
        }
        if self.pii {
            cur = apply(cur, pii_rules());
        }
        if self.custom.is_some() {
            cur = Cow::Owned(self.apply_custom(&cur));
        }
        cur
    }
}

/// `RECALL_SCRUB_STORE_REDACTED=0|1` overrides `settings.json`. Benchmark harnesses set it to `0` so
/// published numbers are measured on unredacted text and stay comparable. Unparseable = ignored
/// with a warning (never silently the other value).
pub fn env_store_redacted() -> Option<bool> {
    let v = std::env::var("RECALL_SCRUB_STORE_REDACTED").ok()?;
    match v.trim().to_lowercase().as_str() {
        "1" | "true" | "on" => Some(true),
        "0" | "false" | "off" => Some(false),
        other => {
            eprintln!("recall: RECALL_SCRUB_STORE_REDACTED={other:?} is not 0/1; ignored");
            None
        }
    }
}

/// Some(true) when `settings.json` itself states a readable `scrub.store_redacted` (either value:
/// an explicit off is a choice too), Some(false) when the file is absent or is a JSON object with
/// no such value, None when the file is there but is not a readable JSON object.
fn store_redacted_is_set(settings: &std::path::Path) -> Option<bool> {
    match std::fs::read_to_string(settings) {
        Ok(t) => store_redacted_stated(&t),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Some(false),
        Err(_) => None,
    }
}

/// [`store_redacted_is_set`] on the file's content.
fn store_redacted_stated(text: &str) -> Option<bool> {
    match serde_json::from_str::<serde_json::Value>(text) {
        // the same values `Config::from_json` accepts; anything else there falls back to the default
        Ok(v @ serde_json::Value::Object(_)) => Some(match v.pointer("/scrub/store_redacted") {
            Some(serde_json::Value::Bool(_)) => true,
            Some(serde_json::Value::String(s)) => matches!(s.to_lowercase().as_str(), "on" | "true" | "off" | "false"),
            _ => false,
        }),
        _ => None,
    }
}

/// Set for the rest of this process when the store it opened is one that redacts unless told
/// otherwise: nothing indexed and no stated choice (a new profile), or the database's own marker
/// that its index was started redacted. `settings.json` is then only a copy of the choice: failing
/// to write it, or losing it later, does not change what gets stored. An env pin or a readable
/// `scrub.store_redacted` in the file still wins (see `Config::resolve`).
/// Never cleared: it assumes one profile per process, which holds for every command.
static REDACT_THIS_RUN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// For the store: the database it just opened carries the "started redacted" marker.
pub fn redact_this_run() {
    REDACT_THIS_RUN.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// True once the redaction policy is decided for this profile: an env pin, a `settings.json` that
/// states `scrub.store_redacted`, a database marked as started redacted, or a database that
/// already holds documents (an existing install keeps its current off value). False on a profile
/// with nothing indexed and no stated choice, whatever else created it (`settings set vault`,
/// `recall ui`), where the next writable open switches redaction on; `recall stored-form` must not
/// answer then, or it would describe unredacted storage that indexing will not do. A database that
/// cannot be read counts as nothing indexed. Read-only: the open applies the marker to this
/// process (see `Store::open_readonly`) and writes nothing.
pub fn policy_initialized() -> bool {
    env_store_redacted().is_some()
        || store_redacted_is_set(&crate::settings::settings_path()) == Some(true)
        || crate::store::Store::open_readonly().is_ok_and(|s| s.started_redacted() || matches!(s.has_documents(), Ok(true)))
}

/// First-run default: a profile with nothing indexed yet stores redacted text, so the index never
/// holds secrets or PII. `Store::open` calls this when the database has no documents, so it does
/// not matter which command created the profile or whether `settings.json` already exists with
/// other keys. A profile that already holds documents never reaches this and keeps what it has.
///
/// Fail closed: unless the env pin or a readable `scrub.store_redacted` states a choice, this
/// process redacts from here on, and `Store::open` records that in the database, which is what
/// later processes follow. Saving the key to `settings.json` is a convenience copy; if that fails
/// the run still redacts and says so.
pub fn init_new_install() {
    // An explicit env pin means the caller already decided; do not write a competing default.
    if env_store_redacted().is_some() {
        return;
    }
    let settings = crate::settings::settings_path();
    let Some(saved) = init_new_install_in(settings.parent().unwrap_or(std::path::Path::new("."))) else { return };
    REDACT_THIS_RUN.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Err(e) = saved {
        eprintln!(
            "recall: WARNING: could not save scrub.store_redacted=on to {} ({e:#}). This run still stores redacted text, and so will later ones: the choice is recorded in the database.",
            settings.display()
        );
    }
}

/// None when `dir/settings.json` already states a readable `scrub.store_redacted`: the user chose,
/// nothing to do. Otherwise Some: the profile must redact, and the result says whether the key was
/// saved (added to the file, every other key kept). A file that is not a JSON object is not
/// rewritten; that is an Err.
pub fn init_new_install_in(dir: &std::path::Path) -> Option<anyhow::Result<()>> {
    let settings = dir.join("settings.json");
    match store_redacted_is_set(&settings) {
        Some(true) => None,
        Some(false) => Some(crate::settings::Settings::set_in(&settings, "scrub.store_redacted", "on").map(|_| ())),
        None => Some(Err(anyhow::anyhow!("it is not a readable JSON object"))),
    }
}

/// Where a redaction sits in the ORIGINAL file: what kind, and on which line. Never the value.
/// This is the pointer an agent follows to open the real file (if it is allowed to) when a hit
/// shows `[EMAIL]` and the index never held the address.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Redaction {
    pub kind: String,
    /// 1-based line number in the file on disk.
    pub line: usize,
}

const MAX_POINTERS: usize = 50;
const MAX_POINTER_FILE_BYTES: u64 = 512 * 1024; // same ceiling as ingest

fn placeholder_re() -> &'static Regex {
    static PH: OnceLock<Regex> = OnceLock::new();
    PH.get_or_init(|| Regex::new(r"\[[A-Z_]+(?::[a-z_0-9]+)?\]").expect("placeholder pattern"))
}

impl Scrubber {
    /// One entry per placeholder the scrub would write, with its 1-based line. Scrubs line by
    /// line, so a PEM block is flagged on its BEGIN line only. Capped at [`MAX_POINTERS`].
    pub fn redactions(&self, text: &str) -> Vec<Redaction> {
        let mut out = Vec::new();
        if self.is_noop() {
            return out;
        }
        for (i, line) in text.lines().enumerate() {
            let scrubbed = self.scrub(line);
            if scrubbed == line {
                continue;
            }
            // Placeholders in the output that were not already literally in the input.
            let mut already: Vec<&str> = placeholder_re().find_iter(line).map(|m| m.as_str()).collect();
            let before = out.len();
            for m in placeholder_re().find_iter(&scrubbed) {
                if let Some(k) = already.iter().position(|a| *a == m.as_str()) {
                    already.swap_remove(k);
                    continue;
                }
                out.push(Redaction { kind: m.as_str().trim_matches(['[', ']']).to_string(), line: i + 1 });
                if out.len() >= MAX_POINTERS {
                    return out;
                }
            }
            // The line changed but every placeholder was already literally present (e.g. the
            // value contained "[SECRET]"): still point at it.
            if out.len() == before {
                out.push(Redaction { kind: "REDACTED".into(), line: i + 1 });
                if out.len() >= MAX_POINTERS {
                    return out;
                }
            }
        }
        out
    }
}

/// Pointers are computed for the first this-many hits of a query only, so a `-k 500` query cannot
/// turn into 250 MiB of file reads.
pub const MAX_POINTER_HITS: usize = 50;

/// Pointers for one hit: reads the document's file under `root` and reports where redactions are.
/// Empty for ad-hoc saves (no path), paths that escape the root, missing or oversized files.
pub fn pointers(sc: &Scrubber, root: &std::path::Path, rel: &str) -> Vec<Redaction> {
    use std::path::Component;
    if sc.is_noop() || rel.is_empty() {
        return Vec::new();
    }
    let rel_path = std::path::Path::new(rel);
    if rel_path.is_absolute() || rel_path.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Vec::new();
    }
    let Some(f) = open_confined(root, rel_path) else { return Vec::new() };
    use std::io::Read;
    // Bounded even if the file grows after the open.
    let mut buf = Vec::new();
    if f.take(MAX_POINTER_FILE_BYTES + 1).read_to_end(&mut buf).is_err() || buf.len() as u64 > MAX_POINTER_FILE_BYTES {
        return Vec::new();
    }
    sc.redactions(&String::from_utf8_lossy(&buf))
}

/// The real path of an OPEN file, as the kernel reports it for that descriptor.
#[cfg(target_os = "linux")]
fn handle_path(f: &std::fs::File) -> Option<PathBuf> {
    use std::os::fd::AsRawFd;
    std::fs::read_link(format!("/proc/self/fd/{}", f.as_raw_fd())).ok()
}

#[cfg(target_os = "macos")]
fn handle_path(f: &std::fs::File) -> Option<PathBuf> {
    use std::os::fd::AsRawFd;
    use std::os::unix::ffi::OsStrExt;
    let mut buf = [0u8; libc::PATH_MAX as usize];
    // SAFETY: F_GETPATH writes a NUL-terminated path of at most PATH_MAX bytes into `buf`.
    if unsafe { libc::fcntl(f.as_raw_fd(), libc::F_GETPATH, buf.as_mut_ptr()) } == -1 {
        return None;
    }
    let n = buf.iter().position(|&b| b == 0)?;
    Some(PathBuf::from(std::ffi::OsStr::from_bytes(&buf[..n])))
}

/// Other Unixes: no handle-to-path primitive wired, so refuse (pointers are best-effort).
#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
fn handle_path(_f: &std::fs::File) -> Option<PathBuf> {
    None
}

/// Open `root/rel` and only then prove the OPEN HANDLE is a regular file inside `root`. Checking a
/// path and opening it later leaves a window to swap a directory for a link; here the verdict is
/// taken from the handle, and every read goes through that handle.
///   * Windows: `GetFinalPathNameByHandleW` on the handle gives the real path it points at.
///   * Unix: the file is opened `O_NOFOLLOW | O_NONBLOCK` (a FIFO opens instantly and is then
///     refused as a non-regular file instead of blocking the query), and the descriptor's own path
///     (`/proc/self/fd/N`, `F_GETPATH`) must lie inside the root.
fn open_confined(root: &std::path::Path, rel: &std::path::Path) -> Option<std::fs::File> {
    let real_root = std::fs::canonicalize(root).ok()?;
    let file = root.join(rel);
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        let f = std::fs::File::open(&file).ok()?;
        let meta = f.metadata().ok()?;
        if !meta.is_file() || meta.len() > MAX_POINTER_FILE_BYTES {
            return None;
        }
        let mut buf = vec![0u16; 4096];
        // SAFETY: the handle is valid for the life of `f`; `buf` is writable for `buf.len()` u16.
        let n = unsafe {
            windows_sys::Win32::Storage::FileSystem::GetFinalPathNameByHandleW(
                f.as_raw_handle() as _,
                buf.as_mut_ptr(),
                buf.len() as u32,
                0,
            )
        } as usize;
        if n == 0 || n >= buf.len() {
            return None;
        }
        let real = String::from_utf16(&buf[..n]).ok()?.to_lowercase();
        let root_s = real_root.to_string_lossy().to_lowercase();
        let root_s = root_s.trim_end_matches('\\');
        (real.starts_with(root_s) && real[root_s.len()..].starts_with('\\')).then_some(f)
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let real = std::fs::canonicalize(&file).ok()?;
        if !real.starts_with(&real_root) {
            return None;
        }
        let f = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&real)
            .ok()?;
        let meta = f.metadata().ok()?;
        if !meta.is_file() || meta.len() > MAX_POINTER_FILE_BYTES {
            return None;
        }
        // The verdict comes from the HANDLE, never from re-resolving the path (an attacker who can
        // flip a directory link between two path lookups defeats any path-based comparison).
        let at = handle_path(&f)?;
        // Strict descendant, and never an unlinked file: Linux appends " (deleted)" to the path of
        // an unlinked inode, which can collide with a directory of that name.
        (at != real_root && at.starts_with(&real_root) && !at.to_string_lossy().ends_with(" (deleted)")).then_some(f)
    }
}

/// Compact human form: `EMAIL@L12, API_KEY:openai@L30`.
pub fn pointer_line(rs: &[Redaction]) -> String {
    rs.iter().map(|r| format!("{}@L{}", r.kind, r.line)).collect::<Vec<_>>().join(", ")
}

/// Output-time scrub with the user's current settings. Returns the input untouched (no allocation)
/// when every layer is off or nothing matched.
pub fn scrub_output(s: &Scrubber, text: &str) -> String {
    s.scrub(text).into_owned()
}

/// Index-time scrub: only when the user opted into `store_redacted`. Loads settings once per
/// process.
pub fn scrub_stored(text: String) -> String {
    static S: OnceLock<Option<Scrubber>> = OnceLock::new();
    let s = S.get_or_init(|| {
        let cfg = Config::load();
        // A profile that already holds documents follows the file; if the file cannot be read, off
        // is only the fallback, not a choice, so say it every time rather than store raw quietly.
        if !cfg.store_redacted && env_store_redacted().is_none() && store_redacted_is_set(&crate::settings::settings_path()).is_none() {
            eprintln!("recall: WARNING: settings.json is not a readable JSON object, so scrub.store_redacted counts as off: text stored by this run is NOT redacted. Fix the file, then run `recall settings set scrub.store_redacted on`.");
        }
        cfg.store_redacted.then(|| Scrubber::new(&cfg))
    });
    match s {
        Some(s) if !s.is_noop() => s.scrub(&text).into_owned(),
        _ => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Scrubber {
        Scrubber::new(&Config { custom: vec!["Project Falcon".into(), "re:ACME-\\d{4}".into()], ..Config::default() })
    }

    // Canaries are assembled at runtime so this file itself never holds a live-looking key.
    fn canaries() -> Vec<(&'static str, String)> {
        let a = |p: &str, n: usize, ch: &str| format!("{p}{}", ch.repeat(n));
        vec![
            ("openai", a("sk-proj-", 3, "Ab9Z1x7Q4m")),
            ("anthropic", a("sk-ant-api03-", 3, "Qw8Er5Ty2u")),
            ("github", a("ghp_", 2, "A1b2C3d4E5f6G7h8I9")),
            ("aws", format!("AKIA{}", "ABCDEFGH12345678")),
            ("stripe", format!("sk_live_{}", "4eC39HqLyjWDarjtT1zdp7dc")),
            ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r".to_string()),
            ("pem", "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA7bq98\n-----END RSA PRIVATE KEY-----".to_string()),
            ("urlpw", "postgres://admin:s3cr3tPassw0rd@db.internal:5432/app".to_string()),
            ("env", "STRIPE_WEBHOOK_SECRET=whsec_9f8e7d6c5b4a3".to_string()),
            ("bearer", "Authorization: Bearer 9f8e7d6c5b4a39f8e7d6c5b4a3".to_string()),
            ("email", "jane.canary@example-corp.com".to_string()),
            ("phone", "(555) 867-5309".to_string()),
            ("addr", "1600 Pennsylvania Avenue".to_string()),
            ("card", "4111 1111 1111 1111".to_string()),
            ("ssn", "078-05-1120".to_string()),
            ("ip", "203.0.113.77".to_string()),
            ("custom", "Project Falcon".to_string()),
            ("custom_re", "ACME-4821".to_string()),
        ]
    }

    #[test]
    fn every_canary_is_removed_with_layers_on() {
        let s = all();
        for (name, c) in canaries() {
            let input = format!("note: {c} end");
            let out = s.scrub(&input);
            // The distinctive core of the canary must be gone.
            let core = c.trim_start_matches("Authorization: Bearer ").trim_start_matches("STRIPE_WEBHOOK_SECRET=");
            assert!(!out.contains(core), "{name} survived: {out}");
            assert!(out.contains("[") && out.contains("]"), "{name}: no placeholder: {out}");
        }
    }

    #[test]
    fn every_canary_is_kept_with_layers_off() {
        let s = Scrubber::new(&Config { secrets: false, pii: false, ..Config::default() });
        assert!(s.is_noop());
        for (name, c) in canaries() {
            assert_eq!(s.scrub(&c), c, "{name} changed with layers off");
        }
    }

    #[test]
    fn layers_are_independent() {
        let stripe = canaries().into_iter().find(|(name, _)| *name == "stripe").unwrap().1;
        let mixed = format!("a@b-corp.com {stripe}");
        let secrets_only = Scrubber::new(&Config { pii: false, ..Config::default() });
        assert!(secrets_only.scrub(&mixed).contains("a@b-corp.com"));
        assert!(!secrets_only.scrub(&stripe).contains("4eC39"));
        let pii_only = Scrubber::new(&Config { secrets: false, ..Config::default() });
        assert!(pii_only.scrub("a@b-corp.com").contains("[EMAIL]"));
        assert!(pii_only.scrub(&stripe).contains("4eC39"));
    }

    #[test]
    fn typed_placeholders_and_idempotent() {
        let s = all();
        let once = s.scrub("k=sk-ant-api03-Qw8Er5Ty2uQw8Er5Ty2u mail me@corp-x.org").into_owned();
        assert!(once.contains("[API_KEY:anthropic]") && once.contains("[EMAIL]"), "{once}");
        assert_eq!(s.scrub(&once), once);
    }

    #[test]
    fn url_keeps_scheme_and_host_loses_password() {
        let o = all().scrub("postgres://admin:s3cr3tPassw0rd@db.internal:5432/app").into_owned();
        assert_eq!(o, "postgres://[URL_CREDENTIALS]@db.internal:5432/app");
    }

    #[test]
    fn no_false_positives_on_common_code() {
        let s = Scrubber::new(&Config::default());
        let fixtures = [
            "550e8400-e29b-41d4-a716-446655440000",                         // UUID
            "commit 9fceb02d0ae598e95dc970b74767f19372d61af8",              // git SHA-1
            "sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", // SHA-256
            "aGVsbG8gd29ybGQgdGhpcyBpcyBhIGJhc2U2NCB0ZXN0IGZpeHR1cmU=",     // base64 fixture
            "let ts = 1730000000000; // ms",                                // 13-digit timestamp
            "ts=1730000000",                                                // 10-digit epoch
            "build v10.0.19045.1 MAX_TOKENS=4096",                           // versions, numeric limits
            "listen 127.0.0.1:8080 bind 0.0.0.0",                           // loopback
            "date 2024-11-05 build 12-34-5678",                             // dates / ids
            "icon@2x.png logo@1.svg",                                       // asset names
            "the task-force-assessment-documentation-framework",            // prose with sk- inside
            "bearer or basic authentication in header",                     // prose after scheme
            "token = self.next_token()",                                    // code, unquoted
            "password = args.password",                                     // code reference
            "let api_key = \"${API_KEY}\";",                                // env reference
            "API_KEY=your_key_here",                                        // placeholder
            "OPENAI_API_KEY=$OPENAI_API_KEY",                               // shell ref
            "TOKEN_FILE=/etc/app/token",                                    // path
            "https://example.com/a/b?x=1",                                  // url, no creds
            "cargo build --release -j 8",                                   // plain code
            "4111111111111112",                                             // fails Luhn
        ];
        for f in fixtures {
            assert_eq!(s.scrub(f), f, "false positive on {f:?} -> {}", s.scrub(f));
        }
    }

    #[test]
    fn scientific_text_is_left_alone() {
        let s = Scrubber::new(&Config::default());
        for keep in [
            "combined with the basic helix-loop-helix transcription factor Ne",
            "for basic research. The generation of diseased models",
            "contains basic mistakes. It attributes all deforestation",
            "tyrosine hydroxylase (TH, EC 1.14.16.2) from beef brain",
            "see version 1.2.3.4 of the protocol",
            "available at diagnomx.eu/vs/5309904231178427.",
            "the Social Security number 000-00-0000, presumably many",
            "1975 International Court of Justice 12, the 1996 Jaffa Road bus bombings",
            "2 Supreme Court Justice Scalia quoted in",
        ] {
            assert_eq!(s.scrub(keep), keep, "false positive on {keep:?}");
        }
        for leak in [
            "Authorization: Basic dXNlcjpwYXNzd29yZDEyMw==", "Bearer ab12cd34ef56gh78", "SSN 078-05-1120", "10 Downing Street",
            "Authorization: Basic YTpiYw==", "Authorization: Bearer abcdefghijklmnopqrstuvwxyzabcdef", "Bearer abcdefghijklmnopqrstuvwxyzabcdef",
            "card=4111111111111111", "pan=4111 1111 1111 1111", r#""card_number": "4111111111111111""#,
            "Deployment host: build 192.168.1.42", "Authorization: Basic YTpi", "Authorization: Bearer abc1234", "card=4111111111111111 12/27", "card=4111  1111 1111 1111", "pay 4111  1111 1111 1111 now",
        ] {
            assert_ne!(s.scrub(leak), leak, "missed {leak:?}");
        }
    }

    #[test]
    fn env_pin_overrides_and_is_reported() {
        // single test touching this env var, so no cross-test race
        std::env::set_var("RECALL_SCRUB_STORE_REDACTED", "0");
        assert_eq!(env_store_redacted(), Some(false));
        let (c, source) = Config::resolve(Some(r#"{"scrub":{"store_redacted":true}}"#), true, env_store_redacted());
        assert!(!c.store_redacted, "the env pin beats the file and the marker");
        assert_eq!(source, "env RECALL_SCRUB_STORE_REDACTED");
        init_new_install();
        assert!(!REDACT_THIS_RUN.load(std::sync::atomic::Ordering::Relaxed), "an env pin is the choice; the new-install default must not apply");
        std::env::set_var("RECALL_SCRUB_STORE_REDACTED", "1");
        assert_eq!(env_store_redacted(), Some(true));
        std::env::set_var("RECALL_SCRUB_STORE_REDACTED", "maybe");
        assert_eq!(env_store_redacted(), None);
        std::env::remove_var("RECALL_SCRUB_STORE_REDACTED");
        assert_eq!(env_store_redacted(), None);
    }

    #[test]
    fn stored_form_is_lossy_utf8_and_leaves_plain_text_alone() {
        // plain text is never changed whatever the machine's settings are
        assert_eq!(crate::store::stored_form(b"plain words\r\nsecond line".to_vec()), "plain words\r\nsecond line");
        assert_eq!(crate::store::stored_form(vec![b'a', 0xFF, b'b']), "a\u{FFFD}b");
    }

    #[test]
    fn luhn_and_prefix_gate_cards() {
        let s = Scrubber::new(&Config::default());
        assert_eq!(s.scrub("4111111111111111"), "[CARD]");
        assert_eq!(s.scrub("5500-0000-0000-0004"), "[CARD]");
        assert_eq!(s.scrub("3782 822463 10005"), "[CARD]");
        assert_eq!(s.scrub("4111111111111112"), "4111111111111112");
    }

    #[test]
    fn config_parses_and_defaults() {
        assert_eq!(Config::from_json("{}"), Config::default());
        assert_eq!(Config::from_json("not json"), Config::default());
        let c = Config::from_json(r#"{"refresh":"nightly","scrub":{"secrets":false,"pii":"off","custom":["x"],"store_redacted":true}}"#);
        assert!(!c.secrets && !c.pii && c.store_redacted && c.custom == vec!["x".to_string()]);
        assert!(Config::default().secrets && Config::default().pii && !Config::default().store_redacted);
    }

    #[test]
    fn hidden_characters_do_not_hide_a_secret() {
        let s = Scrubber::new(&Config::default());
        // Built from pieces so the repository never holds a key-shaped literal.
        let (pre, rest, tail) = ("sk", "proj", "Ab9Z1x7Q4m");
        let key = format!("{pre}-{rest}-{tail}{tail}");
        assert_eq!(s.scrub(&key), "[API_KEY:openai]", "control: the plain key is caught");
        for (what, hidden) in [
            ("zero-width space after the prefix", format!("{pre}-\u{200B}{rest}-{tail}{tail}")),
            ("zero-width space inside the body", format!("{pre}-{rest}-{tail}\u{200B}{tail}")),
            ("zero-width joiner", format!("{pre}-{rest}-{tail}\u{200D}{tail}")),
            ("soft hyphen", format!("{pre}-{rest}-{tail}\u{00AD}{tail}")),
            ("bidi override", format!("{pre}-{rest}-\u{202E}{tail}{tail}")),
            ("byte-order mark", format!("{pre}-{rest}-{tail}\u{FEFF}{tail}")),
            ("NUL", format!("{pre}-{rest}-{tail}\0{tail}")),
            ("ANSI colour code", format!("{pre}-{rest}-{tail}\x1b[31m{tail}")),
        ] {
            let input = format!("token: {hidden} end");
            let out = s.scrub(&input);
            assert_eq!(out, "token: [API_KEY:openai] end", "{what} let a key through: {out:?}");
        }
        // a PEM block cut up by invisible characters
        let pem = "-----BEGIN PRIVATE\u{200B} KEY-----\nMIIEvQIBADANBg\u{200B}kqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----";
        assert!(s.scrub(pem).contains("[PRIVATE_KEY]"), "{:?}", s.scrub(pem));
        // idempotent, like every other layer
        let once = s.scrub(&format!("a {pre}-\u{200B}{rest}-{tail}{tail} b")).into_owned();
        assert_eq!(s.scrub(&once), once);
    }

    #[test]
    fn text_with_no_secret_is_returned_untouched_even_with_joiners() {
        let s = Scrubber::new(&Config::default());
        for keep in [
            "family \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467} photo",
            "persian \u{0645}\u{06CC}\u{200C}\u{062E}\u{0648}\u{0627}\u{0647}\u{0645}",
            "colour \x1b[31mred\x1b[0m and a NUL\0 in prose",
        ] {
            let out = s.scrub(keep);
            assert!(matches!(out, Cow::Borrowed(_)) && out == keep, "changed {keep:?} -> {out:?}");
        }
    }

    #[test]
    fn cleaning_never_removes_a_detection_the_raw_text_already_had() {
        let s = Scrubber::new(&Config::default());
        let (pre, rest, tail) = ("sk", "proj", "Ab9Z1x7Q4m");
        // The NUL is the only thing giving the key its leading word boundary.
        let input = format!("x\0{pre}-{rest}-{tail}{tail}");
        let raw_only = s.scrub_clean(&input);
        assert!(raw_only.contains("[API_KEY:openai]"), "premise: the raw text matches: {raw_only:?}");
        let out = s.scrub(&input);
        assert!(out.contains("[API_KEY:openai]") && !out.contains(tail), "{out:?}");
    }

    #[test]
    fn colon_separated_ansi_parameters_do_not_split_a_key() {
        let s = Scrubber::new(&Config::default());
        let (pre, rest, tail) = ("sk", "proj", "Ab9Z1x7Q4m");
        let input = format!("k {pre}-{rest}-{tail}\x1b[38:2:1:2:3m{tail} e");
        assert_eq!(s.scrub(&input), "k [API_KEY:openai] e");
    }

    #[test]
    fn a_long_prefix_before_the_hidden_character_does_not_leak_the_suffix() {
        let s = Scrubber::new(&Config::default());
        let (pre, rest, tail) = ("sk", "proj", "Ab9Z1x7Q4m");
        for hide in ["\u{200B}", "\0", "\x1b[38:2:1:2:3m"] {
            let input = format!("{pre}-{rest}-{tail}{tail}{hide}{tail}");
            let out = s.scrub(&input);
            assert_eq!(out, "[API_KEY:openai]", "{hide:?} left part of the key: {out:?}");
        }
    }

    #[test]
    fn unrelated_rules_do_not_strip_joiners_from_clean_text() {
        let text = "family \u{1F468}\u{200D}\u{1F469} at 127.0.0.1 and \u{0645}\u{06CC}\u{200C}\u{062E}";
        let with_custom = Scrubber::new(&Config { custom: vec!["Project Falcon".into()], ..Config::default() });
        for sc in [&with_custom, &Scrubber::new(&Config::default())] {
            let out = sc.scrub(text);
            assert_eq!(out, text, "joiners were stripped although nothing was redacted");
        }
    }

    #[test]
    fn an_unrelated_redaction_does_not_hide_a_key_that_needs_its_boundary() {
        let s = Scrubber::new(&Config::default());
        let (pre, rest, tail) = ("sk", "proj", "Ab9Z1x7Q4m");
        let input = format!("a@corp.com x\0{pre}-{rest}-{tail}{tail}");
        let out = s.scrub(&input);
        assert!(out.contains("[EMAIL]") && out.contains("[API_KEY:openai]") && !out.contains(tail), "{out:?}");
    }

    #[test]
    fn a_split_key_does_not_cost_unrelated_text_its_joiners() {
        let s = Scrubber::new(&Config::default());
        let (pre, rest, tail) = ("sk", "proj", "Ab9Z1x7Q4m");
        let input = format!(
            "family \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}; \u{0645}\u{06CC}\u{200C}\u{062E}\u{0648}; token: {pre}-{rest}-{tail}\u{200B}{tail}"
        );
        let out = s.scrub(&input);
        assert_eq!(
            out,
            "family \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}; \u{0645}\u{06CC}\u{200C}\u{062E}\u{0648}; token: [API_KEY:openai]"
        );
    }

    #[test]
    fn c1_ansi_sequences_do_not_split_a_key() {
        let s = Scrubber::new(&Config::default());
        let (pre, rest, tail) = ("sk", "proj", "Ab9Z1x7Q4m");
        let input = format!("{pre}-{rest}-{tail}{tail}\u{9b}38;2;1;2;3m{tail}");
        assert_eq!(s.scrub(&input), "[API_KEY:openai]");
    }

    #[test]
    fn many_hidden_runs_stay_bounded_and_clean_text_is_untouched() {
        let s = Scrubber::new(&Config::default());
        let text = "ab\u{200B}".repeat(5_000);
        let started = std::time::Instant::now();
        let out = s.scrub(&text);
        assert!(matches!(out, Cow::Borrowed(_)) && out == text);
        assert!(started.elapsed() < std::time::Duration::from_secs(5), "{:?}", started.elapsed());
    }

    #[test]
    fn raw_detections_survive_when_cleaning_would_lose_the_boundaries() {
        let s = Scrubber::new(&Config::default());
        // An AWS-style id needs a word boundary on BOTH sides; the NULs are the only boundaries.
        let aws = format!("x\0{}{}\0y", "AK", "IAABCDEFGH12345678");
        let out = s.scrub(&aws);
        assert!(out.contains("[API_KEY:aws]"), "{out:?}");
        // Many harmless hidden runs in front of a key that needs its NUL boundary.
        let (pre, rest, tail) = ("sk", "proj", "Ab9Z1x7Q4m");
        let many = format!("{}x\0{pre}-{rest}-{tail}{tail}", "a\0b ".repeat(40));
        let out = s.scrub(&many);
        assert!(out.contains("[API_KEY:openai]") && !out.contains(tail), "{out:?}");
    }

    /// `cargo test --release scrub_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn scrub_cost() {
        let snippet = "In the quarterly notes we discussed the retrieval pipeline, RRF fusion weights, and the             chunk budget for BGE (commit 9fceb02d0ae598e95dc970b74767f19372d61af8, build 1.4.2). Contact             jordan@example-corp.com or (555) 867-5309 about 12 Main Street. token = self.next_token()";
        // Cold = a fresh process's first scrub of a realistic snippet: only gated-in rules compile.
        let t0 = std::time::Instant::now();
        let s = Scrubber::new(&Config::default());
        let _ = s.scrub(snippet);
        let cold = t0.elapsed();
        let t_plain = std::time::Instant::now();
        let _ = s.scrub("plain words only");
        eprintln!("scrub_cost: plain text after warm {:?}", t_plain.elapsed());
        let n = 4000;
        let t1 = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(s.scrub(std::hint::black_box(snippet)));
        }
        let per = t1.elapsed() / n;
        eprintln!("scrub_cost: cold compile+load {:?}; steady per 300-char snippet {:?}; 20 hits ~ {:?}", cold, per, per * 20);
    }

    #[test]
    #[ignore]
    fn filler_cold() {
        let t0 = std::time::Instant::now();
        let s = Scrubber::new(&Config::default());
        let filler = format!("filler text number 12 {}", "lorem ipsum ".repeat(60));
        for _ in 0..40 {
            let _ = s.scrub(&filler);
        }
        let d = t0.elapsed();
        let hit: Vec<_> = secret_rules().iter().chain(pii_rules().iter()).filter(|r| r.re.re.get().is_some()).map(|r| r.re.pat[..r.re.pat.len().min(40)].to_string()).collect();
        eprintln!("filler_cold: {:?}; compiled: {:?}", d, hit);
    }

    /// `SCRUB_PROBE_CORPUS=<corpus.jsonl> cargo test --release scrub_probe -- --ignored --nocapture`
    /// Which rule fires on which docs of a public corpus, with the changed window.
    #[test]
    #[ignore]
    fn scrub_probe() {
        let Ok(path) = std::env::var("SCRUB_PROBE_CORPUS") else { return };
        let text = std::fs::read_to_string(path).unwrap();
        let mut by_rule: std::collections::BTreeMap<String, (usize, Vec<String>)> = Default::default();
        let mut docs = 0usize;
        for line in text.lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let body = format!("{} {}", v["title"].as_str().unwrap_or(""), v["text"].as_str().unwrap_or(""));
            docs += 1;
            for r in secret_rules().iter().chain(pii_rules().iter()) {
                let out = apply(Cow::Borrowed(body.as_str()), std::slice::from_ref(r));
                if out != body.as_str() {
                    let at = body.bytes().zip(out.bytes()).position(|(a, b)| a != b).unwrap_or(0);
                    let lo = body.floor_char_boundary(at.saturating_sub(25));
                    let hi = body.ceil_char_boundary((at + 40).min(body.len()));
                    let e = by_rule.entry(r.re.pat.chars().take(48).collect()).or_default();
                    e.0 += 1;
                    if e.1.len() < 6 {
                        e.1.push(body[lo..hi].replace('\n', " "));
                    }
                }
            }
        }
        eprintln!("scrub_probe: {docs} docs");
        for (k, (n, ex)) in by_rule {
            eprintln!("RULE {k}  docs={n}");
            for s in ex {
                eprintln!("    ...{s}...");
            }
        }
    }

    #[test]
    #[ignore]
    fn rule_compile_cost() {
        for r in secret_rules().iter().chain(pii_rules().iter()) {
            let t = std::time::Instant::now();
            r.re.regex();
            let d = t.elapsed();
            if d.as_micros() > 300 {
                eprintln!("compile {:>8?}  {}", d, &r.re.pat[..r.re.pat.len().min(60)]);
            }
        }
    }

    #[test]
    fn review_round1_cases() {
        let s = Scrubber::new(&Config::default());
        for leak in [
            "PASSWORD=hunter2xyz", "TOKEN=abc123def456", "KEY=abc123def456", "DB_PASSWORD=12345678",
            r#"password = "correct horse battery staple""#, r#"password = "abc'defghijk""#,
            "ids 4111111111111111 12", "call +442079460958", "call +44 20 7946 0958",
        ] {
            let o = s.scrub(leak);
            assert!(o.contains('[') && o != leak, "leaked: {leak} -> {o}");
        }
        for keep in [r#"token_type = "identifier""#, r#"secret_count = "disabled""#, "DB_PASSWORD=$DB_PASS", "MAX_TOKENS=4096"] {
            assert_eq!(s.scrub(keep), keep, "false positive: {keep}");
        }
        for leak in [r#"PASSWORD="correct horse battery staple""#, "PASSWORD=[realSecret123", "PASSWORD=*realSecret123", "PASSWORD=yourActualPassword93"] {
            let o = s.scrub(leak);
            assert!(!o.contains("horse") && !o.contains("realSecret") && !o.contains("Actual"), "leaked: {leak} -> {o}");
        }
        assert!(!s.scrub("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA7bq98").contains("MIIEow"));
        for keep in ["API_KEY=your_api_key", "API_KEY=<your-key>", "API_KEY=[REDACTED]"] {
            assert_eq!(s.scrub(keep), keep, "false positive: {keep}");
        }
        let chain = Scrubber::new(&Config { custom: vec!["Project Falcon".into(), "CUSTOM".into()], ..Config::default() });
        assert_eq!(chain.scrub("Project Falcon"), "[CUSTOM]");
        // custom term never rewrites a placeholder, and is idempotent through two passes
        let c = Scrubber::new(&Config { custom: vec!["EMAIL".into(), "C++".into()], ..Config::default() });
        let once = c.scrub("me@corp-x.org likes C++ and EMAIL").into_owned();
        assert_eq!(once, "[EMAIL] likes [CUSTOM] and [CUSTOM]");
        assert_eq!(c.scrub(&once), once);
    }

    #[test]
    fn pointers_name_kind_and_line_never_the_value() {
        let s = Scrubber::new(&Config::default());
        let canary = format!("sk-proj-{}", "Ab9Z1x7Q4m".repeat(3));
        let text = format!("# notes\nplain line\nmail jane.canary@example-corp.com here\nkey = \"{canary}\"\nthe literal [EMAIL] word\n");
        let r = s.redactions(&text);
        assert_eq!(
            r,
            vec![Redaction { kind: "EMAIL".into(), line: 3 }, Redaction { kind: "API_KEY:openai".into(), line: 4 }],
        );
        let json = serde_json::to_string(&r).unwrap();
        assert!(!json.contains("canary") && !json.contains("Ab9Z"), "value leaked into pointer: {json}");
        assert_eq!(pointer_line(&r), "EMAIL@L3, API_KEY:openai@L4");
        assert!(Scrubber::new(&Config { secrets: false, pii: false, ..Config::default() }).redactions(&text).is_empty());
    }

    #[test]
    fn pointers_read_the_file_and_refuse_escapes() {
        let dir = std::env::temp_dir().join(format!("scrub_ptr_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub/n.md"), "a\nb@corp-x.org\n").unwrap();
        let s = Scrubber::new(&Config::default());
        assert_eq!(pointers(&s, &dir, "sub/n.md"), vec![Redaction { kind: "EMAIL".into(), line: 2 }]);
        assert!(pointers(&s, &dir, "../x.md").is_empty());
        assert!(pointers(&s, &dir, "").is_empty());
        assert!(pointers(&s, &dir, "missing.md").is_empty());
        // a file that resolves outside the root is refused (symlink creation may need privileges;
        // the containment check itself is what is under test, via a `..`-free alias of a sibling)
        let outside = std::env::temp_dir().join(format!("scrub_ptr_out_{}", std::process::id()));
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("o.md"), "x@corp-x.org\n").unwrap();
        #[cfg(windows)]
        let linked = std::os::windows::fs::symlink_file(outside.join("o.md"), dir.join("link.md")).is_ok();
        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(outside.join("o.md"), dir.join("link.md")).is_ok();
        if linked {
            assert!(pointers(&s, &dir, "link.md").is_empty(), "symlink escaped the vault");
        }
        let _ = std::fs::remove_dir_all(&outside);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pointer_cap_holds_on_the_fallback_path() {
        let s = Scrubber::new(&Config::default());
        let text = "password = \"abc[SECRET]xyz1234\"
".repeat(1000);
        assert_eq!(s.redactions(&text).len(), MAX_POINTERS);
    }

    #[test]
    fn non_regular_targets_are_refused() {
        let dir = std::env::temp_dir().join(format!("scrub_ptr_nr_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("adir.md")).unwrap();
        let s = Scrubber::new(&Config::default());
        assert!(pointers(&s, &dir, "adir.md").is_empty());
        #[cfg(unix)]
        {
            let fifo = std::ffi::CString::new(dir.join("pipe.md").to_str().unwrap()).unwrap();
            // SAFETY: valid NUL-terminated path.
            if unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) } == 0 {
                let t = std::time::Instant::now();
                assert!(pointers(&s, &dir, "pipe.md").is_empty());
                assert!(t.elapsed().as_secs() < 5, "FIFO blocked the query");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_handle_is_inside_root_and_regular() {
        let dir = std::env::temp_dir().join(format!("scrub_ptr_oc_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub/ok.md"), "m@corp-x.org\n").unwrap();
        assert!(open_confined(&dir, std::path::Path::new("sub/ok.md")).is_some());
        assert!(open_confined(&dir, std::path::Path::new("sub/nope.md")).is_none());
        // a sibling dir sharing the root's name as a prefix is not inside it
        let sib = std::env::temp_dir().join(format!("scrub_ptr_oc_{}_evil", std::process::id()));
        std::fs::create_dir_all(&sib).unwrap();
        std::fs::write(sib.join("x.md"), "x").unwrap();
        assert!(open_confined(&dir, std::path::Path::new("sub/../../x.md")).is_none());
        let _ = std::fs::remove_dir_all(&sib);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_redacted_is_decided_from_one_reading_of_the_file() {
        let on = |text: Option<&str>, marker: bool| {
            let (c, source) = Config::resolve(text, marker, None);
            (c.store_redacted, source)
        };
        const DB: &str = "database (this index was started redacted)";
        const LEGACY: &str = "default (existing index, no stated choice)";
        // marker on: redact unless the file itself states a choice
        assert_eq!(on(None, true), (true, DB), "no settings file");
        assert_eq!(on(Some(r#"{"vault":"/n"}"#), true), (true, DB), "key absent");
        assert_eq!(on(Some("{ not json"), true), (true, DB), "unreadable follows the marker");
        assert_eq!(on(Some(r#"{"scrub":{"store_redacted":"maybe"}}"#), true), (true, DB), "a value the loader ignores is not a choice");
        assert_eq!(on(Some(r#"{"scrub":{"store_redacted":false}}"#), true), (false, "settings.json"), "an explicit off wins");
        assert_eq!(on(Some(r#"{"scrub":{"store_redacted":"on"}}"#), true), (true, "settings.json"));
        // no marker: the file, else the legacy default
        assert_eq!(on(Some(r#"{"scrub":{"store_redacted":true}}"#), false), (true, "settings.json"));
        assert_eq!(on(Some(r#"{"vault":"/n"}"#), false), (false, LEGACY));
        assert_eq!(on(Some("{ not json"), false), (false, LEGACY));
        assert_eq!(on(None, false), (false, LEGACY));
        // the env pin beats both, either way
        assert!(Config::resolve(Some(r#"{"scrub":{"store_redacted":false}}"#), false, Some(true)).0.store_redacted);
        assert!(!Config::resolve(None, true, Some(false)).0.store_redacted);
        // the other scrub settings come from the same reading
        let (c, _) = Config::resolve(Some(r#"{"scrub":{"pii":false,"custom":["x"]}}"#), true, None);
        assert!(!c.pii && c.secrets && c.store_redacted && c.custom == vec!["x".to_string()]);
    }

    #[test]
    fn a_profile_with_nothing_indexed_gets_store_redacted_unless_it_already_chose() {
        let base = std::env::temp_dir().join(format!("scrub_new_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let fresh = base.join("fresh");
        assert!(matches!(init_new_install_in(&fresh), Some(Ok(()))));
        assert!(Config::load_from(&fresh.join("settings.json")).store_redacted);
        assert!(init_new_install_in(&fresh).is_none(), "must not rewrite a stated choice");
        // The documented install order: `settings set vault` creates settings.json before the
        // first index. That file must not read as "existing install, leave redaction off".
        let vault_first = base.join("vault_first");
        let path = vault_first.join("settings.json");
        crate::settings::Settings::set_in(&path, "vault", base.to_str().unwrap()).unwrap();
        assert!(matches!(init_new_install_in(&vault_first), Some(Ok(()))), "a settings file with only a vault key is still a new profile");
        assert!(Config::load_from(&path).store_redacted);
        assert!(crate::settings::Settings::load_from(&path).vault.is_some(), "the vault setting is kept");
        // an explicit off is the user's choice
        let chose_off = base.join("off");
        std::fs::create_dir_all(&chose_off).unwrap();
        std::fs::write(chose_off.join("settings.json"), r#"{"scrub":{"store_redacted":false}}"#).unwrap();
        assert!(init_new_install_in(&chose_off).is_none());
        assert!(!Config::load_from(&chose_off.join("settings.json")).store_redacted);
        // a value the loader would ignore is not a choice
        std::fs::write(chose_off.join("settings.json"), r#"{"scrub":{"store_redacted":"maybe"}}"#).unwrap();
        assert!(matches!(init_new_install_in(&chose_off), Some(Ok(()))));
        assert!(Config::load_from(&chose_off.join("settings.json")).store_redacted);
        // a file that is not a JSON object still means "redact" (Some), but is left byte for byte
        let broken = base.join("broken");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("settings.json"), "{ not json").unwrap();
        assert!(matches!(init_new_install_in(&broken), Some(Err(_))), "no readable choice must fail closed, not read as an existing install");
        assert_eq!(std::fs::read_to_string(broken.join("settings.json")).unwrap(), "{ not json");
        // what `Store::open` asks before calling it: only a store with no documents is new
        let store = crate::store::Store::open_at(&base.join("recall.db")).unwrap();
        assert!(!store.has_documents().unwrap());
        store.insert("a.md", "a", "body", "vault").unwrap();
        assert!(store.has_documents().unwrap(), "an indexed profile keeps what it has");
        // the database's own record of how its index was started
        assert!(!store.started_redacted(), "no marker until a writable open decides");
        store.set_started_redacted(true).unwrap();
        assert!(store.started_redacted());
        store.set_started_redacted(false).unwrap();
        assert!(!store.started_redacted());
        drop(store);
        // A database made before `profile_meta` existed (main's schema), opened read-only: no
        // marker, no error, and the table is not created by the read.
        let old = base.join("old.db");
        {
            let c = rusqlite::Connection::open(&old).unwrap();
            c.execute_batch("CREATE TABLE documents (id INTEGER PRIMARY KEY, body TEXT NOT NULL); INSERT INTO documents(body) VALUES ('x');").unwrap();
        }
        let ro = crate::store::Store::from_conn(
            rusqlite::Connection::open_with_flags(&old, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap(),
        );
        assert!(!ro.started_redacted(), "a missing table reads as no marker: a legacy index stays as it is");
        assert!(ro.has_documents().unwrap());
        let tables: i64 = ro.conn.query_row("SELECT count(*) FROM sqlite_master WHERE name = 'profile_meta'", [], |r| r.get(0)).unwrap();
        assert_eq!(tables, 0, "a read-only open must not create the table");
        drop(ro);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn changed_line_always_gets_a_pointer() {
        let s = Scrubber::new(&Config::default());
        let r = s.redactions("password = \"abc[SECRET]xyz1234\"\n");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].line, 1);
    }

    #[test]
    fn bad_custom_regex_is_skipped_not_fatal() {
        let s = Scrubber::new(&Config { custom: vec!["re:(unclosed".into(), "word".into()], ..Config::default() });
        assert_eq!(s.scrub("a word b"), "a [CUSTOM] b");
    }
}
