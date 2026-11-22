// gemini.rs — the `gemini` retrieval channel over Vertex AI `gemini-embedding-001`.
//
// WHY THIS FILE EXISTS
// --------------------
// Gemini vectors were WRITE-ONLY until 2026-07-31. `vertex_backfill.py` /
// `vertex_embed_parallel.py` filled `embeddings3` / `embeddings4`, but no Rust code read
// them, so `recall query` could not use a single one of them. "The Gemini embeddings are
// done" was true of the table and false of the product.
//
// This mirrors `voyage.rs` deliberately, line-shape for line-shape, so the two channels
// stay diffable. The differences are only the ones that must differ:
//
//   * AUTH. Voyage is a bearer API key from the environment. Vertex needs a short-lived
//     OAuth access token, minted here from Application Default Credentials (a
//     `service_account` key, or an `authorized_user` client_id + client_secret +
//     refresh_token).
//   * DIMS. 3072 vs 1024, so the resident cache is ~4x the bytes.
//   * TABLE. `embeddings4`, overridable via RECALL_GEMINI_TABLE for the next rebuild
//     without a recompile.
//
// TWO AUTH PATHS, AND WHY (2026-08-06)
// ------------------------------------
// There are two entirely different Google endpoints for the same model, and they bill to
// two different pots:
//
//   * VERTEX (`{loc}-aiplatform.googleapis.com`, OAuth bearer token from a service-account
//     JSON) — this is the one that **spends the owner's free Google credit**. It stays the
//     DEFAULT whenever a credential for it resolves. Do not regress this.
//   * API KEY (`generativelanguage.googleapis.com/v1beta/...:embedContent?key=…`, a plain
//     AI Studio key) — ten seconds to obtain, no service account, no gcloud, no project
//     id. It does NOT touch the Vertex credit. It exists so a machine with no service
//     account is not simply dead in the water.
//
// The selection order is in `choose_auth_mode`: explicit `RECALL_GEMINI_AUTH` wins, else a
// resolvable Vertex credential, else an API key, else a named error. The two endpoints have
// DIFFERENT response shapes and are parsed separately; both assert `GEMINI_DIM` on every
// response, because a short query vector scored against 3072-dim stored vectors is silently
// wrong rather than loud — the exact failure class that poisoned `embeddings3`.
// SECURITY: only the QUERY TEXT is ever sent. Document bodies never leave the machine.
// No token, refresh token, client secret or API key is ever logged, returned in an error
// string, or written to the database — errors carry an HTTP status, an ENV VAR NAME or a
// FILE PATH, and nothing else. Guarded by `api_key_value_never_reaches_an_error_string`.

use anyhow::{anyhow, Result};
use rusqlite::OptionalExtension;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::store::{is_query_excluded_path, Hit, PathExcludeList, Store, TierFilter};

/// Output dimensionality requested from, and asserted on, every embedding.
/// `gemini-embedding-001` is Matryoshka-trained so this is truncatable in principle —
/// but a truncated query vector scored against full-length stored vectors is silently
/// wrong, so the two are pinned to one constant on purpose.
pub const GEMINI_DIM: usize = 3072;

pub const GEMINI_MODEL: &str = "gemini-embedding-001";

/// Document side used `RETRIEVAL_DOCUMENT`; the query side MUST use `RETRIEVAL_QUERY`.
/// These are an asymmetric pair — using the document task type for queries measurably
/// degrades retrieval, and produces no error to tell you so.
const TASK_TYPE: &str = "RETRIEVAL_QUERY";

const TIMEOUT_SECS: u64 = 20;
const DEFAULT_LOCATION: &str = "us-central1";
const DEFAULT_TABLE: &str = "embeddings4";

// ---------------------------------------------------------------- warn once

static WARNED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
    std::sync::OnceLock::new();

fn warn_once(reason: &str) {
    let set = WARNED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    if let Ok(mut s) = set.lock() {
        if s.insert(reason.to_string()) {
            eprintln!("recall gemini: {reason}");
        }
    }
}

// ---------------------------------------------------------------- config

fn project() -> Option<String> {
    std::env::var("GOOGLE_CLOUD_PROJECT")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn location() -> String {
    std::env::var("GOOGLE_CLOUD_LOCATION")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_LOCATION.to_string())
}

/// Vector table name. Env-overridable so the next rebuild does not need a recompile.
///
/// The value is interpolated into SQL, so it is validated to `[A-Za-z0-9_]+` first — an
/// env var is not a trusted input just because it is local.
fn table() -> String {
    let t = std::env::var("RECALL_GEMINI_TABLE")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_TABLE.to_string());
    if t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !t.is_empty() {
        t
    } else {
        warn_once("RECALL_GEMINI_TABLE is not a plain identifier — using the default");
        DEFAULT_TABLE.to_string()
    }
}

// ---------------------------------------------------------------- auth mode

/// Which credential — and therefore which endpoint, and which bill — this channel uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    /// Vertex AI, OAuth bearer token from a service-account (or user) credential.
    /// Spends the owner's Google credit. The default whenever it is available.
    Vertex,
    /// `generativelanguage.googleapis.com` with a plain AI Studio API key.
    /// Does NOT spend the Vertex credit.
    ApiKey,
}

const AUTH_ENV: &str = "RECALL_GEMINI_AUTH";
/// Checked in this order; first non-empty wins.
const API_KEY_ENVS: [&str; 2] = ["GEMINI_API_KEY", "GOOGLE_API_KEY"];

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// The AI Studio key, if one is set. NEVER log, print, or format this into an error.
fn api_key() -> Option<String> {
    API_KEY_ENVS.iter().find_map(|n| env_nonempty(n))
}

/// The one message a user should be able to act on with no further reading. It names env
/// vars and file paths only — never a value.
fn credential_help() -> &'static str {
    "no Google credential found for the `gemini` channel. Pick one:\n  \
     (a) VERTEX — spends the Google credit. Point GOOGLE_APPLICATION_CREDENTIALS at a \
     service-account JSON, or drop one at %APPDATA%\\gcloud\\application_default_credentials.json \
     (~/.config/gcloud/... on unix), and set GOOGLE_CLOUD_PROJECT.\n  \
     (b) API KEY — a plain AI Studio key, does NOT spend the credit. Set GEMINI_API_KEY \
     (or GOOGLE_API_KEY).\n  \
     Force one path with RECALL_GEMINI_AUTH=vertex|apikey."
}

/// Selection rule, kept pure so it is testable without touching the filesystem, the
/// network, or the process environment.
///
/// 1. `RECALL_GEMINI_AUTH` — explicit always wins, even over a resolvable credential of the
///    other kind. That is the whole point of an override.
/// 2. a resolvable Vertex credential -> Vertex. This preserves today's behaviour and keeps
///    the free credit as the default spend.
/// 3. an API key -> ApiKey.
/// 4. neither -> an error naming BOTH options. Never a panic: this channel fails open into
///    the fts5 rescue tier, and a panic would take the whole query down with it.
fn choose_auth_mode(
    explicit: Option<&str>,
    has_vertex_credential: bool,
    has_api_key: bool,
) -> Result<AuthMode> {
    if let Some(raw) = explicit {
        // The value is NOT echoed back: a user who pastes their key into this var by
        // mistake must not have it reflected into a log line.
        return match raw.trim().to_ascii_lowercase().as_str() {
            "vertex" => Ok(AuthMode::Vertex),
            "apikey" | "api_key" | "api-key" => Ok(AuthMode::ApiKey),
            _ => Err(anyhow!(
                "{AUTH_ENV} is set to an unrecognised value — expected `vertex` or `apikey`"
            )),
        };
    }
    if has_vertex_credential {
        return Ok(AuthMode::Vertex);
    }
    if has_api_key {
        return Ok(AuthMode::ApiKey);
    }
    Err(anyhow!("{}", credential_help()))
}

/// Is there any credential the Vertex path can actually mint a token from?
///
/// Deliberately broader than "a service_account JSON exists": `mint_token` also supports the
/// `authorized_user` refresh-token flow, and that is shipped behaviour on machines with no
/// service-account key. Narrowing this to service accounts only would silently move such a
/// machine onto the API-key path — i.e. off the credit and onto a card.
fn has_vertex_credential() -> bool {
    service_account_adc().is_some() || authorized_user_adc().is_ok()
}

/// Resolved once per process — credentials do not appear mid-run, and `service_account_adc`
/// reads and parses files, which has no business being on the per-query hot path.
fn auth_mode() -> Result<AuthMode> {
    static MODE: std::sync::OnceLock<std::result::Result<AuthMode, String>> =
        std::sync::OnceLock::new();
    MODE.get_or_init(|| {
        choose_auth_mode(
            env_nonempty(AUTH_ENV).as_deref(),
            has_vertex_credential(),
            api_key().is_some(),
        )
        .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow!(e))
}

// ---------------------------------------------------------------- auth

/// Cached access token and the instant it stops being usable.
static TOKEN: std::sync::OnceLock<std::sync::Mutex<Option<(String, Instant)>>> =
    std::sync::OnceLock::new();

/// Candidate credential files, in the order the Google client libraries check them.
fn adc_candidates() -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
        let p = std::path::PathBuf::from(p);
        if p.is_file() {
            out.push(p);
        }
    }
    let base = std::env::var("APPDATA")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| std::path::PathBuf::from(h).join(".config"))
        });
    if let Some(base) = base {
        let p = base.join("gcloud").join("application_default_credentials.json");
        if p.is_file() {
            out.push(p);
        }
    }
    out
}

/// First credential file that is a usable `authorized_user`, with its parsed JSON.
///
/// 🔴 Order matters and is NOT the library default. `GOOGLE_APPLICATION_CREDENTIALS` on
/// this machine points at a **service_account** key, which cannot use the refresh-token
/// flow — it needs RS256 JWT signing, i.e. a crypto dependency. Taking the first file
/// unconditionally (as the Google libraries do) made every query fail auth, and because
/// this channel fails OPEN the failure was invisible: fts5 rescued the query and returned
/// plausible results with BM25 scores. It looked like Gemini was working.
///
/// So: scan ALL candidates and take the first `authorized_user`. Billing still follows
/// GOOGLE_CLOUD_PROJECT, not the principal, so the credit is charged either way.
fn authorized_user_adc() -> Result<serde_json::Value> {
    let mut saw = Vec::new();
    for path in adc_candidates() {
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(j) = serde_json::from_str::<serde_json::Value>(&raw) else {
            continue;
        };
        let kind = j["type"].as_str().unwrap_or("unknown").to_string();
        if kind == "authorized_user" {
            return Ok(j);
        }
        saw.push(kind);
    }
    Err(anyhow!(
        "no authorized_user credential found (saw: {}). Run `gcloud auth application-default login`",
        if saw.is_empty() { "none".into() } else { saw.join(", ") }
    ))
}

/// First credential file that is a `service_account`.
fn service_account_adc() -> Option<serde_json::Value> {
    for path in adc_candidates() {
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(j) = serde_json::from_str::<serde_json::Value>(&raw) else {
            continue;
        };
        if j["type"].as_str() == Some("service_account") {
            return Some(j);
        }
    }
    None
}

/// Mint an access token from a service-account key (RS256 JWT bearer grant).
///
/// Needed because the `authorized_user` ADC on this machine is **denied**
/// `aiplatform.endpoints.predict` on the project (verified live: 403 with and without an
/// `x-goog-user-project` header, so it is a genuine IAM gap, not a missing quota project).
/// The service account is the principal the Python backfill already authenticates as, so
/// using it here keeps both halves of the system on one identity and one bill.
///
/// `ring` (Apache-2.0 AND ISC) and `base64` are already in the tree via rustls, so this
/// adds no new supply-chain surface.
#[cfg(feature = "gemini")]
fn mint_token_service_account(j: &serde_json::Value) -> Result<(String, Duration)> {
    use base64::Engine as _;
    const URLSAFE: base64::engine::general_purpose::GeneralPurpose =
        base64::engine::general_purpose::URL_SAFE_NO_PAD;

    let email = j["client_email"]
        .as_str()
        .ok_or_else(|| anyhow!("service account key has no client_email"))?;
    let pem = j["private_key"]
        .as_str()
        .ok_or_else(|| anyhow!("service account key has no private_key"))?;

    // PEM -> DER. Google ships PKCS#8 ("BEGIN PRIVATE KEY"), which is what ring wants.
    let der_b64: String = pem
        .lines()
        .filter(|l| !l.starts_with("-----"))
        .collect::<String>();
    let der = base64::engine::general_purpose::STANDARD
        .decode(der_b64.trim())
        .map_err(|_| anyhow!("service account private_key is not valid base64 PEM"))?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let header = URLSAFE.encode(br#"{"alg":"RS256","typ":"JWT"}"#);
    let claims = URLSAFE.encode(
        serde_json::json!({
            "iss": email,
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": "https://oauth2.googleapis.com/token",
            "iat": now,
            "exp": now + 3600,
        })
        .to_string(),
    );
    let signing_input = format!("{header}.{claims}");

    let key = ring::signature::RsaKeyPair::from_pkcs8(&der)
        .map_err(|_| anyhow!("service account private_key is not a usable PKCS#8 RSA key"))?;
    let mut sig = vec![0u8; key.public().modulus_len()];
    key.sign(
        &ring::signature::RSA_PKCS1_SHA256,
        &ring::rand::SystemRandom::new(),
        signing_input.as_bytes(),
        &mut sig,
    )
    .map_err(|_| anyhow!("failed to sign the service-account assertion"))?;
    let jwt = format!("{signing_input}.{}", URLSAFE.encode(&sig));

    let resp = ureq::post("https://oauth2.googleapis.com/token")
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .send_form(&[
            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
            ("assertion", &jwt),
        ])
        .map_err(|e| match e {
            ureq::Error::Status(code, _) => anyhow!("oauth(jwt) http {code}"),
            ureq::Error::Transport(_) => anyhow!("oauth(jwt) transport/timeout"),
        })?;
    let body: serde_json::Value = resp.into_json()?;
    let tok = body["access_token"]
        .as_str()
        .ok_or_else(|| anyhow!("oauth(jwt) response had no access_token"))?
        .to_string();
    let ttl = body["expires_in"].as_u64().unwrap_or(3600).saturating_sub(60);
    Ok((tok, Duration::from_secs(ttl.max(60))))
}

#[cfg(feature = "gemini")]
fn mint_token() -> Result<(String, Duration)> {
    // Service account FIRST — it is the principal that actually holds
    // aiplatform.endpoints.predict here, and it is what the Python backfill uses.
    // The authorized_user path stays as the fallback for machines with no SA key.
    if let Some(sa) = service_account_adc() {
        match mint_token_service_account(&sa) {
            Ok(v) => return Ok(v),
            Err(e) => warn_once(&format!("service-account auth failed ({e}); trying user ADC")),
        }
    }
    let j = authorized_user_adc()?;
    let (id, secret, refresh) = (
        j["client_id"].as_str().unwrap_or(""),
        j["client_secret"].as_str().unwrap_or(""),
        j["refresh_token"].as_str().unwrap_or(""),
    );
    if id.is_empty() || secret.is_empty() || refresh.is_empty() {
        return Err(anyhow!("ADC file is missing client_id/client_secret/refresh_token"));
    }

    let resp = ureq::post("https://oauth2.googleapis.com/token")
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .send_form(&[
            ("client_id", id),
            ("client_secret", secret),
            ("refresh_token", refresh),
            ("grant_type", "refresh_token"),
        ])
        // NEVER surface the body — a token endpoint echoes credential material on error.
        .map_err(|e| match e {
            ureq::Error::Status(code, _) => anyhow!("oauth http {code}"),
            ureq::Error::Transport(_) => anyhow!("oauth transport/timeout"),
        })?;

    let body: serde_json::Value = resp.into_json()?;
    let tok = body["access_token"]
        .as_str()
        .ok_or_else(|| anyhow!("oauth response had no access_token"))?
        .to_string();
    // Refresh a minute early so a token never expires mid-flight.
    let ttl = body["expires_in"].as_u64().unwrap_or(3600).saturating_sub(60);
    Ok((tok, Duration::from_secs(ttl.max(60))))
}

#[cfg(not(feature = "gemini"))]
fn mint_token() -> Result<(String, Duration)> {
    Err(anyhow!("built without the `gemini` feature"))
}

fn access_token() -> Result<String> {
    let cell = TOKEN.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = cell.lock().map_err(|_| anyhow!("token lock poisoned"))?;
    if let Some((tok, expires)) = guard.as_ref() {
        if Instant::now() < *expires {
            return Ok(tok.clone());
        }
    }
    let (tok, ttl) = mint_token()?;
    *guard = Some((tok.clone(), Instant::now() + ttl));
    Ok(tok)
}

// ---------------------------------------------------------------- query cache

fn ensure_cache_table(store: &Store) -> Result<()> {
    store.conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS query_embeddings (
             query TEXT NOT NULL,
             model TEXT NOT NULL,
             dim   INTEGER NOT NULL,
             vec   BLOB NOT NULL,
             created_at TEXT NOT NULL DEFAULT (datetime('now')),
             PRIMARY KEY (query, model, dim)
         );",
    )?;
    Ok(())
}

fn cache_get(store: &Store, query: &str) -> Result<Option<Vec<f32>>> {
    let blob: Option<Vec<u8>> = store
        .conn
        .query_row(
            "SELECT vec FROM query_embeddings WHERE query = ?1 AND model = ?2 AND dim = ?3",
            rusqlite::params![query, GEMINI_MODEL, GEMINI_DIM as i64],
            |r| r.get(0),
        )
        .optional()?;
    Ok(blob.as_deref().and_then(decode_f32))
}

fn cache_put(store: &Store, query: &str, vec: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(vec.len() * 4);
    for v in vec {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    store.conn.execute(
        "INSERT OR REPLACE INTO query_embeddings (query, model, dim, vec) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![query, GEMINI_MODEL, GEMINI_DIM as i64, bytes],
    )?;
    Ok(())
}

// ---------------------------------------------------------------- vectors

fn decode_f32(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 || bytes.len() / 4 != GEMINI_DIM {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn l2_normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

// ---------------------------------------------------------------- response parsing

/// Turn a JSON array of numbers into a vector, asserting the dimension.
///
/// FAIL LOUDLY, never truncate and never pad. A query vector of the wrong width scored
/// against 3072-dim stored vectors produces plausible-looking nonsense with no error — the
/// same class of silent corruption that made `embeddings3` unusable. `source` is the
/// endpoint name so the caller can tell which of the two paths misbehaved; it is a
/// hard-coded literal, never anything derived from a credential.
///
/// Non-numeric entries are an error too. The previous `filter_map(as_f64)` silently DROPPED
/// them, which could turn a 3072-length array that passed the count check into a shorter
/// vector — defeating the assert it had just satisfied.
fn parse_embedding_values(values: Option<&serde_json::Value>, source: &str) -> Result<Vec<f32>> {
    let arr = values
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("{source} response had no embedding values"))?;
    if arr.len() != GEMINI_DIM {
        return Err(anyhow!(
            "{source} returned dim {}, expected {GEMINI_DIM} — refusing the vector rather \
             than score a mismatched query against the {GEMINI_DIM}-dim table",
            arr.len()
        ));
    }
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        out.push(
            v.as_f64()
                .ok_or_else(|| anyhow!("{source} response had a non-numeric embedding value"))?
                as f32,
        );
    }
    Ok(out)
}

/// Vertex `:predict` shape — `predictions[0].embeddings.values`.
fn parse_vertex_response(body: &serde_json::Value) -> Result<Vec<f32>> {
    parse_embedding_values(
        body.pointer("/predictions/0/embeddings/values"),
        "vertex",
    )
}

/// Generative Language `:embedContent` shape — `embedding.values`. A DIFFERENT shape from
/// Vertex's, which is why this is parsed separately instead of assumed to match.
fn parse_apikey_response(body: &serde_json::Value) -> Result<Vec<f32>> {
    parse_embedding_values(body.pointer("/embedding/values"), "generativelanguage")
}

// ---------------------------------------------------------------- embed

/// Dispatch on the resolved credential. Both arms return a `GEMINI_DIM` vector or an error.
#[cfg(feature = "gemini")]
fn embed_query_remote(query: &str) -> Result<Vec<f32>> {
    match auth_mode()? {
        AuthMode::Vertex => embed_query_vertex(query),
        AuthMode::ApiKey => embed_query_apikey(query),
    }
}

/// AI Studio key path. No project id, no location, no OAuth exchange — one POST.
///
/// The key goes through `.query()`, not `format!` into the URL string, so it never exists in
/// a `String` that could be captured by a panic message, a log line, or an error carrying
/// the request URL.
#[cfg(feature = "gemini")]
fn embed_query_apikey(query: &str) -> Result<Vec<f32>> {
    let key = api_key().ok_or_else(|| {
        anyhow!("{AUTH_ENV}=apikey but neither GEMINI_API_KEY nor GOOGLE_API_KEY is set")
    })?;
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:embedContent"
    );

    let resp = ureq::post(&url)
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .query("key", &key)
        .set("Content-Type", "application/json")
        .send_json(ureq::json!({
            "model": format!("models/{GEMINI_MODEL}"),
            "content": { "parts": [ { "text": query } ] },
            "taskType": TASK_TYPE,
            "outputDimensionality": GEMINI_DIM,
        }))
        // NEVER surface the body or the URL — both carry the key on this endpoint.
        .map_err(|e| match e {
            ureq::Error::Status(code, _) => anyhow!("generativelanguage http {code}"),
            ureq::Error::Transport(_) => anyhow!("generativelanguage transport/timeout"),
        })?;

    let body: serde_json::Value = resp.into_json()?;
    parse_apikey_response(&body)
}

#[cfg(feature = "gemini")]
fn embed_query_vertex(query: &str) -> Result<Vec<f32>> {
    let project = project().ok_or_else(|| anyhow!("GOOGLE_CLOUD_PROJECT is not set"))?;
    let loc = location();
    let token = access_token()?;
    let url = format!(
        "https://{loc}-aiplatform.googleapis.com/v1/projects/{project}\
         /locations/{loc}/publishers/google/models/{GEMINI_MODEL}:predict"
    );

    let resp = ureq::post(&url)
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .set("Authorization", &format!("Bearer {token}"))
        .set("Content-Type", "application/json")
        .send_json(ureq::json!({
            "instances": [{ "content": query, "task_type": TASK_TYPE }],
            "parameters": { "outputDimensionality": GEMINI_DIM },
        }))
        .map_err(|e| match e {
            ureq::Error::Status(code, _) => anyhow!("vertex http {code}"),
            ureq::Error::Transport(_) => anyhow!("vertex transport/timeout"),
        })?;

    // Assert the dimension rather than trust it. A silently-shorter query vector scored
    // against 3072-dim stored vectors is the exact failure mode that produced the
    // 1024-vs-2048 Voyage split, and it surfaces as "retrieval got worse", not an error.
    let body: serde_json::Value = resp.into_json()?;
    parse_vertex_response(&body)
}

#[cfg(not(feature = "gemini"))]
fn embed_query_remote(_query: &str) -> Result<Vec<f32>> {
    Err(anyhow!("built without the `gemini` feature"))
}

/// Cache-first query embedding. Network is touched only on a miss.
fn query_vector(store: &Store, query: &str) -> Result<Vec<f32>> {
    ensure_cache_table(store)?;
    if let Some(mut v) = cache_get(store, query)? {
        l2_normalize(&mut v);
        return Ok(v);
    }
    let raw = embed_query_remote(query)?;
    let _ = cache_put(store, query, &raw);
    let mut v = raw;
    l2_normalize(&mut v);
    Ok(v)
}

// ---------------------------------------------------------------- resident cache

/// Every live chunk vector held in RAM in one contiguous allocation.
///
/// At 3072 dims this is ~4x the Voyage footprint — roughly 12 bytes per dimension-row, so
/// ~500k chunks is ~6 GB. That is deliberate: the machine has 135 GB, and the alternative
/// (re-reading multi-GB of blobs from a 7 GB SQLite file on every query) measured 3.3s per
/// query on the smaller Voyage table alone.
///
/// Set `RECALL_GEMINI_RESIDENT=0` to force the low-memory streaming path.
struct ResidentVectors {
    ids: Vec<i64>,
    data: Vec<f32>,
    inv_norms: Vec<f32>,
}

impl ResidentVectors {
    fn rows(&self) -> usize {
        self.ids.len()
    }

    fn load(store: &Store) -> Result<Self> {
        // Mirror-copy snapshots (`tools/state/skill-radar/disabled-loader-roots-*`) and
        // generated eval results are stripped AT QUERY TIME, exactly as the BGE channel
        // does in `Store::load_all_embeddings`. Skipping this is not cosmetic: measured
        // 2026-07-31, the top 3 hits for a semantic query were three copies of the same
        // mirrored SKILL.md. The vault holds ~39k such snapshot docs, so without this the
        // channel spends most of its ranking slots on duplicates of itself.
        let exclude = PathExcludeList::load();
        let sql = format!(
            "SELECT e.doc_id, d.path, e.vec
               FROM {} e
               JOIN documents d ON d.id = e.doc_id
              WHERE d.valid_to IS NULL{}
                AND e.dim = ?1",
            table(),
            TierFilter::load().sql_clause("d")
        );
        let mut stmt = store.conn.prepare(&sql)?;

        let mut ids = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        let mut inv_norms = Vec::new();

        let mut rows = stmt.query(rusqlite::params![GEMINI_DIM as i64])?;
        while let Some(row) = rows.next()? {
            let doc_id: i64 = row.get(0)?;
            let path: String = row.get(1)?;
            if is_query_excluded_path(&exclude, &path) {
                continue;
            }
            let blob: &[u8] = row.get_ref(2)?.as_blob().unwrap_or(&[]);
            if blob.len() != GEMINI_DIM * 4 {
                continue;
            }
            data.reserve(GEMINI_DIM);
            let mut norm = 0.0f32;
            for w in blob.chunks_exact(4) {
                let x = f32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                data.push(x);
                norm += x * x;
            }
            ids.push(doc_id);
            inv_norms.push(if norm > 0.0 { 1.0 / norm.sqrt() } else { 0.0 });
        }

        Ok(Self {
            ids,
            data,
            inv_norms,
        })
    }

    /// Best (highest cosine) score per doc_id. `qvec` must be L2-normalised.
    fn best_per_doc(&self, qvec: &[f32]) -> HashMap<i64, f32> {
        let rows = self.rows();
        if rows == 0 {
            return HashMap::new();
        }
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, 8);
        let per = rows.div_ceil(threads);

        let partials: Vec<HashMap<i64, f32>> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for t in 0..threads {
                let lo = t * per;
                if lo >= rows {
                    break;
                }
                let hi = ((t + 1) * per).min(rows);
                handles.push(scope.spawn(move || {
                    let mut best: HashMap<i64, f32> = HashMap::new();
                    for i in lo..hi {
                        let row = &self.data[i * GEMINI_DIM..(i + 1) * GEMINI_DIM];
                        let mut dot = 0.0f32;
                        for (x, q) in row.iter().zip(qvec.iter()) {
                            dot += x * q;
                        }
                        let score = dot * self.inv_norms[i];
                        let id = self.ids[i];
                        best.entry(id)
                            .and_modify(|s| {
                                if score > *s {
                                    *s = score
                                }
                            })
                            .or_insert(score);
                    }
                    best
                }));
            }
            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        let mut best: HashMap<i64, f32> = HashMap::new();
        for part in partials {
            for (id, score) in part {
                best.entry(id)
                    .and_modify(|s| {
                        if score > *s {
                            *s = score
                        }
                    })
                    .or_insert(score);
            }
        }
        best
    }
}

static RESIDENT: std::sync::OnceLock<Option<ResidentVectors>> = std::sync::OnceLock::new();

fn resident_enabled() -> bool {
    !matches!(
        std::env::var("RECALL_GEMINI_RESIDENT").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

fn resident(store: &Store) -> Option<&'static ResidentVectors> {
    RESIDENT
        .get_or_init(|| match ResidentVectors::load(store) {
            Ok(v) if v.rows() > 0 => {
                // Announce the loaded count, like the BGE channel does.
                //
                // This cache is loaded ONCE per process and never refreshed. In the warm
                // daemon that means a daemon started mid-backfill serves a FROZEN snapshot
                // of however many vectors existed at startup — forever, with no error and
                // no way to tell from query output. Printing the count is the only cheap
                // signal that the daemon needs restarting after a backfill finishes.
                // Whole-corpus count: operator only (Codex tenancy b1).
                if crate::scope::current().is_operator() {
                    eprintln!(
                        "recall gemini: loaded {} chunk vectors (restart the daemon after a backfill)",
                        v.rows()
                    );
                }
                Some(v)
            }
            Ok(_) => None,
            Err(e) => {
                warn_once(&format!("resident load failed: {e}"));
                None
            }
        })
        .as_ref()
}

fn rank(mut best: HashMap<i64, f32>, depth: usize) -> Vec<i64> {
    // Tenancy: skip docs the current reader cannot see BEFORE truncating, so k stays full.
    crate::scope::drop_denied(&mut best);
    let mut ranked: Vec<(i64, f32)> = best.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(depth);
    ranked.into_iter().map(|(id, _)| id).collect()
}

// ---------------------------------------------------------------- search

/// Score the Gemini table against the query vector and return doc ids, best first.
///
/// Two details that cost real accuracy if skipped, both learned on the Voyage channel:
///   1. The table has NO foreign key to `documents`, so it accumulates orphan vectors.
///      The join on `documents.valid_to IS NULL` filters orphans and superseded revisions.
///      Without it, dead rows occupy real ranking slots — that is how R@1 hit 0.0000.
///   2. A document is several chunks. MAX-POOL per doc: a document is as relevant as its
///      single best chunk, otherwise long documents with many mediocre chunks outrank
///      short precise ones.
fn scored_doc_ids(store: &Store, qvec: &[f32], depth: usize) -> Result<Vec<i64>> {
    if resident_enabled() {
        if let Some(res) = resident(store) {
            return Ok(rank(res.best_per_doc(qvec), depth));
        }
    }

    let exclude = PathExcludeList::load();
    // Per-query slow path, so it is scoped here (the resident load above is not: it serves every
    // reader and is filtered per query instead).
    let sql = format!(
        "SELECT e.doc_id, d.path, e.vec
           FROM {} e
           JOIN documents d ON d.id = e.doc_id
          WHERE d.valid_to IS NULL{}{}
            AND e.dim = ?1",
        table(),
        TierFilter::load().sql_clause("d"),
        crate::scope::clause("d")
    );
    let mut stmt = store.conn.prepare(&sql)?;

    let mut best: HashMap<i64, f32> = HashMap::new();
    let mut rows = stmt.query(rusqlite::params![GEMINI_DIM as i64])?;
    while let Some(row) = rows.next()? {
        let doc_id: i64 = row.get(0)?;
        let path: String = row.get(1)?;
        if is_query_excluded_path(&exclude, &path) {
            continue;
        }
        let blob: &[u8] = row.get_ref(2)?.as_blob().unwrap_or(&[]);
        if blob.len() != GEMINI_DIM * 4 {
            continue;
        }
        let mut dot = 0.0f32;
        let mut norm = 0.0f32;
        for (i, w) in blob.chunks_exact(4).enumerate() {
            let x = f32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            dot += x * qvec[i];
            norm += x * x;
        }
        let score = if norm > 0.0 { dot / norm.sqrt() } else { 0.0 };
        best.entry(doc_id)
            .and_modify(|s| {
                if score > *s {
                    *s = score
                }
            })
            .or_insert(score);
    }
    Ok(rank(best, depth))
}

/// Number of documents currently readable through this channel. Used by `recall status`
/// and `doctor` so "Gemini is done" can be checked instead of believed.
pub fn readable_docs(store: &Store) -> Result<i64> {
    // Counts docs the QUERY PATH can return — so it applies the same exclude policy the
    // scan does. A count that ignores the filter would overstate coverage by the ~39k
    // mirror-copy snapshots and report progress the user can never actually retrieve.
    let exclude = PathExcludeList::load();
    let sql = format!(
        "SELECT DISTINCT e.doc_id, d.path
           FROM {} e
           JOIN documents d ON d.id = e.doc_id
          WHERE d.valid_to IS NULL{} AND e.dim = ?1",
        table(),
        TierFilter::load().sql_clause("d")
    );
    let Ok(mut stmt) = store.conn.prepare(&sql) else {
        return Ok(0);
    };
    let mut n = 0i64;
    let mut rows = stmt.query(rusqlite::params![GEMINI_DIM as i64])?;
    while let Some(row) = rows.next()? {
        let path: String = row.get(1)?;
        if !is_query_excluded_path(&exclude, &path) {
            n += 1;
        }
    }
    Ok(n)
}

/// The `gemini` retrieval channel. Mirrors `voyage_ids` so `fused()` treats it like any
/// other channel.
///
/// ALWAYS returns Ok. Any failure degrades to an empty list so the fused query still
/// returns FTS5 results via the rescue tier.
pub fn gemini_ids(
    store: &Store,
    query: &str,
    depth: usize,
    id_hit: &mut HashMap<i64, Hit>,
) -> Result<Vec<i64>> {
    let qvec = match query_vector(store, query) {
        Ok(v) => v,
        Err(e) => {
            warn_once(&e.to_string());
            return Ok(vec![]);
        }
    };

    let ids = match scored_doc_ids(store, &qvec, depth) {
        Ok(ids) => ids,
        Err(e) => {
            warn_once(&e.to_string());
            return Ok(vec![]);
        }
    };

    for id in &ids {
        if !id_hit.contains_key(id) {
            if let Some(h) = store.hit_by_id(*id)? {
                id_hit.insert(*id, h);
            }
        }
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_rejects_wrong_dimension() {
        assert!(decode_f32(&[0u8; 16]).is_none());
        assert!(decode_f32(&vec![0u8; GEMINI_DIM * 4]).is_some());
    }

    #[test]
    fn normalize_makes_unit_length() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn normalize_tolerates_zero_vector() {
        let mut v = vec![0.0f32; 4];
        l2_normalize(&mut v);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    /// An env var reaching a format!-built SQL string is an injection sink even when it is
    /// "just local config". Anything that is not a bare identifier must be refused.
    #[test]
    fn table_name_rejects_non_identifiers() {
        std::env::set_var("RECALL_GEMINI_TABLE", "embeddings4; DROP TABLE documents");
        assert_eq!(table(), DEFAULT_TABLE);
        std::env::set_var("RECALL_GEMINI_TABLE", "embeddings_v5");
        assert_eq!(table(), "embeddings_v5");
        std::env::remove_var("RECALL_GEMINI_TABLE");
        assert_eq!(table(), DEFAULT_TABLE);
    }

    // ------------------------------------------------------ auth selection
    //
    // All four branches of the selection rule, driven through the pure function so no test
    // touches the filesystem, the process environment, or the network. Zero API calls.

    #[test]
    fn explicit_env_var_wins_over_whatever_is_resolvable() {
        // vertex forced even though only an API key exists
        assert_eq!(
            choose_auth_mode(Some("vertex"), false, true).unwrap(),
            AuthMode::Vertex
        );
        // apikey forced even though a Vertex credential exists — the override must be able
        // to move the owner OFF the credit deliberately, not just onto it
        assert_eq!(
            choose_auth_mode(Some("apikey"), true, false).unwrap(),
            AuthMode::ApiKey
        );
        // spelling tolerance + case insensitivity
        for v in ["  VERTEX ", "Vertex"] {
            assert_eq!(choose_auth_mode(Some(v), false, true).unwrap(), AuthMode::Vertex);
        }
        for v in ["API_KEY", "api-key", " ApiKey "] {
            assert_eq!(choose_auth_mode(Some(v), true, false).unwrap(), AuthMode::ApiKey);
        }
    }

    #[test]
    fn service_account_beats_api_key_when_nothing_is_forced() {
        // The whole point: Vertex is where the free credit lives, so it stays the default
        // even when an API key is also sitting in the environment.
        assert_eq!(
            choose_auth_mode(None, true, true).unwrap(),
            AuthMode::Vertex
        );
        assert_eq!(
            choose_auth_mode(None, true, false).unwrap(),
            AuthMode::Vertex
        );
    }

    #[test]
    fn api_key_is_used_only_when_no_vertex_credential_resolves() {
        assert_eq!(
            choose_auth_mode(None, false, true).unwrap(),
            AuthMode::ApiKey
        );
    }

    #[test]
    fn no_credential_is_a_named_error_not_a_panic() {
        let e = choose_auth_mode(None, false, false).unwrap_err().to_string();
        for expected in [
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "RECALL_GEMINI_AUTH",
        ] {
            assert!(e.contains(expected), "error should name {expected}: {e}");
        }
    }

    #[test]
    fn unrecognised_auth_value_is_rejected() {
        let e = choose_auth_mode(Some("vertexx"), true, true)
            .unwrap_err()
            .to_string();
        assert!(e.contains(AUTH_ENV) && e.contains("vertex") && e.contains("apikey"));
    }

    // ------------------------------------------------------ response shapes

    fn values(n: usize) -> serde_json::Value {
        serde_json::Value::Array(vec![serde_json::json!(0.1); n])
    }

    #[test]
    fn both_endpoints_parse_their_own_shape() {
        let vertex = serde_json::json!({
            "predictions": [ { "embeddings": { "values": values(GEMINI_DIM) } } ]
        });
        let apikey = serde_json::json!({ "embedding": { "values": values(GEMINI_DIM) } });
        assert_eq!(parse_vertex_response(&vertex).unwrap().len(), GEMINI_DIM);
        assert_eq!(parse_apikey_response(&apikey).unwrap().len(), GEMINI_DIM);
        // and NOT each other's — the shapes genuinely differ, so a swap must fail loudly
        assert!(parse_vertex_response(&apikey).is_err());
        assert!(parse_apikey_response(&vertex).is_err());
    }

    /// The poison-the-table case. A 768-dim answer from the API-key endpoint must be
    /// refused, not truncated, padded, or written.
    #[test]
    fn dimension_mismatch_is_rejected_on_both_paths() {
        for n in [1usize, 768, GEMINI_DIM - 1, GEMINI_DIM + 1] {
            let e = parse_apikey_response(&serde_json::json!({ "embedding": { "values": values(n) } }))
                .unwrap_err()
                .to_string();
            assert!(e.contains(&n.to_string()) && e.contains(&GEMINI_DIM.to_string()), "{e}");

            assert!(parse_vertex_response(&serde_json::json!({
                "predictions": [ { "embeddings": { "values": values(n) } } ]
            }))
            .is_err());
        }
    }

    /// A right-length array with a non-numeric entry used to be silently shortened by
    /// `filter_map`, defeating the length check that had just passed.
    #[test]
    fn non_numeric_value_is_rejected_not_dropped() {
        let mut v = vec![serde_json::json!(0.1); GEMINI_DIM];
        v[7] = serde_json::json!("nan");
        let body = serde_json::json!({ "embedding": { "values": serde_json::Value::Array(v) } });
        assert!(parse_apikey_response(&body).is_err());
    }

    /// 🔴 A credential value must never reach any string this module can emit. The value
    /// below is a low-entropy placeholder, not a key — nothing here calls the network.
    #[test]
    fn api_key_value_never_reaches_an_error_string() {
        const SENTINEL: &str = "fixture-placeholder-not-a-key-000";
        std::env::set_var("GEMINI_API_KEY", SENTINEL);
        // no assert_eq! here: a failure message must not print the value it read
        assert!(
            api_key().as_deref() == Some(SENTINEL),
            "api_key() did not return the fixture value"
        );

        let mut emitted = vec![
            credential_help().to_string(),
            choose_auth_mode(None, false, false).unwrap_err().to_string(),
            // the worst case: the user pastes the key into RECALL_GEMINI_AUTH by mistake
            choose_auth_mode(Some(SENTINEL), false, true)
                .unwrap_err()
                .to_string(),
        ];
        emitted.push(
            parse_apikey_response(&serde_json::json!({ "embedding": { "values": values(4) } }))
                .unwrap_err()
                .to_string(),
        );
        emitted.push(
            parse_vertex_response(&serde_json::json!({ "nope": true }))
                .unwrap_err()
                .to_string(),
        );
        for e in &emitted {
            assert!(!e.contains(SENTINEL), "credential value leaked into an error");
        }

        // whitespace-only is not a credential
        std::env::set_var("GEMINI_API_KEY", "   ");
        assert!(env_nonempty("GEMINI_API_KEY").is_none());
        std::env::remove_var("GEMINI_API_KEY");
    }
}
