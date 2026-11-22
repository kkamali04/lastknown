// ui.rs — `recall ui`: one self-contained local page (status, settings, test search).
//
// Plain std::net HTTP/1.1, one request per connection, a thread per connection. No new web stack:
// settings go through settings.rs, status through the same payload `recall status -j` prints, and
// a test search runs THIS exe as `recall query --json`, so the page sees exactly what the CLI
// returns; the UI then scrubs every string in it (paths and labels too, not only snippets).
//
// Security: bound to 127.0.0.1 on a random port; a random per-launch token is required on every
// request (`?t=` for the page, `X-Recall-Token` for the API, which also forces a CORS preflight
// that this server never answers); the Host header must be 127.0.0.1:<port> or localhost:<port>
// (DNS rebinding); both are checked before any body is read; writes are POST only; GET opens the
// store read-only; no CORS headers; scrub.custom terms are never sent back.

use anyhow::{Context, Result};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const PAGE: &str = include_str!("ui.html");
const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY_BYTES: usize = 64 * 1024;
const MAX_QUERY_CHARS: usize = 2000;
const MAX_CONNECTIONS: usize = 32;
/// Whole request (headers and body) must arrive within this.
const REQUEST_DEADLINE: Duration = Duration::from_secs(5);
const MAX_CHILD_OUTPUT: u64 = 8 * 1024 * 1024;
/// Warm daemon answers in ~0.1-7 s; this only bounds a wedged one. Kept ABOVE the CLI's own
/// daemon read timeout (5 s today, 30 s with the worker pool) so the CLI's error reaches the page.
const DAEMON_QUERY_TIMEOUT: Duration = Duration::from_secs(45);
/// In-process: fts5 alone is well under this; a cold bge load of the full corpus is not, and is
/// stopped with a message that says to start the daemon.
const INPROC_QUERY_TIMEOUT: Duration = Duration::from_secs(60);

static ACTIVE: AtomicUsize = AtomicUsize::new(0);
static QUERY_BUSY: AtomicBool = AtomicBool::new(false);
// ponytail: serialises this process's writes only; a CLI `settings set` (or a second `recall ui`)
// at the same instant can still lose one of the two updates (read-modify-write). Each write is
// atomic, so no reader ever sees a torn file. Upgrade path: a lock file shared with set_in.
static SETTINGS_LOCK: Mutex<()> = Mutex::new(());

pub struct Ctx {
    pub port: u16,
    pub token: String,
    pub settings_path: PathBuf,
}

#[derive(Debug, Default)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
    fn query_param(&self, name: &str) -> Option<&str> {
        self.query.split('&').find_map(|kv| kv.strip_prefix(name)?.strip_prefix('='))
    }
}

pub struct Response {
    pub status: u16,
    pub ctype: &'static str,
    pub body: Vec<u8>,
}

fn json(status: u16, v: serde_json::Value) -> Response {
    Response { status, ctype: "application/json", body: v.to_string().into_bytes() }
}

fn err(status: u16, msg: &str) -> Response {
    json(status, serde_json::json!({ "error": msg }))
}

pub fn run(open: bool) -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").context("bind 127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    let ctx = std::sync::Arc::new(Ctx { port, token: new_token()?, settings_path: crate::settings::settings_path() });
    let url = format!("http://127.0.0.1:{port}/?t={}", ctx.token);
    println!("recall ui: {url}");
    println!("(Ctrl+C to stop; the token above is this launch's only key)");
    std::io::stdout().flush().ok();
    if open {
        open_browser(&url);
    }
    for stream in listener.incoming().flatten() {
        if ACTIVE.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
            ACTIVE.fetch_sub(1, Ordering::SeqCst);
            continue; // dropped: closes the socket
        }
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            serve(stream, &ctx);
            ACTIVE.fetch_sub(1, Ordering::SeqCst);
        });
    }
    Ok(())
}

fn new_token() -> Result<String> {
    let mut b = [0u8; 32];
    getrandom::fill(&mut b).map_err(|e| anyhow::anyhow!("OS random source failed: {e}"))?;
    Ok(b.iter().map(|x| format!("{x:02x}")).collect())
}

fn open_browser(url: &str) {
    // The URL is ours (digits and hex only), but no shell parses it either way.
    #[cfg(windows)]
    let r = Command::new("rundll32").args(["url.dll,FileProtocolHandler", url]).spawn();
    #[cfg(target_os = "macos")]
    let r = Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let r = Command::new("xdg-open").arg(url).spawn();
    if r.is_err() {
        eprintln!("recall ui: could not open a browser; open the URL above yourself");
    }
}

/// Reads that share ONE absolute deadline, so a client trickling a byte every few seconds cannot
/// hold a connection slot open past it (a per-read timeout alone would let it).
struct Deadline<'a> {
    s: &'a TcpStream,
    end: Instant,
}

impl Read for Deadline<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.end.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "request took too long"));
        }
        self.s.set_read_timeout(Some(left))?;
        let mut s = self.s;
        s.read(buf)
    }
}

fn serve(stream: TcpStream, ctx: &Ctx) {
    stream.set_write_timeout(Some(Duration::from_secs(10))).ok();
    let reader = Deadline { s: &stream, end: Instant::now() + REQUEST_DEADLINE };
    let mut r = BufReader::new(reader.take((MAX_HEADER_BYTES + MAX_BODY_BYTES) as u64));
    let resp = match read_head(&mut r) {
        Err(e) => err(400, &format!("{e:#}")),
        // Host and token are judged on the headers alone, before a single body byte is read.
        Ok(mut req) => match gate(&req, ctx) {
            Some(denied) => denied,
            None => match read_body(&mut r, &mut req) {
                Ok(()) => route(&req, ctx),
                Err(e) => err(400, &format!("{e:#}")),
            },
        },
    };
    drop(r);
    let _ = write_response(&stream, &resp);
}

fn read_head(r: &mut impl BufRead) -> Result<Request> {
    let mut line = String::new();
    let mut seen = r.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
    anyhow::ensure!(!method.is_empty() && target.starts_with('/'), "bad request line");
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target, String::new()),
    };
    let mut headers = Vec::new();
    loop {
        line.clear();
        let n = r.read_line(&mut line)?;
        seen += n;
        anyhow::ensure!(seen <= MAX_HEADER_BYTES, "headers too large");
        let t = line.trim_end_matches(['\r', '\n']);
        if n == 0 || t.is_empty() {
            break;
        }
        if let Some((k, v)) = t.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    Ok(Request { method, path, query, headers, body: Vec::new() })
}

fn read_body(r: &mut impl Read, req: &mut Request) -> Result<()> {
    let len: usize = req.header("content-length").map(|v| v.parse()).transpose()?.unwrap_or(0);
    anyhow::ensure!(len <= MAX_BODY_BYTES, "body too large");
    req.body = vec![0; len];
    r.read_exact(&mut req.body)?;
    Ok(())
}

fn write_response(mut stream: &TcpStream, resp: &Response) -> std::io::Result<()> {
    let reason = match resp.status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        502 => "Bad Gateway",
        429 => "Too Many Requests",
        504 => "Gateway Timeout",
        _ => "Internal Server Error",
    };
    // No Access-Control-* headers, ever. The token rides in the page URL, so no Referer leaves it.
    write!(
        stream,
        "HTTP/1.1 {} {reason}\r\nContent-Type: {}; charset=utf-8\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n\
         X-Frame-Options: DENY\r\nContent-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; \
         style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'\r\n\
         Connection: close\r\n\r\n",
        resp.status,
        resp.ctype,
        resp.body.len()
    )?;
    stream.write_all(&resp.body)?;
    stream.flush()
}

/// Constant-time compare, so the token cannot be guessed a byte at a time from response timing.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn host_ok(req: &Request, port: u16) -> bool {
    req.header("host").is_some_and(|h| {
        let h = h.to_ascii_lowercase();
        h == format!("127.0.0.1:{port}") || h == format!("localhost:{port}")
    })
}

/// `Some(403)` unless the Host and the token are right. Headers only.
fn gate(req: &Request, ctx: &Ctx) -> Option<Response> {
    if !host_ok(req, ctx.port) {
        return Some(err(403, "bad Host header"));
    }
    let token = if req.path.starts_with("/api/") { req.header("x-recall-token") } else { req.query_param("t") };
    if !token.is_some_and(|t| same(t, &ctx.token)) {
        return Some(err(403, "missing or wrong token"));
    }
    None
}

/// Gate then route, for a request already read in full (the unit tests' entry point).
pub fn handle(req: &Request, ctx: &Ctx) -> Response {
    gate(req, ctx).unwrap_or_else(|| route(req, ctx))
}

fn route(req: &Request, ctx: &Ctx) -> Response {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") => Response { status: 200, ctype: "text/html", body: PAGE.as_bytes().to_vec() },
        ("GET", "/api/status") => json(200, status()),
        ("GET", "/api/activity") => json(200, activity(&crate::settings::Settings::load_from(&ctx.settings_path))),
        ("GET", "/api/settings") => json(200, settings_view(&crate::settings::Settings::load_from(&ctx.settings_path), ctx)),
        ("POST", "/api/settings") => set_setting(req, ctx),
        ("POST", "/api/query") => run_query(req),
        ("POST", "/api/setup/scan") => setup_scan(req),
        ("POST", "/api/setup/start") => setup_start(req, ctx),
        ("GET", "/api/setup/job") => json(200, job_view()),
        (_, "/" | "/api/status" | "/api/activity" | "/api/settings" | "/api/query" | "/api/setup/scan" | "/api/setup/start" | "/api/setup/job") => {
            err(405, "method not allowed")
        }
        _ => err(404, "not found"),
    }
}

/// `~/.recall/daemon.json`, read without creating `~/.recall` (daemon::DaemonInfo::read does).
fn daemon_addr() -> Option<String> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok()?;
    let text = std::fs::read_to_string(PathBuf::from(home).join(".recall").join("daemon.json")).ok()?;
    serde_json::from_str::<serde_json::Value>(&text).ok()?["addr"].as_str().map(String::from)
}

fn daemon_up() -> Option<String> {
    daemon_addr().filter(|a| crate::daemon::ping_daemon(a).is_some())
}

fn status() -> serde_json::Value {
    let store = crate::Store::open_readonly()
        .and_then(|s| crate::status_json_payload_from_store(&s))
        .map_err(|e| format!("{e:#}"))
        .and_then(|p| serde_json::to_value(p).map_err(|e| e.to_string()));
    let mut out = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "wire_proto": crate::daemon::WIRE_PROTO,
        "daemon": if daemon_up().is_some() { "up" } else { "down" },
        "store": store.as_ref().ok(),
        "store_error": store.as_ref().err(),
        "vault": crate::store::vault_root().display().to_string(),
        "vault_source": crate::store::vault_source(),
    });
    scrub_all(&mut out, &crate::scrub::Scrubber::new(&crate::scrub::Config::load()));
    out
}

const ACTIVITY_ROWS: usize = 10;
const ERROR_LOG_TAIL_BYTES: u64 = 64 * 1024;
const MAX_STATE_BYTES: u64 = 4 * 1024 * 1024;

/// What recall records nowhere, stated rather than invented (no new logging system).
const NOT_RECORDED: [&str; 4] = [
    "embed run history: `recall embed` writes no per-run record",
    "refresh history: refresh-state.json keeps only the LAST run",
    "per index run: documents added / changed and errors (index_runs holds files seen, tombstoned, enum time)",
    "daemon query count: the daemon counts nothing",
];

fn recall_home() -> Option<PathBuf> {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok()?;
    Some(PathBuf::from(home).join(".recall"))
}

/// Last `index_runs` rows, newest first. `enum_ms` is absent on a db older than that column
/// (the read-only opener does not migrate), so that case retries without it.
fn index_runs(store: &crate::Store) -> Result<Vec<serde_json::Value>> {
    let q = |cols: &str| -> rusqlite::Result<Vec<serde_json::Value>> {
        let mut st = store.conn.prepare(&format!("SELECT ran_at, enumerated, used_fallback, tombstoned{cols} FROM index_runs ORDER BY id DESC LIMIT {ACTIVITY_ROWS}"))?;
        let rows = st.query_map([], |r| {
            // stored as unix seconds in text; shown as a date, anything else passed through
            let t: String = r.get(0)?;
            let ran_at = t.parse::<i64>().map(crate::store::freshness::fmt_utc).unwrap_or(t);
            Ok(serde_json::json!({
                "ran_at": ran_at,
                "files_seen": r.get::<_, i64>(1)?,
                "walk_fallback": r.get::<_, i64>(2)? != 0, "tombstoned": r.get::<_, i64>(3)?,
                "enum_ms": if cols.is_empty() { None } else { r.get::<_, Option<i64>>(4)? },
            }))
        })?;
        rows.collect()
    };
    Ok(q(", enum_ms").or_else(|_| q(""))?)
}

/// The last few `error-log.jsonl` entries, newest first, scrubbed (an error can quote anything).
fn recent_errors(path: &std::path::Path, sc: &crate::scrub::Scrubber) -> Vec<serde_json::Value> {
    use std::io::{Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else { return Vec::new() };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(ERROR_LOG_TAIL_BYTES);
    let mut buf = Vec::new();
    // take(): appends racing this read cannot grow it past the snapshot
    if f.seek(SeekFrom::Start(start)).is_err() || (&mut f).take(len - start).read_to_end(&mut buf).is_err() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&buf);
    let mut lines: Vec<&str> = text.lines().collect();
    if start > 0 && !lines.is_empty() {
        lines.remove(0); // a partial first line from the seek
    }
    lines.iter().rev().filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()).take(ACTIVITY_ROWS)
        .map(|mut v| { scrub_all(&mut v, sc); v }).collect()
}

fn activity(settings: &crate::settings::Settings) -> serde_json::Value {
    use crate::store::freshness::{fmt_utc, now_secs};
    let sc = crate::scrub::Scrubber::new(&settings.scrub);
    let now = now_secs();
    let store = crate::Store::open_readonly();
    let (runs, stale) = match &store {
        Ok(s) => (
            index_runs(s).map_err(|e| format!("{e:#}")),
            s.freshness_counts().map(|c| serde_json::json!({"changed": c.changed, "deleted": c.deleted, "new_pending": c.new_pending})).map_err(|e| format!("{e:#}")),
        ),
        Err(e) => (Err(format!("{e:#}")), Err(format!("{e:#}"))),
    };
    // Polled every 10 s: an oversized state file is reported, not parsed each time.
    let state_len = recall_home().and_then(|h| std::fs::metadata(h.join("refresh-state.json")).ok()).map_or(0, |m| m.len());
    let st = if state_len > MAX_STATE_BYTES { Default::default() } else { crate::refresh::RefreshState::load() };
    let next = crate::refresh::next_due(settings, st.last_ok_unix, now);
    let mut last_summary = serde_json::json!(st.last_summary);
    scrub_all(&mut last_summary, &sc);
    // daemon.json is written once at daemon start, so its mtime is the start time.
    let dj = recall_home().map(|h| h.join("daemon.json"));
    let started = dj.as_ref().and_then(|p| std::fs::metadata(p).ok()).and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs() as i64);
    let up = daemon_up().is_some();
    let errors = std::env::var("RECALL_ERROR_LOG").map(PathBuf::from).ok()
        .or_else(|| recall_home().map(|h| h.join("error-log.jsonl")))
        .map(|p| recent_errors(&p, &sc)).unwrap_or_default();
    let mut out = serde_json::json!({
        "now": fmt_utc(now),
        "refresh_state_error": (state_len > MAX_STATE_BYTES).then(|| format!("refresh-state.json is {state_len} bytes; not read")),
        "index_runs": runs.as_ref().ok(), "index_runs_error": runs.as_ref().err(),
        "stale": stale.as_ref().ok(), "stale_error": stale.as_ref().err(),
        "refresh": {
            "setting": settings.refresh.label(),
            "last_ok": st.last_ok_unix.map(fmt_utc),
            "last_attempt": st.last_attempt_unix.map(fmt_utc),
            "last_summary": last_summary,
            "pending_embed": st.pending_embed.len(),
            "next_due": next.map(fmt_utc),
            "next_due_in_s": next.map(|t| (t - now).max(0)),
        },
        "daemon": {
            "up": up,
            "started": if up { started.map(fmt_utc) } else { None },
            "uptime_s": if up { started.map(|t| (now - t).max(0)) } else { None },
            "query_count": serde_json::Value::Null,
        },
        "errors": errors,
        "not_recorded": NOT_RECORDED,
    });
    // Every string, error texts included: a db path can hold a custom term (e.g. a user name).
    scrub_all(&mut out, &sc);
    out
}

/// scrub.custom holds the very terms the user wants hidden, so only their count leaves.
fn shown(s: &crate::settings::Settings, key: &str) -> String {
    if key == "scrub.custom" {
        let n = s.scrub.custom.len();
        return format!("{n} term{} set (hidden)", if n == 1 { "" } else { "s" });
    }
    s.get(key).unwrap_or_default()
}

fn settings_view(s: &crate::settings::Settings, ctx: &Ctx) -> serde_json::Value {
    let rows: Vec<_> = crate::settings::KEYS
        .iter()
        .map(|k| serde_json::json!({ "key": k, "value": shown(s, k), "masked": *k == "scrub.custom" }))
        .collect();
    serde_json::json!({
        "settings": rows,
        // Not a settings.rs key: read-only here, changed only via RECALL_FUSION_MODE.
        "fusion": { "mode": crate::query::effective_fusion_mode(), "editable": false, "note": "env only (RECALL_FUSION_MODE)" },
        "file": crate::scrub::Scrubber::new(&s.scrub).scrub(&ctx.settings_path.display().to_string()).into_owned(),
    })
}

fn set_setting(req: &Request, ctx: &Ctx) -> Response {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&req.body) else {
        return err(400, "body must be JSON {\"key\":..,\"value\":..}");
    };
    let (Some(key), Some(value)) = (v["key"].as_str(), v["value"].as_str()) else {
        return err(400, "body must be JSON {\"key\":..,\"value\":..}");
    };
    let _guard = SETTINGS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    match crate::settings::Settings::set_in(&ctx.settings_path, key, value) {
        Ok(s) => json(200, serde_json::json!({ "key": key, "value": shown(&s, key) })),
        // settings.rs errors never echo a scrub.custom entry (see parse_custom); a write error can
        // name the file's path, so it is scrubbed like every other path.
        Err(e) => err(400, &crate::scrub::Scrubber::new(&crate::scrub::Config::load_from(&ctx.settings_path)).scrub(&format!("{e:#}"))),
    }
}

/// Every string in `v`, recursively: paths, `lacks` labels and notes can hold a custom term too.
fn scrub_all(v: &mut serde_json::Value, sc: &crate::scrub::Scrubber) {
    match v {
        serde_json::Value::String(s) => *s = sc.scrub(s).into_owned(),
        serde_json::Value::Array(a) => a.iter_mut().for_each(|x| scrub_all(x, sc)),
        serde_json::Value::Object(o) => o.values_mut().for_each(|x| scrub_all(x, sc)),
        _ => {}
    }
}

/// Clears QUERY_BUSY however run_query returns.
struct Busy;
impl Drop for Busy {
    fn drop(&mut self) {
        QUERY_BUSY.store(false, Ordering::SeqCst);
    }
}

fn run_query(req: &Request) -> Response {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&req.body) else {
        return err(400, "body must be JSON {\"q\":..}");
    };
    let q = v["q"].as_str().unwrap_or("").trim();
    if q.is_empty() || q.chars().count() > MAX_QUERY_CHARS || q.contains('\0') {
        return err(400, "q must be 1-2000 characters");
    }
    let top = v["top"].as_u64().unwrap_or(10).clamp(1, 50).to_string();
    // Optional channel list; the CLI validates names, this only keeps it to a plain comma list.
    let channels = v["channels"].as_str().unwrap_or("").trim();
    if channels.len() > 64 || !channels.bytes().all(|b| b.is_ascii_alphanumeric() || b == b',') {
        return err(400, "channels must be a comma list like fts5,bge");
    }
    if QUERY_BUSY.swap(true, Ordering::SeqCst) {
        return err(429, "a search is already running; wait for it to finish");
    }
    let _busy = Busy;
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => return err(500, &format!("current_exe: {e}")),
    };
    // A live daemon is REQUIRED when one answers ping: without --via-daemon the CLI silently falls
    // back to a cold in-process load (minutes on a full corpus) whenever the daemon refuses.
    let via_daemon = daemon_up().is_some();
    let mut cmd = Command::new(exe);
    // the Search footer shows "bge: model ready" (tidy_notes), which the CLI prints only when verbose
    cmd.env("RECALL_VERBOSE", "1");
    cmd.args(["query", "--json", "--top", &top]);
    if via_daemon {
        cmd.arg("--via-daemon");
    }
    if !channels.is_empty() {
        cmd.args(["--channels", channels]);
    }
    // `--` before user text: argv injection is the caller's job too (see Commands::Query).
    cmd.arg("--").arg(q).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let limit = if via_daemon { DAEMON_QUERY_TIMEOUT } else { INPROC_QUERY_TIMEOUT };
    let started = Instant::now();
    let (status, stdout, stderr) = match run_bounded(cmd, limit) {
        Ok(Some(r)) => r,
        Ok(None) => {
            let why = if via_daemon {
                "the daemon did not answer"
            } else {
                "no warm daemon is running, so recall cold-loads every vector for a semantic channel. \
                 Start it with `recall daemon up`, or search the fts5 channel only"
            };
            return err(504, &format!("search stopped after {} s: {why}", limit.as_secs()));
        }
        Err(e) => return err(500, &format!("could not run recall query: {e:#}")),
    };
    let sc = crate::scrub::Scrubber::new(&crate::scrub::Config::load());
    // RECALL_VERBOSE (set above for the bge footer) also prints the "closest:" reasons; the page shows
    // them only when the closest_labels setting is on, like the CLI (Codex cli-quality r1 #5). Filtered
    // BEFORE scrubbing: a custom scrub term "closest" would otherwise hide the prefix (r2 #3).
    let notes = without_closest_reasons(stderr.trim(), crate::query::closest_labels_on());
    let notes = sc.scrub(&notes).into_owned();
    let mut hits = serde_json::from_str::<serde_json::Value>(stdout.trim()).ok();
    if let Some(h) = hits.as_mut() {
        scrub_all(h, &sc);
    }
    let ok = status.success() && hits.is_some();
    let error = (!ok).then(|| {
        let tail: Vec<&str> = notes.lines().rev().take(6).collect::<Vec<_>>().into_iter().rev().collect();
        if tail.is_empty() { format!("recall query exited with {status}") } else { tail.join("\n") }
    });
    json(200, serde_json::json!({
        "ok": ok, "hits": hits, "notes": tidy_notes(&notes), "error": error,
        // the CLI's `closest_labels` setting, so the page shows the banner and chip exactly when the CLI does
        "closest_labels": crate::query::closest_labels_on(),
        "path": if via_daemon { "daemon" } else { "in-process" },
        "ms": started.elapsed().as_millis() as u64,
    }))
}

fn without_closest_reasons(notes: &str, labels: bool) -> String {
    if labels {
        return notes.to_string();
    }
    notes.lines().filter(|l| !l.trim_start().starts_with("closest:")).collect::<Vec<_>>().join("\n")
}

/// The child's stderr for the Search footer. The bge loader narrates its start-up (one line carries
/// the full onnxruntime path, `dll=...`); a web page shows that as one "bge: model ready". Every
/// other line, including bge warnings and errors ("no vectors stored", "ANN path error"), is kept.
fn tidy_notes(notes: &str) -> String {
    const CHATTER: [&str; 4] = ["loading vectors", "loaded ", "initializing ONNX", "backend="];
    let mut out: Vec<String> = Vec::new();
    let mut ready = false;
    for line in notes.lines() {
        match line.trim_start().strip_prefix("bge:").map(str::trim) {
            Some("model ready") => ready = true,
            Some(rest) if CHATTER.iter().any(|c| rest.starts_with(c)) => {}
            _ => out.push(hide_dll(line)),
        }
    }
    if ready {
        out.push("bge: model ready".into());
    }
    out.join("\n")
}

/// `dll=<path>` -> `dll=<hidden>`; the path ends at whitespace, `)`, `;` or `,` so the rest of the
/// line (the cause of an error, say) survives.
fn hide_dll(line: &str) -> String {
    let mut out = String::new();
    let mut rest = line;
    while let Some(i) = rest.find("dll=") {
        out.push_str(&rest[..i + 4]);
        let tail = &rest[i + 4..];
        let end = tail.find(|c: char| c.is_whitespace() || matches!(c, ')' | ';' | ',')).unwrap_or(tail.len());
        out.push_str("<hidden>");
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

// ── First-run setup ───────────────────────────────────────────────────────────────

/// Vendor/build folder names offered as extra excludes. Only names no note is likely to start
/// with: exclude patterns are segment PREFIXES, so `build` or `vendor` would also drop
/// `vendor-contracts.md`.
const VENDOR_DIRS: [&str; 5] = ["node_modules", "bower_components", "site-packages", "__pycache__", "venv"];
/// One budget for the whole scan (both rg runs and the disk walk share it).
const MAX_SCAN_ENTRIES: usize = 500_000;
const SCAN_DEADLINE: Duration = Duration::from_secs(60);
/// Child output lines are cut here before scrubbing; the rest of a longer line is dropped.
const MAX_LINE_BYTES: usize = 2000;

fn folder_arg(req: &Request) -> std::result::Result<PathBuf, Response> {
    let v: serde_json::Value = serde_json::from_slice(&req.body).map_err(|_| err(400, "body must be JSON {\"folder\":..}"))?;
    let f = v["folder"].as_str().unwrap_or("").trim().trim_matches('"');
    let p = PathBuf::from(f);
    if f.is_empty() || !p.is_dir() {
        return Err(err(400, "that is not an existing folder"));
    }
    Ok(std::path::absolute(&p).unwrap_or(p))
}

/// Pure tally of a file list (vault-relative, `/`-separated) against the rules `index` applies:
/// `.md` / `.qmd`, not excluded, not over [`crate::store::MAX_DOC_BYTES`]. Deterministic.
fn tally(files: &[(String, u64)], exclude: &crate::store::PathExcludeList) -> serde_json::Value {
    use std::collections::BTreeMap;
    let (mut will, mut bytes, mut other, mut too_large) = (0usize, 0u64, 0usize, 0usize);
    let mut excluded: BTreeMap<String, usize> = BTreeMap::new();
    let mut types: BTreeMap<String, usize> = BTreeMap::new();
    let mut vendor: BTreeMap<&str, usize> = BTreeMap::new();
    for (rel, size) in files {
        let ext = rel.rsplit_once('.').filter(|(_, e)| !e.contains('/')).map(|(_, e)| e.to_lowercase()).unwrap_or_default();
        *types.entry(if ext.is_empty() { "(none)".into() } else { format!(".{ext}") }).or_default() += 1;
        if ext != "md" && ext != "qmd" {
            other += 1;
            continue;
        }
        if let Some(p) = exclude.matching_pattern(rel) {
            *excluded.entry(p.to_string()).or_default() += 1;
            continue;
        }
        if *size as usize > crate::store::MAX_DOC_BYTES {
            too_large += 1;
            continue;
        }
        will += 1;
        bytes += size;
        for v in VENDOR_DIRS.iter().filter(|v| crate::store::segment_match(v, rel)) {
            *vendor.entry(v).or_default() += 1;
        }
    }
    let mut types: Vec<_> = types.into_iter().collect();
    types.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    types.truncate(12);
    serde_json::json!({
        "will_index": will, "will_index_mb": bytes as f64 / 1_048_576.0, "other_files": other,
        "too_large": too_large, "too_large_kb": crate::store::MAX_DOC_BYTES / 1024,
        "excluded_total": excluded.values().sum::<usize>(),
        "excluded": excluded.iter().map(|(p, n)| serde_json::json!({"pattern": p, "files": n})).collect::<Vec<_>>(),
        "vendor": vendor.iter().map(|(p, n)| serde_json::json!({"pattern": p, "files": n})).collect::<Vec<_>>(),
        "types": types.iter().map(|(e, n)| serde_json::json!({"ext": e, "files": n, "indexed": e == ".md" || e == ".qmd"})).collect::<Vec<_>>(),
    })
}

/// Read `r` to EOF, calling `f` with each line cut at [`MAX_LINE_BYTES`]: memory stays bounded
/// whatever the child prints, and the pipe is always drained.
fn bounded_lines(mut r: impl Read, mut f: impl FnMut(String)) {
    let (mut buf, mut line) = ([0u8; 8192], Vec::with_capacity(256));
    loop {
        let n = match r.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        for &b in &buf[..n] {
            if b == b'\n' {
                f(String::from_utf8_lossy(&line).trim_end().to_string());
                line.clear();
            } else if line.len() < MAX_LINE_BYTES {
                line.push(b);
            }
        }
    }
    if !line.is_empty() {
        f(String::from_utf8_lossy(&line).trim_end().to_string());
    }
}

/// Files from `rg <args> <root>`, vault-relative, with sizes for notes. Spends `budget` and stops
/// at `deadline` (killing rg; a stalled rg cannot hold the request). `Ok(None)` = rg missing;
/// `Err` = rg failed, so index would fall back to its own walk and this preview would not match.
fn rg_list(
    args: &[&str],
    root: &std::path::Path,
    deadline: Instant,
    budget: &mut usize,
    truncated: &mut bool,
) -> std::result::Result<Option<Vec<(String, u64)>>, String> {
    let Ok(mut child) = Command::new("rg").args(args).arg(root).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn() else {
        return Ok(None);
    };
    let (tx, rx) = std::sync::mpsc::sync_channel::<String>(1024);
    let out = child.stdout.take().expect("piped");
    std::thread::spawn(move || bounded_lines(out, |l| {
        let _ = tx.send(l);
    }));
    let root_norm = root.to_string_lossy().replace('\\', "/");
    let mut files = Vec::new();
    loop {
        if *budget == 0 {
            *truncated = true;
            break;
        }
        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(line) => {
                let abs = line.replace('\\', "/");
                let rel = abs.strip_prefix(root_norm.as_str()).map(|s| s.trim_start_matches('/')).unwrap_or(&abs).to_string();
                let size = if rel.ends_with(".md") || rel.ends_with(".qmd") { std::fs::metadata(&line).map_or(0, |m| m.len()) } else { 0 };
                files.push((rel, size));
                *budget -= 1;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                *truncated = true;
                break;
            }
        }
    }
    let _ = child.kill(); // this child's own handle; a no-op once rg has exited
    let status = child.wait().map_err(|e| e.to_string())?;
    // rg exits 1 for "no files"; any other unfinished run is a failure index would not share.
    if !*truncated && !status.success() && !(status.code() == Some(1) && files.is_empty()) {
        return Err(format!("rg failed ({status})"));
    }
    Ok(Some(files))
}

/// `.md` / `.qmd` on disk under `root`, skipping only dot-folders (what ignore rules are compared
/// against). Every entry spends `budget`; `None` = cut short, so no exact comparison is possible.
fn md_on_disk(root: &std::path::Path, deadline: Instant, budget: &mut usize) -> Option<usize> {
    let (mut n, mut stack) = (0usize, vec![root.to_path_buf()]);
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            if *budget == 0 || Instant::now() >= deadline {
                return None;
            }
            *budget -= 1;
            let name = e.file_name().to_string_lossy().to_lowercase();
            match e.file_type() {
                Ok(t) if t.is_dir() && !name.starts_with('.') => stack.push(e.path()),
                Ok(t) if t.is_file() && (name.ends_with(".md") || name.ends_with(".qmd")) => n += 1,
                _ => {}
            }
        }
    }
    Some(n)
}

/// Read-only preview of `index` on `folder`: the SAME `rg --files -g *.md -g *.qmd` index runs
/// (so .gitignore / .rgignore rules, a parent repo's included, apply exactly as they will), the
/// same exclude list and size cap, plus a plain count on disk so notes hidden by ignore rules
/// show up. Fails closed when rg fails, because index would then use a different walk. Writes nothing.
fn setup_scan(req: &Request) -> Response {
    let root = match folder_arg(req) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let sc = crate::scrub::Scrubber::new(&crate::scrub::Config::load());
    let (deadline, mut budget, mut truncated) = (Instant::now() + SCAN_DEADLINE, MAX_SCAN_ENTRIES, false);
    let mut files = match rg_list(&["--files", "-g", "*.md", "-g", "*.qmd"], &root, deadline, &mut budget, &mut truncated) {
        Ok(Some(f)) => f,
        Ok(None) => return err(400, &format!("ripgrep (rg) is not installed, so there is no preview. {}", crate::store::RG_MISSING)),
        Err(e) => {
            let msg = format!("{e}: no preview, because index would fall back to a walk that ignores .gitignore");
            return err(502, &sc.scrub(&msg));
        }
    };
    let notes_seen = files.len();
    // Other file types are informational only (index never reads them).
    if let Ok(Some(mut others)) = rg_list(&["--files", "-g", "!*.md", "-g", "!*.qmd"], &root, deadline, &mut budget, &mut truncated) {
        files.append(&mut others);
    }
    let mut out = tally(&files, &crate::store::PathExcludeList::load());
    let on_disk = if truncated { None } else { md_on_disk(&root, deadline, &mut budget) };
    out["notes_on_disk"] = on_disk.into();
    out["hidden_by_ignore"] = on_disk.map(|d| d.saturating_sub(notes_seen)).into();
    out["scanned"] = files.len().into();
    out["truncated"] = (truncated || on_disk.is_none()).into();
    out["folder"] = root.display().to_string().into();
    scrub_all(&mut out, &sc);
    json(200, out)
}

#[derive(Clone, serde::Serialize)]
struct JobStep {
    name: &'static str,
    state: &'static str,
    ms: Option<u64>,
}

#[derive(Clone, serde::Serialize)]
struct Job {
    state: &'static str,
    steps: Vec<JobStep>,
    lines: std::collections::VecDeque<String>,
    error: Option<String>,
}

static JOB: Mutex<Option<Job>> = Mutex::new(None);

fn job_view() -> serde_json::Value {
    match JOB.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
        Some(j) => serde_json::to_value(j).unwrap_or_default(),
        None => serde_json::json!({ "state": "idle", "steps": [], "lines": [] }),
    }
}

fn with_job(f: impl FnOnce(&mut Job)) {
    if let Some(j) = JOB.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
        f(j);
    }
}

/// Step 4: save `vault`, add the chosen vendor excludes, then run `index` and `embed` as children
/// of this exe, one after the other, streaming their output (scrubbed) into the job view.
fn setup_start(req: &Request, ctx: &Ctx) -> Response {
    let root = match folder_arg(req) {
        Ok(p) => p,
        Err(r) => return r,
    };
    // RECALL_VAULT outranks the setting; refuse rather than index a folder nobody picked.
    if let Some(env) = std::env::var("RECALL_VAULT").ok().filter(|e| !e.is_empty()) {
        if std::path::absolute(&env).map(|e| e != root).unwrap_or(true) {
            return err(409, "RECALL_VAULT is set and overrides this choice. Unset it, or pick that folder.");
        }
    }
    let skip: Vec<String> = serde_json::from_slice::<serde_json::Value>(&req.body)
        .ok()
        .and_then(|v| v["skip"].as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|s| s.as_str())
        .filter(|s| VENDOR_DIRS.contains(s))
        .map(String::from)
        .collect();
    {
        let mut slot = JOB.lock().unwrap_or_else(|p| p.into_inner());
        if slot.as_ref().is_some_and(|j| j.state == "running") {
            return err(409, "setup is already running");
        }
        let step = |name| JobStep { name, state: "waiting", ms: None };
        *slot = Some(Job { state: "running", steps: vec![step("Index notes"), step("Compute meaning vectors")], lines: Default::default(), error: None });
    }
    let fail = |status: u16, e: String| {
        let e = crate::scrub::Scrubber::new(&crate::scrub::Config::load()).scrub(&e).into_owned();
        with_job(|j| {
            j.state = "failed";
            j.error = Some(e.clone());
        });
        err(status, &e)
    };
    {
        let _guard = SETTINGS_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        if let Err(e) = crate::settings::Settings::set_in(&ctx.settings_path, "vault", &root.display().to_string()) {
            return fail(400, format!("{e:#}"));
        }
        if let Err(e) = add_excludes(&crate::store::exclude_file_path(), &skip) {
            return fail(500, format!("could not update exclude.txt: {e:#}"));
        }
    }
    // The root goes to the children explicitly, so a settings save mid-run cannot redirect them.
    std::thread::spawn(move || run_setup_job(root));
    json(200, job_view())
}

/// Append patterns to exclude.txt that are not already there (case-insensitive), via a temp file
/// renamed in place. Only a missing file starts empty; any other read error stops the change.
fn add_excludes(path: &std::path::Path, add: &[String]) -> Result<()> {
    let have = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
    };
    let known: std::collections::HashSet<String> =
        have.lines().map(|l| l.split('#').next().unwrap_or("").trim().to_lowercase()).collect();
    let new: Vec<&String> = add.iter().filter(|a| !known.contains(&a.to_lowercase())).collect();
    if new.is_empty() {
        return Ok(());
    }
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let mut text = have;
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str("# added by `recall ui` setup\n");
    for a in new {
        text.push_str(a);
        text.push('\n');
    }
    let tmp = path.with_extension(format!("txt.{}.tmp", std::process::id()));
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// ponytail: no timeout or cancel; indexing a large vault legitimately takes many minutes. The
// page shows live output, and closing `recall ui` ends the children with it only if the OS does.
fn run_setup_job(root: PathBuf) {
    let cfg = crate::scrub::Config::load();
    let sc = crate::scrub::Scrubber::new(&cfg);
    let steps: [&[&str]; 2] = [&["index"], &["embed", "--allow-benchmark-regression"]];
    for (i, args) in steps.iter().enumerate() {
        with_job(|j| j.steps[i].state = "running");
        let started = Instant::now();
        let ok = run_streaming(args, &root, &cfg);
        let ms = Some(started.elapsed().as_millis() as u64);
        let failed = ok.is_err();
        with_job(|j| {
            j.steps[i].ms = ms;
            j.steps[i].state = if failed { "failed" } else { "done" };
            if let Err(e) = &ok {
                j.state = "failed";
                j.error = Some(sc.scrub(e).into_owned());
            }
        });
        if failed {
            return;
        }
    }
    with_job(|j| j.state = "done");
}

/// Run this exe with `args` on `root`, pushing each output line (cut, scrubbed, last 14 kept)
/// into the job. ponytail: scrubbing is per line, so a multi-line secret (a PEM block) is only
/// caught line by line; index and embed print paths and counts, not file bodies.
fn run_streaming(args: &[&str], root: &std::path::Path, cfg: &crate::scrub::Config) -> std::result::Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut child = Command::new(exe)
        .args(args)
        .env("RECALL_VAULT", root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start recall {}: {e}", args[0]))?;
    let pipe = |r: Box<dyn Read + Send>, cfg: crate::scrub::Config| {
        std::thread::spawn(move || {
            let sc = crate::scrub::Scrubber::new(&cfg);
            bounded_lines(r, |line| {
                if line.trim().is_empty() {
                    return;
                }
                let line = sc.scrub(&line).into_owned();
                with_job(|j| {
                    j.lines.push_back(line);
                    while j.lines.len() > 14 {
                        j.lines.pop_front();
                    }
                });
            });
        })
    };
    let a = pipe(Box::new(child.stdout.take().expect("piped")), cfg.clone());
    let b = pipe(Box::new(child.stderr.take().expect("piped")), cfg.clone());
    let status = child.wait().map_err(|e| e.to_string())?;
    let _ = (a.join(), b.join());
    if status.success() {
        Ok(())
    } else {
        let last = JOB.lock().unwrap_or_else(|p| p.into_inner()).as_ref().and_then(|j| j.lines.back().cloned());
        Err(format!("recall {} exited with {status}{}", args[0], last.map(|l| format!(": {l}")).unwrap_or_default()))
    }
}

/// Run `cmd` for at most `limit`; `Ok(None)` = timed out (the child is killed and reaped).
/// Output is read on two threads so a full pipe cannot wedge the child, and capped.
fn run_bounded(mut cmd: Command, limit: Duration) -> Result<Option<(std::process::ExitStatus, String, String)>> {
    let mut child = cmd.spawn()?;
    let drain = |p: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(p) = p {
                let _ = p.take(MAX_CHILD_OUTPUT).read_to_end(&mut buf);
            }
            String::from_utf8_lossy(&buf).into_owned()
        })
    };
    let out = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let errs = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let deadline = Instant::now() + limit;
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break Some(s);
        }
        if Instant::now() >= deadline {
            let _ = child.kill(); // this child's own handle, i.e. by PID
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    let (stdout, stderr) = (out.join().unwrap_or_default(), errs.join().unwrap_or_default());
    Ok(status.map(|s| (s, stdout, stderr)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(tag: &str) -> Ctx {
        let dir = std::env::temp_dir().join(format!("recall-ui-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Ctx { port: 4321, token: "ab".repeat(32), settings_path: dir.join("settings.json") }
    }

    fn req(method: &str, target: &str, host: &str, token: Option<&str>, body: &str) -> Request {
        let (path, query) = target.split_once('?').unwrap_or((target, ""));
        let mut headers = vec![("Host".to_string(), host.to_string())];
        if let Some(t) = token {
            headers.push(("X-Recall-Token".into(), t.into()));
        }
        Request { method: method.into(), path: path.into(), query: query.into(), headers, body: body.as_bytes().to_vec() }
    }

    fn body(r: &Response) -> serde_json::Value {
        serde_json::from_slice(&r.body).unwrap()
    }

    /// One real request through `serve` on a loopback socket; returns the raw response.
    fn over_socket(ctx: Ctx, raw: &[u8]) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let a = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (b, _) = l.accept().unwrap();
        let t = std::thread::spawn(move || serve(b, &ctx));
        (&a).write_all(raw).unwrap();
        let mut buf = Vec::new();
        (&a).read_to_end(&mut buf).unwrap();
        t.join().unwrap();
        String::from_utf8_lossy(&buf).to_ascii_lowercase()
    }

    #[test]
    fn tidy_notes_hides_the_dll_path_and_keeps_warnings() {
        let raw = "bge: loading vectors from DB\nbge: loaded 5 vectors\nbge: initializing ONNX model\nbge: backend=cpu (requested, arena=off; dll=/opt/ort/libonnxruntime.so.1)\nbge: model ready\nbge: ANN path error (x); falling back to exact scan\nabstain: closest match\nsome: odd line dll=/opt/ort/x.so";
        let t = tidy_notes(raw);
        assert!(!t.contains("/opt/ort"), "{t}");
        assert_eq!(
            t,
            "bge: ANN path error (x); falling back to exact scan\nabstain: closest match\nsome: odd line dll=<hidden>\nbge: model ready"
        );
        assert_eq!(tidy_notes("bge: backend=cpu (dll=/x)"), "");
        assert_eq!(tidy_notes("bge: no vectors stored"), "bge: no vectors stored");
    }

    #[test]
    fn closest_reasons_reach_the_page_only_with_closest_labels_on() {
        let raw = "closest: no confident match (x); closest matches follow\nbge: model ready\nabstain: no confident match (y)";
        assert_eq!(without_closest_reasons(raw, false), "bge: model ready\nabstain: no confident match (y)");
        assert_eq!(without_closest_reasons(raw, true), raw);
    }

    #[test]
    fn query_page_never_calls_the_rrf_sum_the_ranking() {
        let html = include_str!("ui.html");
        // convex is the default fusion and a reranker can reorder, so an "rrf <n>" chip mislabels the order.
        assert!(!html.contains(r#""rrf " + Number(h.rrf)"#), "chip must not say rrf");
        assert!(html.contains(r#""rank-sum " + Number(h.rrf)"#));
    }

    #[test]
    fn hide_dll_keeps_the_text_around_the_path() {
        assert_eq!(hide_dll("bge: error dll=/x/y.so; permission denied"), "bge: error dll=<hidden>; permission denied");
        assert_eq!(hide_dll("a (dll=/p) b dll=/q"), "a (dll=<hidden>) b dll=<hidden>");
        assert_eq!(hide_dll("no path here"), "no path here");
    }

    #[test]
    fn token_is_required_on_the_page_and_every_api_route() {
        let c = ctx("tok");
        let h = "127.0.0.1:4321";
        let t = c.token.clone();
        assert_eq!(handle(&req("GET", "/", h, None, ""), &c).status, 403);
        assert_eq!(handle(&req("GET", "/?t=nope", h, None, ""), &c).status, 403);
        assert_eq!(handle(&req("GET", &format!("/?t={t}"), h, None, ""), &c).status, 200);
        for (m, p) in [("GET", "/api/settings"), ("GET", "/api/status"), ("GET", "/api/activity"), ("POST", "/api/settings"), ("POST", "/api/query")] {
            assert_eq!(handle(&req(m, p, h, None, ""), &c).status, 403, "{m} {p} without token");
            assert_eq!(handle(&req(m, p, h, Some(&t[1..]), ""), &c).status, 403, "{m} {p} short token");
            // the page's ?t= does not unlock the API: the header is required there
            assert_eq!(handle(&req(m, &format!("{p}?t={t}"), h, None, ""), &c).status, 403);
        }
        assert_eq!(handle(&req("GET", "/api/settings", h, Some(&t), ""), &c).status, 200);
        assert!(!same("abc", "abd") && same("abc", "abc") && !same("ab", "abc"));
    }

    #[test]
    fn bad_host_is_rejected_even_with_the_token() {
        let c = ctx("host");
        let t = c.token.clone();
        for h in ["evil.example:4321", "127.0.0.1:9999", "127.0.0.1", "localhost", "127.0.0.1:4321.evil.example", ""] {
            assert_eq!(handle(&req("GET", "/api/settings", h, Some(&t), ""), &c).status, 403, "host {h:?}");
        }
        assert_eq!(handle(&req("GET", "/api/settings", "LocalHost:4321", Some(&t), ""), &c).status, 200);
        let mut no_host = req("GET", "/api/settings", "", Some(&t), "");
        no_host.headers.retain(|(k, _)| k != "Host");
        assert_eq!(handle(&no_host, &c).status, 403);
    }

    #[test]
    fn a_tokenless_request_is_refused_before_its_body_is_read() {
        // Headers promise a body that never comes: the 403 must not wait for it (the 5 s read
        // timeout would make this slow), and a bogus Content-Length must not turn it into a 400.
        let raw = b"POST /api/settings HTTP/1.1\r\nHost: 127.0.0.1:4321\r\nContent-Length: 999999999\r\n\r\n";
        let started = Instant::now();
        let text = over_socket(ctx("early"), raw);
        assert!(text.starts_with("http/1.1 403"), "{text}");
        assert!(started.elapsed() < Duration::from_secs(3), "waited for the body");
    }

    #[test]
    fn a_client_trickling_headers_is_cut_off_at_the_request_deadline() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let a = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (b, _) = l.accept().unwrap();
        let started = Instant::now();
        let t = std::thread::spawn(move || serve(b, &ctx("trickle")));
        // one byte every 1 s: each read is well inside any per-read timeout
        for byte in b"GET /api/status HTTP/1.1\r\nHost: 127.0.0.1:4321\r\nX-Pad: aaaaaaaaaaaaaaaaaaaa" {
            if (&a).write_all(&[*byte]).is_err() || t.is_finished() {
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        t.join().unwrap();
        let took = started.elapsed();
        assert!(took >= REQUEST_DEADLINE && took < REQUEST_DEADLINE + Duration::from_secs(3), "{took:?}");
    }

    #[test]
    fn settings_round_trip_and_writes_are_post_only() {
        let c = ctx("rt");
        let (h, t) = ("127.0.0.1:4321", c.token.clone());
        let set = |k: &str, v: &str| handle(&req("POST", "/api/settings", h, Some(&t), &serde_json::json!({"key": k, "value": v}).to_string()), &c);
        assert_eq!(set("refresh", "every 2 hours").status, 200);
        assert_eq!(set("scrub.pii", "off").status, 200);
        let r = set("scrub.custom", r#"["Project Falcon","re:ACME-\\d{4}"]"#);
        assert_eq!(r.status, 200);
        assert!(!String::from_utf8_lossy(&r.body).contains("Falcon"), "custom terms never echoed");
        assert_eq!(set("refresh", "whenever").status, 400, "validated by settings.rs");
        assert_eq!(set("fusion", "rrf").status, 400, "not a settings key");
        let got = body(&handle(&req("GET", "/api/settings", h, Some(&t), ""), &c));
        let val = |k: &str| got["settings"].as_array().unwrap().iter().find(|r| r["key"] == k).unwrap()["value"].clone();
        assert_eq!(val("refresh"), "every 2 hours", "bad write left the good value");
        assert_eq!(val("scrub.pii"), "off");
        assert_eq!(val("scrub.custom"), "2 terms set (hidden)");
        assert!(!got.to_string().contains("Falcon"));
        assert_eq!(got["settings"].as_array().unwrap().len(), crate::settings::KEYS.len());
        assert_eq!(got["fusion"]["editable"], false);
        // the file the scrubber reads really holds the list, and `[]` clears it
        assert_eq!(crate::scrub::Config::load_from(&c.settings_path).custom.len(), 2);
        assert_eq!(set("scrub.custom", "[]").status, 200);
        assert!(crate::scrub::Config::load_from(&c.settings_path).custom.is_empty());
        let leftovers = std::fs::read_dir(c.settings_path.parent().unwrap()).unwrap().flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp")).count();
        assert_eq!(leftovers, 0, "atomic write leaves no temp file");
        // GET cannot write; unknown routes 404; bad JSON is a 400
        assert_eq!(handle(&req("GET", "/api/query", h, Some(&t), ""), &c).status, 405);
        assert_eq!(handle(&req("PUT", "/api/settings", h, Some(&t), ""), &c).status, 405);
        assert_eq!(handle(&req("GET", "/api/nope", h, Some(&t), ""), &c).status, 404);
        assert_eq!(handle(&req("POST", "/api/settings", h, Some(&t), "{"), &c).status, 400);
        assert_eq!(handle(&req("POST", "/api/query", h, Some(&t), r#"{"q":"  "}"#), &c).status, 400);
        let flag = r#"{"q":"x","channels":"fts5 --via-daemon"}"#;
        assert_eq!(handle(&req("POST", "/api/query", h, Some(&t), flag), &c).status, 400, "no argv smuggling");
        let _ = std::fs::remove_dir_all(c.settings_path.parent().unwrap());
    }

    #[test]
    fn every_string_in_a_hit_is_scrubbed_not_only_the_snippet() {
        let cfg = crate::scrub::Config { custom: vec!["FalconX".into()], ..Default::default() };
        let sc = crate::scrub::Scrubber::new(&cfg);
        let mut v = serde_json::json!([{ "vault_path": "FalconX/note.md", "lacks": ["FalconX"], "stale_note": "FalconX moved", "rank": 0 }]);
        scrub_all(&mut v, &sc);
        assert!(!v.to_string().contains("FalconX"), "{v}");
        assert_eq!(v[0]["rank"], 0);
    }

    #[test]
    fn recent_errors_are_newest_first_bounded_and_scrubbed() {
        let c = ctx("errlog");
        let dir = c.settings_path.parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        let p = dir.join("error-log.jsonl");
        let mut text = String::from("not json\n");
        for i in 0..15 {
            text.push_str(&format!("{{\"ts\":\"unix:{i}\",\"context\":\"main\",\"error\":\"boom {i} FalconX\"}}\n"));
        }
        std::fs::write(&p, text).unwrap();
        let sc = crate::scrub::Scrubber::new(&crate::scrub::Config { custom: vec!["FalconX".into()], ..Default::default() });
        let got = recent_errors(&p, &sc);
        assert_eq!(got.len(), ACTIVITY_ROWS);
        assert_eq!(got[0]["ts"], "unix:14", "newest first");
        assert!(!serde_json::json!(got).to_string().contains("FalconX"));
        assert!(recent_errors(&dir.join("absent.jsonl"), &sc).is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn scan_tally_counts_what_index_takes_and_why_the_rest_is_skipped() {
        let ex = crate::store::PathExcludeList::with_extra(&["archive"]);
        let f = |p: &str, n: u64| (p.to_string(), n);
        let files = vec![
            f("a.md", 100), f("notes/b.qmd", 50), f("Archive-2020/old.md", 9), f("disabled-x/y.md", 9),
            f("proj/node_modules/pkg/README.md", 7), f("img.png", 0), f("doc.PDF", 0), f("Makefile", 0),
        ];
        let t = tally(&files, &ex);
        assert_eq!(t["will_index"], 3, "{t}");
        assert_eq!(t["excluded_total"], 2);
        assert_eq!(t["other_files"], 3);
        assert_eq!(t["vendor"][0]["pattern"], "node_modules");
        assert!(t["excluded"].as_array().unwrap().iter().any(|x| x["pattern"] == "archive" && x["files"] == 1));
        assert!(t["types"].as_array().unwrap().iter().any(|x| x["ext"] == ".pdf" && x["indexed"] == false));
        assert_eq!(tally(&files, &ex), t, "deterministic");
        // index skips notes over MAX_DOC_BYTES; the preview must not promise them
        let big = vec![f("huge.md", crate::store::MAX_DOC_BYTES as u64 + 1), f("ok.md", 10)];
        let t = tally(&big, &ex);
        assert_eq!((t["will_index"].as_u64(), t["too_large"].as_u64()), (Some(1), Some(1)));
    }

    #[test]
    fn child_output_lines_are_cut_and_the_pipe_is_drained() {
        let mut input = vec![b'a'; MAX_LINE_BYTES * 3];
        input.extend_from_slice(b"\nshort\r\nlast");
        let mut got = Vec::new();
        bounded_lines(&input[..], |l| got.push(l));
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].len(), MAX_LINE_BYTES);
        assert_eq!((got[1].as_str(), got[2].as_str()), ("short", "last"));
    }

    #[test]
    fn add_excludes_appends_only_new_patterns() {
        let dir = std::env::temp_dir().join(format!("recall-ui-excl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let p = dir.join("exclude.txt");
        add_excludes(&p, &["node_modules".into()]).unwrap();
        add_excludes(&p, &["Node_Modules".into(), "venv".into()]).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert_eq!(text.matches("node_modules").count(), 1, "{text}");
        assert!(text.contains("venv"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn index_runs_reads_old_and_new_schemas() {
        let dir = std::env::temp_dir().join(format!("recall-ui-runs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // pre-enum_ms schema: the read-only opener does not migrate it
        let old = crate::Store::from_conn(rusqlite::Connection::open(dir.join("old.db")).unwrap());
        old.conn.execute_batch("CREATE TABLE index_runs (id INTEGER PRIMARY KEY, ran_at TEXT NOT NULL, enumerated INTEGER NOT NULL, used_fallback INTEGER NOT NULL, tombstoned INTEGER NOT NULL DEFAULT 0);
            INSERT INTO index_runs (ran_at, enumerated, used_fallback, tombstoned) VALUES ('2026-11-01T00:00:00Z', 10, 0, 1);").unwrap();
        let rows = index_runs(&old).unwrap();
        assert_eq!((rows[0]["files_seen"].as_i64(), rows[0]["enum_ms"].is_null()), (Some(10), true));
        let new = crate::Store::open_at(&dir.join("new.db")).unwrap();
        new.conn.execute_batch("INSERT INTO index_runs (ran_at, enumerated, used_fallback, tombstoned, enum_ms) VALUES ('a', 1, 0, 0, 5), ('b', 2, 1, 3, 7);").unwrap();
        let rows = index_runs(&new).unwrap();
        assert_eq!((rows[0]["ran_at"].as_str(), rows[0]["enum_ms"].as_i64(), rows[0]["walk_fallback"].as_bool()), (Some("b"), Some(7), Some(true)));
        drop((old, new));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_child_past_its_deadline_is_killed_and_reported() {
        #[cfg(windows)]
        let cmd = {
            let mut c = Command::new("powershell");
            c.args(["-NoProfile", "-Command", "Start-Sleep -Seconds 30"]);
            c
        };
        #[cfg(not(windows))]
        let cmd = {
            let mut c = Command::new("sleep");
            c.arg("30");
            c
        };
        let started = Instant::now();
        let mut cmd = cmd;
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        assert!(run_bounded(cmd, Duration::from_millis(500)).unwrap().is_none());
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn responses_carry_no_cors_headers() {
        let c = ctx("cors");
        let raw = format!("GET /api/settings HTTP/1.1\r\nHost: 127.0.0.1:4321\r\nX-Recall-Token: {}\r\n\r\n", c.token);
        let text = over_socket(c, raw.as_bytes());
        assert!(text.starts_with("http/1.1 200"), "{text}");
        assert!(!text.contains("access-control-"));
        assert!(text.contains("referrer-policy: no-referrer"));
    }
}
