// daemon.rs — Warm resident daemon: holds BGE model + vector matrix in RAM so
// queries skip the ~1.5s cold ONNX/fastembed load.
//
// Transport: loopback TCP on 127.0.0.1:0 (OS-assigned ephemeral port).
//   - Never binds 0.0.0.0 — loopback-only, zero network exposure.
//   - Protocol: NDJSON, one request + one response per connection.
//
// Protocol:
//   Request:  {"q":"text","k":5,"channels":["fts5","bge"]}
//         or  {"cmd":"ping"}
//         or  {"cmd":"shutdown"}
//   Response: {"ok":true,"hits":[{"id":1,"title":"","path":"","snippet":"","score":0.0}]}
//             `score` added 2026-08-21; a client reading an OLDER daemon tolerates its absence
//             via `#[serde(default)]` on `DaemonHit`, degrading to 0.0 rather than failing the
//             whole response.
//             `proto` (WIRE_PROTO) added 2026-10-30: a client refuses a hits reply without it and
//             answers in-process, because an older daemon has no abstain and no per-hit source.
//         or  {"ok":false,"err":"message"}
//         or  {"ok":true,"pong":true,"pid":N,"addr":"127.0.0.1:PORT","docs":N}
//
// Concurrency: the accept thread reads each request and answers ping / shutdown itself, so they
//   never wait behind a query. Queries go to QUERY_WORKERS threads, each with its own Store
//   connection, sharing the BGE matrix behind an RwLock (searches read it, a refresh patch writes
//   it). fastembed's TextEmbedding::embed needs &mut self, so the model sits behind a Mutex.
//
// State file: ~/.recall/daemon.json  → {"addr":"127.0.0.1:PORT","pid":N}
//   Written on startup, removed on clean shutdown.
//   Stale file (connect refused) → callers treat as absent.

use crate::bge::BgeSearch;
use crate::store::Store;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Duration;

// ── State-file path ───────────────────────────────────────────────────────────

/// Path to ~/.recall/daemon.json.
pub fn daemon_json_path() -> Result<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .context("USERPROFILE/HOME not set")?;
    let dir = PathBuf::from(home).join(".recall");
    std::fs::create_dir_all(&dir).context("create ~/.recall")?;
    Ok(dir.join("daemon.json"))
}

// ── State file I/O ────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct DaemonInfo {
    pub addr: String,
    pub pid: u32,
}

impl DaemonInfo {
    pub fn read() -> Option<Self> {
        let path = daemon_json_path().ok()?;
        let text = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&text).ok()
    }

    fn write(&self) -> Result<()> {
        let path = daemon_json_path()?;
        let json = serde_json::to_string(self).context("serialize DaemonInfo")?;
        std::fs::write(&path, json).context("write daemon.json")?;
        Ok(())
    }

    fn remove() {
        if let Ok(p) = daemon_json_path() {
            let _ = std::fs::remove_file(p);
        }
    }
}

// ── Wire protocol ─────────────────────────────────────────────────────────────

/// Version of the query semantics a hits reply carries. 2 = abstain rules + per-hit `source`
/// (2026-10-29). A daemon is long-lived and survives rebuilds, so a new client routinely meets an
/// old one; without this the client silently served the old daemon's answers (nonsense queries
/// got 3 hits, `source: "unknown"`, exit 0 — LOOP-R1-LEGACY-DAEMON-NEWCLIENT).
/// 3 = loop2 rules (top-5 lexical judge, unknown-term veto, number tokens never evidence,
/// exclusion-aware df, no ungated override). Any other value is refused, so a resident round-2
/// daemon can never serve round-2 answers to this client (Codex r3 P1).
/// 4 = loop2 round 2: numbers constrain the supporting doc, a typo's correction must be in it,
/// bge support counts only candidates that themselves support the query, searchable-only df.
/// 5 = loop2 round 3: long-query share floor, every returned hit satisfies the number/typo
/// constraints, 2-digit numbers constrain, exact (budget-free) searchable df, gated fts5 rescue.
/// 6 = v3: a short query's supporting doc must hold its subject (rarest findable term).
/// 7 = v3 provenance: every hit carries `ranks`, `scores` and `rrf` (REQUIRED on the wire).
/// 8 = v3 query cap: the fts5 RANKING matches up to 256 tokens (judge unchanged at 64).
/// 9 = v3 r1 Codex fixes: 4 findable terms need all 4 (5 need 4), graph answers to the lexical
/// rule, `top` is a hard limit, a long query's returned set must hold a supporting hit.
/// 10 = v3 r2: a short query's answer may lack only common terms (df > 200), and paths that
/// differ only in case are one result.
/// 11 = v3 r3: a keyword query's supporting hit holds every subject term (fixed generic-word
/// list, no df), the returned set must hold a supporting hit for every query, case duplicates keep
/// the approved copy, and hits carry `lacks` / `merged`; bge's instruction is for questions only.
/// 12 = v3 r4: a hit lacking a subject term is LABELLED (`lacks`), never refused for it; the fts5
/// rescue's judgement makes its returned set hold a supporting hit; bge embeds per 64-row slice.
/// 13 = v3 r5: every channel set is judged, so rg / qwen3 / gemini / voyage hits carry `lacks` and
/// those channels abstain under the lexical rule like the rest.
/// 14 = v3 r6: two-character subjects ("AI", "qx") are judged and labelled; only a fixed list of
/// two-letter function words is dropped.
/// 15 = v3 r7: one-letter tokens and capitalised acronyms ("R", "IT") are labelled subjects too.
/// 16 = v3 r8: capitalised generic words ("MAKE") are labelled; label subjects are filtered before
/// the 64 cap, and hits carry `labels_truncated` when it bites.
/// 17 = lme-improve: a question that fails an abstain rule may be answered as a closest match;
/// such hits carry `closest_match`.
/// 18 = abstain-calibrated: any query form may be a closest match when it beats the corpus's
/// phantom null; a number no document holds is labelled instead of emptying a closest list.
/// 19 = tenancy v1: a query carries the client's `agent` / `project` / `team`, and the daemon scopes
/// every read to it. A proto-18 daemon would drop those fields and answer as the operator, which is
/// why this client refuses any other version.
/// 20 = answerability-judge: a query that fails the judge is answered with labelled hits by default
/// (RECALL_ABSTAIN_POLICY=label); RECALL_JUDGE=haiku may label a confident set.
/// 21 = cli-quality: one weak-list decision (the "every hit lacks a subject" case folds into closest_match),
/// the pre-registered `agree` override, and `absent` (words no note holds) per hit.
pub const WIRE_PROTO: u64 = 21;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Request {
    Cmd {
        cmd: String,
    },
    Query {
        q: String,
        k: usize,
        channels: Vec<String>,
        /// Tenancy (proto 19): the CLIENT's identity, re-validated here with `scope::resolve`.
        /// Absent = operator, exactly what a pre-19 client means.
        #[serde(default)]
        agent: Option<String>,
        #[serde(default)]
        project: Option<String>,
        #[serde(default)]
        team: Option<String>,
    },
}

#[derive(Debug, Serialize)]
struct HitWire {
    id: i64,
    title: String,
    path: String,
    snippet: String,
    /// 🔴 ADDED 2026-08-21. Its absence was a silent fail-open.
    ///
    /// `fused()` computes a real score, and the daemon used to drop it here, so every result
    /// served by the daemon reported `score: 0.0` while the identical query run in-process
    /// reported the real BM25 value (measured: 37.193 vs 0.0). The daemon is the DEFAULT path,
    /// so effectively every consumer saw zero — and any downstream threshold, cutoff or
    /// score-weighted filter was silently inert rather than erroring.
    score: f64,
    /// Channel(s) that ranked this hit — see `query::source_label`.
    source: String,
    /// Per-channel 0-based rank, raw score and RRF value — see `query::Provenance`.
    ranks: BTreeMap<String, usize>,
    scores: BTreeMap<String, f64>,
    rrf: f64,
    /// `query::Provenance::lacks` / `merged`; omitted when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    lacks: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    merged: Vec<i64>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    labels_truncated: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    closest_match: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    absent: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Response {
    Hits {
        ok: bool,
        proto: u64,
        hits: Vec<HitWire>,
    },
    Pong {
        ok: bool,
        pong: bool,
        pid: u32,
        addr: String,
        docs: i64,
    },
    Error {
        ok: bool,
        err: String,
    },
}

impl Response {
    fn err(msg: impl Into<String>) -> Self {
        Response::Error {
            ok: false,
            err: msg.into(),
        }
    }
}

/// Max seconds to wait for a client to send its one-line request before the
/// accept loop gives up and moves on (prevents a stalled client wedging the daemon).
const CLIENT_READ_TIMEOUT_SECS: u64 = 5;

/// Max seconds the CLI waits for a query answer before answering in-process. A live-vault query
/// takes 1-7 s (an OR over every question word ranks most of the corpus), and the in-process
/// fallback costs ~20 s of model load on top, so giving up at 5 s only multiplied the load: 101
/// back-to-back --top 50 queries fell back one after another while the daemon still ran each
/// abandoned request (2026-11-06).
pub const QUERY_READ_TIMEOUT_SECS: u64 = 30;

/// Max seconds a client will block establishing the TCP connection. A port that
/// is LISTEN-but-not-accepting (e.g. a zombie daemon) would otherwise hang a
/// bare `TcpStream::connect` indefinitely on the CLI fast-path.
const CONNECT_TIMEOUT_SECS: u64 = 2;

/// Upper bound on `k` accepted from an (untrusted) client request, to stop a
/// caller forcing every channel to allocate/rank millions of candidates. Shared
/// with the CLI in-process path (`cmd_query`) so both entry points clamp alike.
pub const MAX_QUERY_K: usize = 500;

/// Hard cap on bytes read for a single NDJSON request line. `read_line` grows its
/// buffer until a newline or EOF; the read *timeout* bounds time but NOT memory,
/// so a client streaming megabytes without a `\n` would balloon the heap before
/// the timeout fires. One real request is well under a KB, so 1 MiB is generous.
const MAX_REQUEST_BYTES: u64 = 1 << 20;

// ── Main accept loop ──────────────────────────────────────────────────────────

/// Start the warm daemon: bind, load model, write state file, accept loop.
/// This function blocks until a `{"cmd":"shutdown"}` request arrives.
pub fn run_daemon() -> Result<()> {
    // Bind to an OS-assigned ephemeral port on loopback only.
    let listener = TcpListener::bind("127.0.0.1:0").context("bind TCP listener on 127.0.0.1:0")?;
    let addr = listener.local_addr().context("get local addr")?;
    let addr_str = addr.to_string();
    let pid = std::process::id();

    // Eagerly load Store + BGE (the expensive warm state).
    eprintln!("recall daemon: opening store…");
    let store = Store::open().context("Store::open in daemon")?;
    eprintln!("recall daemon: loading BGE model + vectors…");
    // Preflight-or-refuse: no DLL (or any load error) stops the boot, with the cause in the message.
    let bge = BgeSearch::load(&store)
        .map_err(|e| anyhow::anyhow!("daemon refuses to start, BgeSearch::load failed: {e:#}"))?;
    eprintln!("recall daemon: BGE ready (loaded={})", bge.is_some());
    let docs = store.count_valid().unwrap_or(0);

    // Write the advertised address and PID so clients can discover us.
    let info = DaemonInfo {
        addr: addr_str.clone(),
        pid,
    };
    info.write()?;

    println!(
        "recall daemon up: {} (pid={}, docs={})",
        addr_str, pid, docs
    );

    // Accept loop — single-threaded, one connection at a time (RefCell safety).
    // File-change watcher + scheduled refresh (store/freshness.rs, refresh.rs). It owns its own
    // connection and reports back the doc ids it re-embedded so this thread can patch the matrix.
    // RECALL_NO_WATCHER=1 opts out; a daemon without it still serves queries exactly as before.
    let reload_rx = if std::env::var_os("RECALL_NO_WATCHER").is_some() {
        None
    } else {
        Some(crate::refresh::spawn_worker())
    };
    let bge: SharedBge = std::sync::Arc::new(Gated::new(bge));
    let result = accept_loop(&listener, &store, bge, reload_rx, Store::open);

    // Clean shutdown: remove state file whether loop exited cleanly or with error.
    DaemonInfo::remove();
    result
}

/// How long the calibrator waits between attempts to open a store that would not open.
const CALIBRATOR_OPEN_RETRY: Duration =
    if cfg!(test) { Duration::from_millis(100) } else { Duration::from_secs(30) };

/// The background abstain calibrator: computes and stores the phantom null for each reader the
/// request thread queued ([`crate::query::defer_calibration_to`]), on its own connection, so a
/// fresh calibration never holds up other clients. A reader queued twice costs one lookup the
/// second time.
fn spawn_calibrator(
    open: impl Fn() -> Result<Store> + Send + 'static,
) -> crate::query::CalibrationQueue {
    let (queue, rx) = crate::query::CalibrationQueue::new();
    // Only the state, never a sender: when the request thread's queue goes, `rx` disconnects and
    // this thread ends, releasing its connection (Codex calib r2).
    let status = queue.status();
    let spawned = std::thread::Builder::new().name("recall-calibrate".into()).spawn(move || {
        // A store that cannot be opened right now (locked during a big refresh) is retried until
        // it opens or the daemon goes away (Codex calib r1 #2). Views queued meanwhile stay
        // pending in a backlog and are calibrated as soon as it opens (Codex calib r3).
        let mut backlog = Vec::new();
        let store = loop {
            match open() {
                Ok(s) => break s,
                Err(e) => {
                    eprintln!("recall calibrator: open store (retrying in {CALIBRATOR_OPEN_RETRY:?}): {e:#}");
                    match rx.recv_timeout(CALIBRATOR_OPEN_RETRY) {
                        Ok(reader) => backlog.push(reader),
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
                    }
                }
            }
        };
        // A refresh writing at the same moment must delay the stored row, not lose it.
        store.conn.busy_timeout(Duration::from_secs(30)).ok();
        for reader in backlog.into_iter().chain(rx.iter()) {
            crate::scope::set(reader.clone());
            let t = std::time::Instant::now();
            if let Err(e) = crate::query::phantom_null(&store) {
                eprintln!("recall calibrator: {e:#}");
            }
            if t.elapsed() > Duration::from_secs(1) {
                eprintln!("recall calibrator: calibrated a view in {:.0?}", t.elapsed());
            }
            let stored = crate::query::phantom_is_stored(&store).unwrap_or(false);
            status.finished(&reader, stored);
        }
    });
    if let Err(e) = spawned {
        // The queue then reports itself not running on first use; queries stay strict, never inline.
        eprintln!("recall daemon: could not start the abstain calibrator: {e}");
    }
    queue
}

/// The BGE channel, shared by the query workers (read) and the refresh patch (write).
type SharedBge = std::sync::Arc<Gated<Option<BgeSearch>>>;

/// Queries served at once. Ping and shutdown never count against it (the accept thread answers
/// them). ponytail: a fixed pool; each worker holds a SQLite connection, and the BGE scan already
/// fans out over up to 8 threads per query, so more workers mostly add contention.
const QUERY_WORKERS: usize = 4;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Queries a worker began, and queries dropped unrun because their client had already gone.
static QUERIES_STARTED: AtomicU64 = AtomicU64::new(0);
static QUERIES_DROPPED: AtomicU64 = AtomicU64::new(0);

/// The longest one query may run before it is cancelled. Longer than any client waits (the CLI
/// gives up at [`QUERY_READ_TIMEOUT_SECS`]), so it only ever cuts work nobody can receive.
const QUERY_BUDGET: Duration =
    if cfg!(test) { Duration::from_millis(1000) } else { Duration::from_secs(60) };

/// Bytes already waiting after the request line (peeked, 1 ms wait)?
fn trailing_input(s: &std::net::TcpStream) -> bool {
    let _ = s.set_read_timeout(Some(Duration::from_millis(1)));
    matches!(s.peek(&mut [0u8; 1]), Ok(n) if n > 0)
}

/// Has the client closed its end? One non-consuming peek with a 1 ms wait: EOF or a reset means
/// gone; no data within the wait means it is still waiting for its answer. The read timeout is a
/// property of the socket, but nothing reads this socket after the request line.
///
/// Protocol: a client sends ONE request line and keeps its socket open until the answer arrives.
/// A half-close (`shutdown(Write)`) reads exactly like a close and the request is dropped; the
/// recall CLI never half-closes, and every other client shells out to it.
///
/// Data counts as gone too: the protocol allows nothing after the request line, and a byte sitting
/// unread would make every later peek report "data" and hide the close behind it (Codex conc r2 #2).
fn client_gone(s: &std::net::TcpStream) -> bool {
    let _ = s.set_read_timeout(Some(Duration::from_millis(1)));
    match s.peek(&mut [0u8; 1]) {
        Ok(_) => true, // EOF, or input the protocol forbids
        Err(e) => !matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut),
    }
}

/// Cancels the query in hand when its client disconnects, it outruns [`QUERY_BUDGET`] or the daemon
/// is shutting down. Once cancelled it interrupts the connection on EVERY tick until the query
/// returns: `sqlite3_interrupt` only stops statements running at that instant, so a single call
/// that lands between two statements would be lost (Codex conc r1 #1). The worker then answers
/// with the reason instead of results.
struct Watchdog {
    done: std::sync::Arc<AtomicBool>,
    why: std::sync::Arc<std::sync::Mutex<Option<&'static str>>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Watchdog {
    fn start(store: &Store, client: &std::net::TcpStream, shutdown: &std::sync::Arc<AtomicBool>) -> Self {
        let done = std::sync::Arc::new(AtomicBool::new(false));
        let cancelled = std::sync::Arc::new(AtomicBool::new(false));
        let why = std::sync::Arc::new(std::sync::Mutex::new(None));
        crate::query::prof::set_cancel(Some(cancelled.clone()));
        // Checked by SQLite every 1,000 VM steps of every statement on this connection, so a
        // cancellation reaches the next statement too, not only the one running at that instant.
        let flag = cancelled.clone();
        store.conn.progress_handler(1000, Some(move || flag.load(Ordering::SeqCst)));
        let interrupt = store.conn.get_interrupt_handle();
        let thread = client.try_clone().ok().and_then(|client| {
            let (done, cancelled, why, shutdown) = (done.clone(), cancelled.clone(), why.clone(), shutdown.clone());
            std::thread::Builder::new()
                .name("recall-query-watch".into())
                .spawn(move || {
                    let t = std::time::Instant::now();
                    loop {
                        // finish() unparks this thread, so a query that ends never waits out the tick
                        std::thread::park_timeout(Duration::from_millis(50));
                        if done.load(Ordering::SeqCst) {
                            return;
                        }
                        if !cancelled.load(Ordering::SeqCst) {
                            let reason = if shutdown.load(Ordering::SeqCst) {
                                Some("the daemon is shutting down")
                            } else if t.elapsed() > QUERY_BUDGET {
                                Some("it ran past its time budget")
                            } else if client_gone(&client) {
                                Some("the client disconnected or sent data after its request")
                            } else {
                                None
                            };
                            let Some(r) = reason else { continue };
                            *why.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
                            cancelled.store(true, Ordering::SeqCst);
                        }
                        interrupt.interrupt();
                    }
                })
                .ok()
        });
        Watchdog { done, why, thread }
    }

    /// Stop watching (joined, so no interrupt can reach the connection's next query); the reason
    /// the query was cancelled, if it was.
    fn finish(mut self, store: &Store) -> Option<&'static str> {
        self.done.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            t.thread().unpark();
            let _ = t.join();
        }
        store.conn.progress_handler(0, None::<fn() -> bool>);
        crate::query::prof::set_cancel(None);
        *self.why.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The BGE matrix behind a turnstile: readers pass `gate` on the way in, and a writer holds it
/// while it waits for the write lock, so a stream of overlapping queries cannot starve a refresh
/// patch (Codex conc r1 #2). Readers hold the lock for their whole query; a writer waits for the
/// queries already running, never for ones that arrive after it.
#[derive(Default)]
pub(crate) struct Gated<T> {
    gate: std::sync::Mutex<()>,
    lock: std::sync::RwLock<T>,
}

impl<T> Gated<T> {
    pub(crate) fn new(v: T) -> Self {
        Self { gate: std::sync::Mutex::new(()), lock: std::sync::RwLock::new(v) }
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, T> {
        let _turn = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.lock.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, T> {
        let _turn = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        self.lock.write().unwrap_or_else(|e| e.into_inner())
    }
}

/// Swap refreshed documents' vectors into the shared matrix, on a thread of its own: it waits for
/// the queries in flight (see [`Gated`]) instead of hoping for a gap between arrivals. A read that
/// fails (a busy database) keeps the ids and retries, which would otherwise pin the old vectors
/// until restart.
fn spawn_reloader(
    rx: std::sync::mpsc::Receiver<Vec<i64>>,
    open: std::sync::Arc<dyn Fn() -> Result<Store> + Send + Sync>,
    bge: SharedBge,
) {
    let spawned = std::thread::Builder::new().name("recall-reload".into()).spawn(move || {
        let mut store = None;
        let mut pending: Vec<i64> = Vec::new();
        for ids in rx.iter() {
            pending.extend(ids);
            pending.extend(rx.try_iter().flatten());
            pending.sort_unstable();
            pending.dedup();
            loop {
                if store.is_none() {
                    store = open().map_err(|e| eprintln!("recall daemon: reload store: {e:#}")).ok();
                }
                match store.as_ref().map(|s| s.load_chunk_embeddings_for_docs(&pending)) {
                    Some(Ok(fresh)) => {
                        if let Some(b) = bge.write().as_mut() {
                            b.patch_docs(&pending, fresh);
                            eprintln!("recall daemon: patched vectors for {} refreshed doc(s)", pending.len());
                        }
                        pending.clear();
                        break;
                    }
                    Some(Err(e)) => eprintln!("recall daemon: could not patch refreshed vectors (will retry): {e:#}"),
                    None => {}
                }
                std::thread::sleep(Duration::from_secs(5));
            }
        }
    });
    if let Err(e) = spawned {
        eprintln!("recall daemon: could not start the vector reloader (refreshed notes keep old vectors): {e}");
    }
}

/// One query handed from the accept thread to a worker: where to answer, and what was asked.
struct Job {
    out: std::net::TcpStream,
    q: String,
    k: usize,
    channels: Vec<String>,
    who: (Option<String>, Option<String>, Option<String>),
}

/// Start the query workers. Each opens its own connection and defers calibration to `calib`;
/// they exit when the job sender is dropped, after finishing the query in hand.
fn spawn_workers(
    n: usize,
    open: std::sync::Arc<dyn Fn() -> Result<Store> + Send + Sync>,
    bge: &SharedBge,
    calib: &crate::query::CalibrationQueue,
    shutdown: &std::sync::Arc<AtomicBool>,
) -> (std::sync::mpsc::Sender<Job>, Vec<std::thread::JoinHandle<()>>) {
    let (tx, rx) = std::sync::mpsc::channel::<Job>();
    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
    let handles = (0..n)
        .filter_map(|i| {
            let (rx, open, bge, calib, shutdown) =
                (rx.clone(), open.clone(), bge.clone(), calib.clone(), shutdown.clone());
            let spawned = std::thread::Builder::new().name(format!("recall-query-{i}")).spawn(move || {
                // A worker without a store still answers, with the error, so no client hangs.
                let store = open().map_err(|e| format!("query worker could not open the store: {e:#}"));
                let _defer = crate::query::defer_calibration_to(calib);
                loop {
                    let job = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                    let Ok(mut job) = job else { return };
                    // A client that gave up while its request sat in the queue gets nothing: running
                    // it anyway is how 101 timed-out --top 50 queries wedged the daemon (2026-11-06).
                    // After shutdown the queue is discarded, not drained (Codex conc r1 #3).
                    if shutdown.load(Ordering::SeqCst) || client_gone(&job.out) {
                        QUERIES_DROPPED.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    QUERIES_STARTED.fetch_add(1, Ordering::Relaxed);
                    let resp = match &store {
                        Ok(store) => {
                            let watch = Watchdog::start(store, &job.out, &shutdown);
                            let bge = bge.read();
                            let resp = handle_scoped_query(store, &bge, &job.q, job.k, &job.channels, job.who);
                            match watch.finish(store) {
                                Some(why) => Response::err(format!("recall daemon: query cancelled: {why}")),
                                None => resp,
                            }
                        }
                        Err(e) => Response::err(e.clone()),
                    };
                    let _ = writeln!(job.out, "{}", serde_json::to_string(&resp).unwrap());
                }
            });
            match spawned {
                Ok(h) => Some(h),
                Err(e) => {
                    eprintln!("recall daemon: could not start query worker {i}: {e}");
                    None
                }
            }
        })
        .collect();
    (tx, handles)
}

fn accept_loop(
    listener: &TcpListener,
    store: &Store,
    bge: SharedBge,
    reload_rx: Option<std::sync::mpsc::Receiver<Vec<i64>>>,
    open_store: impl Fn() -> Result<Store> + Send + Sync + 'static,
) -> Result<()> {
    let open: std::sync::Arc<dyn Fn() -> Result<Store> + Send + Sync> = std::sync::Arc::new(open_store);
    // Calibration never runs on a request thread: one fresh null took 8.5 min of CPU on the live vault.
    let calib = spawn_calibrator({
        let open = open.clone();
        move || open()
    });
    if let Some(rx) = reload_rx {
        spawn_reloader(rx, open.clone(), bge.clone());
    }
    let shutdown = std::sync::Arc::new(AtomicBool::new(false));
    let (jobs, workers) = spawn_workers(QUERY_WORKERS, open, &bge, &calib, &shutdown);
    anyhow::ensure!(!workers.is_empty(), "recall daemon: no query worker could start");
    drop(calib); // the workers hold the queue now; the calibrator exits once they do
    let result = serve(listener, store, &jobs);
    // Running queries are cancelled and queued ones discarded, so this returns in about a tick.
    shutdown.store(true, Ordering::SeqCst);
    drop(jobs);
    for w in workers {
        let _ = w.join();
    }
    result
}

/// The accept thread: read each request, answer ping / shutdown here, hand queries to a worker.
fn serve(listener: &TcpListener, store: &Store, jobs: &std::sync::mpsc::Sender<Job>) -> Result<()> {
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("recall daemon: accept error: {}", e);
                continue;
            }
        };

        let peer = stream
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_default();
        // Freeze guard: a client that connects but never sends a newline would
        // otherwise block this single-threaded accept loop forever. Cap the read.
        let _ = stream.set_read_timeout(Some(Duration::from_secs(CLIENT_READ_TIMEOUT_SECS)));
        let mut write_stream = match stream.try_clone() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("recall daemon: clone stream for {}: {}", peer, e);
                continue;
            }
        };
        // A client that connects but stops reading the response would wedge this
        // single-threaded accept loop on `writeln!` forever — bound the write too.
        let _ = write_stream.set_write_timeout(Some(Duration::from_secs(CLIENT_READ_TIMEOUT_SECS)));
        // Cap the request at MAX_REQUEST_BYTES: `read_line` otherwise buffers an
        // unbounded no-newline stream into RAM until the read timeout fires. Take
        // EOFs the reader at the cap, so an oversized request degrades to a JSON
        // parse error rather than an OOM.
        let mut reader = BufReader::new(stream.take(MAX_REQUEST_BYTES));

        // Read exactly one line (one NDJSON request).
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => continue, // empty connection
            Ok(_) => {}
            Err(e) => {
                eprintln!("recall daemon: read from {}: {}", peer, e);
                continue;
            }
        }
        let line = line.trim_end().to_string();
        // One request line, then the client waits: anything after it is a protocol error, and
        // refused here so a later peek ([`client_gone`]) can only ever see "nothing yet" or the
        // close (Codex conc r1 #4: an unread trailing byte would hide a real disconnect).
        if !reader.buffer().is_empty() || trailing_input(&write_stream) {
            let resp = Response::err("unexpected data after the request line (one request per connection)");
            let _ = writeln!(write_stream, "{}", serde_json::to_string(&resp).unwrap());
            continue;
        }

        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = Response::err(format!("bad request JSON: {}", e));
                let _ = writeln!(write_stream, "{}", serde_json::to_string(&resp).unwrap());
                continue;
            }
        };

        match req {
            Request::Cmd { cmd } if cmd == "ping" => {
                let addr_str = listener
                    .local_addr()
                    .map(|a| a.to_string())
                    .unwrap_or_default();
                let docs = store.count_valid().unwrap_or(0);
                let resp = Response::Pong {
                    ok: true,
                    pong: true,
                    pid: std::process::id(),
                    addr: addr_str,
                    docs,
                };
                let _ = writeln!(write_stream, "{}", serde_json::to_string(&resp).unwrap());
            }
            Request::Cmd { cmd } if cmd == "shutdown" => {
                let resp = Response::Hits {
                    ok: true,
                    proto: WIRE_PROTO,
                    hits: vec![],
                };
                let _ = writeln!(write_stream, "{}", serde_json::to_string(&resp).unwrap());
                return Ok(()); // exit accept loop → caller removes daemon.json
            }
            Request::Cmd { cmd } => {
                let resp = Response::err(format!("unknown cmd: {}", cmd));
                let _ = writeln!(write_stream, "{}", serde_json::to_string(&resp).unwrap());
            }
            Request::Query { q, k, channels, agent, project, team } => {
                let job = Job { out: write_stream, q, k, channels, who: (agent, project, team) };
                if let Err(std::sync::mpsc::SendError(mut job)) = jobs.send(job) {
                    let resp = Response::err("recall daemon: no query worker is running");
                    let _ = writeln!(job.out, "{}", serde_json::to_string(&resp).unwrap());
                }
            }
        }
    }
    Ok(())
}

/// One reader per request, restored to the operator afterwards: the daemon itself always runs as
/// the operator (`daemon up` refuses otherwise), so a request can never inherit the previous
/// caller's identity. A bad identity is an error reply, never an operator answer.
fn handle_scoped_query(
    store: &Store,
    bge: &Option<BgeSearch>,
    q: &str,
    k: usize,
    channels: &[String],
    (agent, project, team): (Option<String>, Option<String>, Option<String>),
) -> Response {
    match crate::scope::resolve(agent, project, team) {
        Ok(reader) => {
            crate::scope::set(reader);
            let r = handle_query(store, bge, q, k, channels);
            crate::scope::set(crate::scope::Reader::Operator);
            r
        }
        Err(e) => Response::err(e.to_string()),
    }
}

fn handle_query(
    store: &Store,
    bge: &Option<BgeSearch>,
    q: &str,
    k: usize,
    channels: &[String],
) -> Response {
    #[cfg(test)]
    {
        // a slow query that, like a SQLite statement, stops when the watchdog cancels it
        let until = std::time::Instant::now() + Duration::from_millis(tests::QUERY_DELAY_MS.load(Ordering::SeqCst));
        let cancelled = crate::query::prof::cancelled;
        while std::time::Instant::now() < until && !cancelled() {
            std::thread::sleep(Duration::from_millis(10));
        }
        // Real SQL, so only sqlite3_interrupt can stop it: one statement that never ends, and an
        // endless run of short statements (an interrupt between two of them must not be lost).
        let sql_err = |e: rusqlite::Error| Response::err(format!("test query stopped: {e}"));
        if q == "__test_endless_statement__" {
            let n = store.conn.query_row(
                "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c",
                [],
                |r| r.get::<_, i64>(0),
            );
            return n.map_or_else(sql_err, |_| Response::err("endless statement ended"));
        }
        if q == "__test_statement_loop__" {
            let t = std::time::Instant::now();
            while t.elapsed() < Duration::from_secs(20) {
                if let Err(e) = store.doc_matches_term(1, "melanie") {
                    return Response::err(format!("test query stopped: {e}"));
                }
            }
            return Response::err("statement loop ran 20 s");
        }
    }
    let k = k.clamp(1, MAX_QUERY_K);
    if let Err(e) = crate::query::validate_channels(channels) {
        return Response::err(e.to_string());
    }
    // Settings re-read per request (one small file), so a toggle takes effect without a restart.
    let sc = crate::scrub::Scrubber::new(&crate::scrub::Config::load());
    match crate::query::fused_sourced(store, bge.as_ref(), None, q, k, channels) {
        Ok(hits) => Response::Hits {
            ok: true,
            proto: WIRE_PROTO,
            hits: hits
                .into_iter()
                .map(|(h, p)| HitWire {
                    id: h.id,
                    title: sc.scrub(&h.title).into_owned(),
                    path: h.path,
                    snippet: sc.scrub(&h.snippet).into_owned(),
                    score: h.score,
                    source: p.source,
                    ranks: p.ranks,
                    scores: p.scores,
                    rrf: p.rrf,
                    lacks: p.lacks,
                    merged: p.merged,
                    labels_truncated: p.labels_truncated,
                    closest_match: p.closest_match,
                    absent: p.absent,
                })
                .collect(),
        },
        Err(e) => Response::err(e.to_string()),
    }
}

// ── Client helpers (used by cmd_query) ───────────────────────────────────────

/// A query result returned by the daemon.
#[derive(Debug, Serialize, Deserialize)]
pub struct DaemonHit {
    pub id: i64,
    pub title: String,
    pub path: String,
    pub snippet: String,
    /// `#[serde(default)]` is REQUIRED, not decoration: a resident daemon can be older than the
    /// client that talks to it — that is the normal state on this machine, where the daemon is
    /// long-lived and survives rebuilds. Without the default, a new client meeting a pre-2026-08-21
    /// daemon fails to deserialise the whole response and every query breaks. With it, the score
    /// degrades to 0.0, which is exactly what the caller got before this field existed.
    #[serde(default)]
    pub score: f64,
    /// Same rule as `score`: a pre-2026-10-29 daemon sends none, which deserialises to "" and
    /// the client prints "unknown" — never the old hard-coded "local".
    #[serde(default)]
    pub source: String,
    /// REQUIRED, no serde default (PLAN.md 1c): the proto check runs first, so a reply that
    /// claims the current proto but lacks any of these fails to deserialise, loudly.
    pub ranks: BTreeMap<String, usize>,
    pub scores: BTreeMap<String, f64>,
    pub rrf: f64,
    /// Proto 11: absent on the wire when empty, so these two default.
    #[serde(default)]
    pub lacks: Vec<String>,
    #[serde(default)]
    pub merged: Vec<i64>,
    /// Proto 16: absent on the wire when false.
    #[serde(default)]
    pub labels_truncated: bool,
    /// Proto 17: absent on the wire when false.
    #[serde(default)]
    pub closest_match: bool,
    /// Proto 21: absent on the wire when false.
    #[serde(default)]
    pub absent: Vec<String>,
}

/// Send a query to the running daemon and return hits.
/// Returns `None` if the daemon is not running or any I/O error occurs —
/// callers must fall through to the in-process path.
pub fn query_daemon(addr: &str, q: &str, k: usize, channels: &[String]) -> Option<Vec<DaemonHit>> {
    use std::net::TcpStream;
    let sock = addr.parse().ok()?;
    let stream =
        TcpStream::connect_timeout(&sock, Duration::from_secs(CONNECT_TIMEOUT_SECS)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(QUERY_READ_TIMEOUT_SECS)))
        .ok()?;
    stream
        .set_write_timeout(Some(Duration::from_secs(CLIENT_READ_TIMEOUT_SECS)))
        .ok()?;
    let mut write_stream = stream.try_clone().ok()?;

    let (agent, project, team) = crate::scope::current().identity();
    let req = serde_json::json!({
        "q": q,
        "k": k,
        "channels": channels,
        "agent": agent,
        "project": project,
        "team": team
    });
    writeln!(write_stream, "{}", req).ok()?;

    let mut line = String::new();
    BufReader::new(stream.take(MAX_REQUEST_BYTES))
        .read_line(&mut line)
        .ok()?;

    parse_hits_reply(&line)
}

/// Hits from one daemon reply line, or `None` (the caller then answers in-process).
///
/// A reply without `proto == WIRE_PROTO` came from a different build than this client. Its answers
/// skip the abstain rules and carry no `source`, so it is bypassed — loudly, never silently.
fn parse_hits_reply(line: &str) -> Option<Vec<DaemonHit>> {
    let val: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    if !val.get("ok")?.as_bool()? {
        return None;
    }
    if val.get("proto").and_then(|p| p.as_u64()) != Some(WIRE_PROTO) {
        eprintln!(
            "recall: the resident daemon is a different build (wire proto != {WIRE_PROTO}: other abstain \
             rules); answering in-process instead. Restart it: \
             `recall daemon down`, then `recall daemon up`"
        );
        return None;
    }
    match serde_json::from_value(val["hits"].clone()) {
        Ok(h) => Some(h),
        Err(e) => {
            eprintln!(
                "recall: the resident daemon's reply (wire proto {WIRE_PROTO}) is malformed ({e}); \
                 answering in-process instead"
            );
            None
        }
    }
}

/// Send a ping to the daemon. Returns `Some(info_string)` on success.
pub fn ping_daemon(addr: &str) -> Option<String> {
    use std::net::TcpStream;
    let sock = addr.parse().ok()?;
    let stream =
        TcpStream::connect_timeout(&sock, Duration::from_secs(CONNECT_TIMEOUT_SECS)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_secs(CLIENT_READ_TIMEOUT_SECS)))
        .ok()?;
    stream
        .set_write_timeout(Some(Duration::from_secs(CLIENT_READ_TIMEOUT_SECS)))
        .ok()?;
    let mut write_stream = stream.try_clone().ok()?;
    writeln!(write_stream, r#"{{"cmd":"ping"}}"#).ok()?;

    let mut line = String::new();
    BufReader::new(stream.take(MAX_REQUEST_BYTES))
        .read_line(&mut line)
        .ok()?;
    let val: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    if val.get("ok")?.as_bool()? {
        Some(format!(
            "addr={} pid={} docs={}",
            val["addr"].as_str().unwrap_or("?"),
            val["pid"].as_u64().unwrap_or(0),
            val["docs"].as_i64().unwrap_or(0)
        ))
    } else {
        None
    }
}

/// Send a shutdown command to the daemon.
pub fn shutdown_daemon(addr: &str) -> Result<()> {
    use std::net::TcpStream;
    let sock = addr.parse().context("parse daemon addr for shutdown")?;
    let stream = TcpStream::connect_timeout(&sock, Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .context("connect to daemon for shutdown")?;
    stream
        .set_read_timeout(Some(Duration::from_secs(CLIENT_READ_TIMEOUT_SECS)))
        .context("set read timeout for shutdown")?;
    stream
        .set_write_timeout(Some(Duration::from_secs(CLIENT_READ_TIMEOUT_SECS)))
        .context("set write timeout for shutdown")?;
    let mut write_stream = stream.try_clone().context("clone stream for shutdown")?;
    writeln!(write_stream, r#"{{"cmd":"shutdown"}}"#).context("send shutdown")?;
    // Read the ack line (daemon sends {"ok":true,"hits":[]})
    let mut line = String::new();
    BufReader::new(stream.take(MAX_REQUEST_BYTES))
        .read_line(&mut line)
        .ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wire(id: i64, score: f64, source: &str) -> HitWire {
        HitWire {
            id,
            title: "T".into(),
            path: "a.md".into(),
            snippet: String::new(),
            score,
            source: source.into(),
            ranks: BTreeMap::from([("fts5".to_string(), 0)]),
            scores: BTreeMap::from([("fts5".to_string(), score)]),
            rrf: 1.0 / 61.0,
            lacks: Vec::new(),
            merged: Vec::new(),
            labels_truncated: false,
            closest_match: false,
            absent: Vec::new(),
        }
    }

    /// T8: in-process provenance survives handle_query -> wire -> parse_hits_reply bit-exact.
    #[test]
    fn handle_query_to_client_keeps_provenance() {
        let _guard = crate::query::tests::env_lock();
        let store = crate::query::tests::in_memory_store();
        store.insert("a.md", "kestra password", "kestra password rotation", "vault").unwrap();
        store.insert("b.md", "B", "kestra password kestra password", "vault").unwrap();
        let ch = vec!["fts5".to_string()];
        let local = crate::query::fused_sourced(&store, None, None, "kestra password", 5, &ch).unwrap();
        let line = serde_json::to_string(&handle_query(&store, &None, "kestra password", 5, &ch)).unwrap();
        let got = parse_hits_reply(&line).expect("current proto is served");
        assert_eq!(got.len(), local.len());
        for (d, (h, p)) in got.iter().zip(&local) {
            assert_eq!((d.id, d.score.to_bits(), &d.source), (h.id, h.score.to_bits(), &p.source));
            assert_eq!(d.ranks, p.ranks);
            assert_eq!(d.rrf.to_bits(), p.rrf.to_bits());
            assert_eq!(d.scores.len(), p.scores.len());
            for (k, v) in &p.scores {
                assert_eq!(d.scores[k].to_bits(), v.to_bits(), "{k}");
            }
        }
    }

    /// T9: a reply claiming the current proto but missing a provenance field is refused.
    #[test]
    fn a_current_proto_reply_missing_provenance_is_refused() {
        for drop in ["ranks", "scores", "rrf"] {
            let mut v = serde_json::to_value(wire(1, 3.0, "fts5")).unwrap();
            v.as_object_mut().unwrap().remove(drop);
            let line = serde_json::json!({"ok": true, "proto": WIRE_PROTO, "hits": [v]}).to_string();
            assert!(parse_hits_reply(&line).is_none(), "missing {drop} must not be served");
        }
        // T19: the previous proto (20: answerability judge, pre-cli-quality closest rules) is refused.
        assert_eq!(WIRE_PROTO, 21);
        let prev = serde_json::json!({"ok": true, "proto": WIRE_PROTO - 1,
            "hits": [serde_json::to_value(wire(1, 3.0, "fts5")).unwrap()]}).to_string();
        assert!(parse_hits_reply(&prev).is_none(), "previous proto refused");
    }

    /// lme-improve: a closest match with EMPTY `lacks` keeps its state across the wire.
    #[test]
    fn closest_match_survives_the_wire_even_with_empty_lacks() {
        let mut w = wire(1, 3.0, "fts5");
        w.closest_match = true;
        let line = serde_json::to_string(&Response::Hits { ok: true, proto: WIRE_PROTO, hits: vec![w] }).unwrap();
        assert!(line.contains("\"closest_match\":true"), "{line}");
        let hits = parse_hits_reply(&line).expect("current proto is served");
        assert!(hits[0].closest_match && hits[0].lacks.is_empty());
        let plain = serde_json::to_string(&Response::Hits { ok: true, proto: WIRE_PROTO, hits: vec![wire(1, 3.0, "fts5")] }).unwrap();
        assert!(!plain.contains("closest_match") && !parse_hits_reply(&plain).unwrap()[0].closest_match);
    }

    #[test]
    fn a_pre_proto_daemon_reply_is_bypassed_not_trusted() {
        // Exactly what the 10-27 daemon sends: hits, no `proto`, no `source`.
        let legacy = r#"{"ok":true,"hits":[{"id":1,"title":"t","path":"a.md","snippet":"","score":3.0}]}"#;
        assert!(parse_hits_reply(legacy).is_none(), "an old daemon's answer must not be served");
        let current = serde_json::to_string(&Response::Hits {
            ok: true,
            proto: WIRE_PROTO,
            hits: vec![wire(1, 3.0, "fts5")],
        })
        .unwrap();
        let hits = parse_hits_reply(&current).expect("a current daemon is served");
        assert_eq!(hits[0].source, "fts5");
        // A round-2 daemon (proto 2, WITH source) applied the old rules: refused too.
        let round2 = r#"{"ok":true,"proto":2,"hits":[{"id":1,"title":"t","path":"a.md","snippet":"","score":3.0,"source":"fts5"}]}"#;
        assert!(parse_hits_reply(round2).is_none(), "a proto-2 daemon's answer must not be served");
        let loop2r1 = r#"{"ok":true,"proto":3,"hits":[{"id":1,"title":"t","path":"a.md","snippet":"","score":3.0,"source":"fts5"}]}"#;
        assert!(parse_hits_reply(loop2r1).is_none(), "a proto-3 (loop2 round 1) daemon must not be served");
        let loop2r2 = r#"{"ok":true,"proto":4,"hits":[{"id":1,"title":"t","path":"a.md","snippet":"","score":3.0,"source":"fts5"}]}"#;
        assert!(parse_hits_reply(loop2r2).is_none(), "a proto-4 (loop2 round 2) daemon must not be served");
        let loop2r3 = r#"{"ok":true,"proto":5,"hits":[{"id":1,"title":"t","path":"a.md","snippet":"","score":3.0,"source":"fts5"}]}"#;
        assert!(parse_hits_reply(loop2r3).is_none(), "a proto-5 (loop2 round 3) daemon must not be served");
        let future = r#"{"ok":true,"proto":99,"hits":[]}"#;
        assert!(parse_hits_reply(future).is_none(), "only this build's rules are served");
        let abstained = format!(r#"{{"ok":true,"proto":{WIRE_PROTO},"hits":[]}}"#);
        assert_eq!(parse_hits_reply(&abstained).map(|h| h.len()), Some(0));
    }

    #[test]
    fn the_wire_carries_score_from_daemon_to_client() {
        // 🔴 REGRESSION GUARD for a silent fail-open. `HitWire` used to have no `score` field at
        // all, so `fused()` computed a real BM25 score and the daemon threw it away during
        // serialisation. Measured before the fix on the same query and binary:
        //
        //     via daemon   -> score 0.0
        //     --no-daemon  -> score 37.193
        //
        // The daemon is the DEFAULT path, so effectively every consumer read zero, and any
        // downstream threshold or score-weighted filter was inert rather than failing loudly.
        //
        // This asserts the CONSUMER's view - what a client actually deserialises - rather than
        // that the struct has a field, because the field existing proves nothing about whether
        // it survives the round trip.
        let json = serde_json::to_string(&wire(42, 37.193, "fts5+bge")).unwrap();
        let back: DaemonHit = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, 42);
        assert!(
            (back.score - 37.193).abs() < 1e-9,
            "score must survive the wire, got {}",
            back.score
        );
    }

    #[test]
    fn a_pre_score_daemon_payload_no_longer_parses_and_is_refused_by_proto_first() {
        // An old daemon sends no proto, so parse_hits_reply refuses it before deserialising (see
        // a_pre_proto_daemon_reply_is_bypassed_not_trusted) and the client answers in-process.
        // `score` / `source` keep serde(default); the provenance fields are required.
        let old_payload = r#"{"id":7,"title":"T","path":"notes/b.md","snippet":"s"}"#;
        assert!(serde_json::from_str::<DaemonHit>(old_payload).is_err());
        let no_legacy = r#"{"id":7,"title":"T","path":"b.md","snippet":"s","ranks":{},"scores":{},"rrf":0.0}"#;
        let back: DaemonHit = serde_json::from_str(no_legacy).unwrap();
        assert_eq!((back.score, back.source.as_str()), (0.0, ""));
    }

    #[test]
    fn the_wire_carries_source_from_daemon_to_client() {
        let back: DaemonHit =
            serde_json::from_str(&serde_json::to_string(&wire(1, 0.8, "fts5+bge")).unwrap()).unwrap();
        assert_eq!(back.source, "fts5+bge");
    }

    /// One daemon, three callers in a row (Codex tenancy r2): each sees only its own boxes, a bad
    /// identity gets an error rather than the operator view, and the daemon is the operator again
    /// after every request.
    #[test]
    fn each_request_is_scoped_to_its_own_caller_and_the_daemon_returns_to_operator() {
        // fused_sourced reads process-wide RECALL_* settings; serialise with the tests that set them.
        let _env = crate::query::tests::env_lock();
        let path = std::env::temp_dir().join(format!("recall-daemon-scope-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let store = Store::open_at(&path).unwrap();
        for (tag, tok) in [("global", "qzglobal"), ("agent:w1", "qzwone"), ("agent:w2", "qzwtwo")] {
            store
                .conn
                .execute(
                    "INSERT INTO documents (path, title, body, created_at, valid_from, scope)
                     VALUES (?1, ?1, ?2, '1', '1', ?3)",
                    rusqlite::params![format!("n/{tok}.md"), format!("wombat provisioning {tok}"), tag],
                )
                .unwrap();
        }
        let ch = vec!["fts5".to_string()];
        let titles = |r: Response| -> Vec<String> {
            match r {
                Response::Hits { hits, .. } => hits.into_iter().map(|h| h.title).collect(),
                other => panic!("expected hits, got {}", serde_json::to_string(&other).unwrap()),
            }
        };
        let as_ = |a: Option<&str>| (a.map(String::from), None, None);
        let w1 = titles(handle_scoped_query(&store, &None, "wombat provisioning", 10, &ch, as_(Some("w1"))));
        assert!(w1.iter().any(|t| t.contains("qzwone")) && !w1.iter().any(|t| t.contains("qzwtwo")), "{w1:?}");
        assert!(crate::scope::current().is_operator(), "daemon must return to the operator after a request");
        let w2 = titles(handle_scoped_query(&store, &None, "wombat provisioning", 10, &ch, as_(Some("w2"))));
        assert!(w2.iter().any(|t| t.contains("qzwtwo")) && !w2.iter().any(|t| t.contains("qzwone")), "{w2:?}");
        let op = titles(handle_scoped_query(&store, &None, "wombat provisioning", 10, &ch, as_(None)));
        assert_eq!(op.len(), 3, "operator sees every box: {op:?}");
        assert!(matches!(
            handle_scoped_query(&store, &None, "wombat", 10, &ch, as_(Some("Bad!"))),
            Response::Error { .. }
        ));
        assert!(crate::scope::current().is_operator());
        // The client sends exactly what resolve() accepts back.
        let r = crate::scope::resolve(Some("w1".into()), Some("p".into()), Some("a".into())).unwrap();
        assert_eq!(r.identity(), (Some("w1".into()), Some("p".into()), Some("a".into())));
        assert_eq!(crate::scope::resolve(r.identity().0, r.identity().1, r.identity().2).unwrap(), r);
    }

    /// Test-only: how long every daemon query takes (a "slow vault"), for the concurrency tests.
    pub(super) static QUERY_DELAY_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// Lead 2026-11-06: one slow query held up every other client, `status` included. Now queries
    /// run side by side and ping is answered by the accept thread while they run.
    #[test]
    fn ping_answers_and_queries_overlap_while_slow_queries_run() {
        use std::net::TcpStream;
        use std::sync::atomic::Ordering;
        let _env = crate::query::tests::env_lock();
        struct Reset(Option<std::net::SocketAddr>);
        impl Drop for Reset {
            fn drop(&mut self) {
                QUERY_DELAY_MS.store(0, Ordering::SeqCst);
                if let Some(Ok(mut s)) = self.0.map(|a| TcpStream::connect(a)) {
                    let _ = writeln!(s, r#"{{"cmd":"shutdown"}}"#);
                }
            }
        }
        let path = std::env::temp_dir().join(format!("recall-daemon-conc-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&path);
        crate::query::tests::add_chat_notes(&Store::open_at(&path).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut reset = Reset(Some(addr));
        QUERY_DELAY_MS.store(800, Ordering::SeqCst); // inside the 1 s test budget
        let p = path.clone();
        let server = std::thread::spawn(move || {
            let store = Store::open_at(&p).unwrap();
            let p2 = p.clone();
            accept_loop(&listener, &store, Default::default(), None, move || Store::open_at(&p2)).unwrap();
        });
        let ask = move |line: String| {
            let t = std::time::Instant::now();
            let mut s = TcpStream::connect(addr).unwrap();
            s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
            writeln!(s, "{line}").unwrap();
            let mut out = String::new();
            BufReader::new(s).read_line(&mut out).unwrap();
            (out, t.elapsed())
        };
        let t0 = std::time::Instant::now();
        let queries: Vec<_> = (0..3)
            .map(|_| {
                std::thread::spawn(move || {
                    ask(r#"{"q":"melanie camping","k":5,"channels":["fts5"]}"#.to_string())
                })
            })
            .collect();
        std::thread::sleep(Duration::from_millis(300)); // all three are inside their 0.8 s query now
        let (pong, took) = ask(r#"{"cmd":"ping"}"#.to_string());
        assert!(pong.contains("\"pong\":true"), "{pong}");
        assert!(took < Duration::from_millis(500), "ping waited behind the queries: {took:?}");
        for q in queries {
            let (out, _) = q.join().unwrap();
            assert!(out.contains("\"ok\":true") && out.contains("\"hits\":[{"), "{out}");
        }
        // three 0.8 s queries one at a time would take >= 2.4 s
        assert!(t0.elapsed() < Duration::from_millis(1800), "queries ran one at a time: {:?}", t0.elapsed());
        QUERY_DELAY_MS.store(0, Ordering::SeqCst);
        ask(r#"{"cmd":"shutdown"}"#.to_string());
        reset.0 = None;
        server.join().unwrap();
        for ext in ["db", "db-wal", "db-shm"] {
            let _ = std::fs::remove_file(path.with_extension(ext));
        }
    }

    /// Live wedge 2026-11-06: 101 back-to-back --top 50 queries whose clients gave up at 5 s; the
    /// daemon ran every abandoned request, one after another, and answered nothing else. Now a
    /// request whose client has gone is dropped (or cancelled mid-run), and a query that outruns
    /// its budget is cut off with a reason instead of holding a worker.
    #[test]
    fn abandoned_queries_are_dropped_and_a_slow_one_is_cut_at_its_budget() {
        use std::net::TcpStream;
        let _env = crate::query::tests::env_lock();
        struct Reset(Option<std::net::SocketAddr>);
        impl Drop for Reset {
            fn drop(&mut self) {
                QUERY_DELAY_MS.store(0, Ordering::SeqCst);
                if let Some(Ok(mut s)) = self.0.map(|a| TcpStream::connect(a)) {
                    let _ = writeln!(s, r#"{{"cmd":"shutdown"}}"#);
                }
            }
        }
        let path = std::env::temp_dir().join(format!("recall-daemon-wedge-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&path);
        crate::query::tests::add_chat_notes(&Store::open_at(&path).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut reset = Reset(Some(addr));
        let p = path.clone();
        let server = std::thread::spawn(move || {
            let store = Store::open_at(&p).unwrap();
            let p2 = p.clone();
            accept_loop(&listener, &store, Default::default(), None, move || Store::open_at(&p2)).unwrap();
        });
        let ask = |line: &str| {
            let t = std::time::Instant::now();
            let mut s = TcpStream::connect(addr).unwrap();
            s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
            writeln!(s, "{line}").unwrap();
            let mut out = String::new();
            BufReader::new(s).read_line(&mut out).unwrap();
            (out, t.elapsed())
        };
        let query = r#"{"q":"melanie camping","k":50,"channels":["fts5"]}"#;
        ask(query); // warm: the workers have opened their stores
        QUERY_DELAY_MS.store(700, Ordering::SeqCst);
        let started = QUERIES_STARTED.load(Ordering::SeqCst);
        // A burst of clients that send a query and give up at once.
        for _ in 0..12 {
            let mut s = TcpStream::connect(addr).unwrap();
            writeln!(s, "{query}").unwrap();
        }
        let (pong, took) = ask(r#"{"cmd":"ping"}"#);
        assert!(pong.contains("\"pong\":true") && took < Duration::from_millis(300), "ping: {took:?} {pong}");
        let (out, took) = ask(query);
        assert!(out.contains("\"hits\":[{"), "{out}");
        // one 0.7 s query of its own; 12 abandoned ones run in turn would be >= 2 s on 4 workers
        assert!(took < Duration::from_millis(1600), "a live query waited behind abandoned ones: {took:?}");
        let ran = QUERIES_STARTED.load(Ordering::SeqCst) - started;
        assert!(ran <= 5, "{ran} queries ran for 12 abandoned clients + 1 live one (4 can be in flight)");
        // A query slower than its budget comes back cancelled, with the reason, instead of hanging.
        QUERY_DELAY_MS.store(10_000, Ordering::SeqCst);
        let (out, took) = ask(query);
        assert!(out.contains("time budget") && took < Duration::from_millis(2500), "{took:?} {out}");
        QUERY_DELAY_MS.store(0, Ordering::SeqCst);
        ask(r#"{"cmd":"shutdown"}"#);
        reset.0 = None;
        server.join().unwrap();
        for ext in ["db", "db-wal", "db-shm"] {
            let _ = std::fs::remove_file(path.with_extension(ext));
        }
    }

    /// A daemon on a fresh 30-note store, its address, and a guard that resets the slow-query hook
    /// and shuts the daemon down even when an assertion panics. Callers hold `env_lock`.
    struct TestDaemon {
        addr: std::net::SocketAddr,
        path: std::path::PathBuf,
        server: Option<std::thread::JoinHandle<()>>,
    }

    impl TestDaemon {
        fn start(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!("recall-daemon-{tag}-{}.db", std::process::id()));
            let _ = std::fs::remove_file(&path);
            crate::query::tests::add_chat_notes(&Store::open_at(&path).unwrap());
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let p = path.clone();
            let server = std::thread::spawn(move || {
                let store = Store::open_at(&p).unwrap();
                let p2 = p.clone();
                accept_loop(&listener, &store, Default::default(), None, move || Store::open_at(&p2)).unwrap();
            });
            let d = TestDaemon { addr, path, server: Some(server) };
            d.ask(r#"{"q":"melanie","k":5,"channels":["fts5"]}"#); // warm: workers have their stores
            d
        }

        fn ask(&self, line: &str) -> (String, Duration) {
            let t = std::time::Instant::now();
            let mut s = std::net::TcpStream::connect(self.addr).unwrap();
            s.set_read_timeout(Some(Duration::from_secs(40))).unwrap();
            writeln!(s, "{line}").unwrap();
            let mut out = String::new();
            let _ = BufReader::new(s).read_line(&mut out);
            (out, t.elapsed())
        }

        /// Ask for shutdown; how long until the accept loop had returned and every worker joined.
        fn shutdown(&mut self) -> Duration {
            let t = std::time::Instant::now();
            self.ask(r#"{"cmd":"shutdown"}"#);
            self.server.take().unwrap().join().unwrap();
            t.elapsed()
        }
    }

    impl Drop for TestDaemon {
        fn drop(&mut self) {
            QUERY_DELAY_MS.store(0, Ordering::SeqCst);
            if self.server.is_some() {
                if let Ok(mut s) = std::net::TcpStream::connect(self.addr) {
                    let _ = writeln!(s, r#"{{"cmd":"shutdown"}}"#);
                }
            }
            for ext in ["db", "db-wal", "db-shm"] {
                let _ = std::fs::remove_file(self.path.with_extension(ext));
            }
        }
    }

    /// Codex conc r1 #1: sqlite3_interrupt only stops the statement running at that instant. A
    /// statement that never ends, and an endless run of short statements (where a single interrupt
    /// lands between two of them), must both be cut at the budget.
    #[test]
    fn a_cancelled_query_is_stopped_inside_and_between_sql_statements() {
        let _env = crate::query::tests::env_lock();
        let mut d = TestDaemon::start("interrupt");
        for q in ["__test_endless_statement__", "__test_statement_loop__"] {
            let (out, took) = d.ask(&format!(r#"{{"q":"{q}","k":5,"channels":["fts5"]}}"#));
            assert!(out.contains("time budget") && took < Duration::from_millis(2500), "{q}: {took:?} {out}");
        }
        // Every worker's connection is usable again (no cancel flag or interrupt leaks into its next
        // query): 8 overlapping 0.3 s queries on 4 workers keep each worker busy twice.
        QUERY_DELAY_MS.store(300, Ordering::SeqCst);
        let addr = d.addr;
        let all: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(move || {
                    let mut s = std::net::TcpStream::connect(addr).unwrap();
                    s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
                    writeln!(s, r#"{{"q":"melanie camping","k":5,"channels":["fts5"]}}"#).unwrap();
                    let mut out = String::new();
                    BufReader::new(s).read_line(&mut out).unwrap();
                    out
                })
            })
            .collect();
        for t in all {
            let out = t.join().unwrap();
            assert!(out.contains("\"hits\":[{"), "{out}");
        }
        d.shutdown();
    }

    /// Codex conc r2 #2: a byte sent after the request has started must not hide the client's
    /// state behind "data waiting": any input after the request line cancels the query.
    #[test]
    fn input_sent_after_a_query_started_cancels_it() {
        let _env = crate::query::tests::env_lock();
        let mut d = TestDaemon::start("late-input");
        QUERY_DELAY_MS.store(800, Ordering::SeqCst);
        let t = std::time::Instant::now();
        let mut s = std::net::TcpStream::connect(d.addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        writeln!(s, r#"{{"q":"melanie camping","k":5,"channels":["fts5"]}}"#).unwrap();
        std::thread::sleep(Duration::from_millis(200)); // the query is running now
        s.write_all(b"x").unwrap();
        let mut out = String::new();
        match BufReader::new(s).read_line(&mut out) {
            Ok(_) => assert!(out.contains("cancelled"), "{out}"),
            // Closing a socket with unread protocol junk can send RST on Windows,
            // instead of delivering the cancellation error followed by EOF.
            Err(e) if cfg!(windows) && e.kind() == std::io::ErrorKind::ConnectionReset => {
                assert!(out.is_empty() || out.contains("cancelled"), "{out}");
            }
            Err(e) => panic!("reading cancellation response: {e}"),
        }
        assert!(t.elapsed() < Duration::from_millis(650), "{:?} {out}", t.elapsed());
        d.shutdown();
    }

    /// Codex conc r1 #3: shutdown discards queued queries and cancels running ones instead of
    /// draining the backlog first.
    #[test]
    fn shutdown_discards_the_queue_instead_of_draining_it() {
        let _env = crate::query::tests::env_lock();
        let mut d = TestDaemon::start("shutdown");
        QUERY_DELAY_MS.store(900, Ordering::SeqCst);
        // 12 clients that wait for their answers: 4 run, 8 queue (>= 2.7 s to drain on 4 workers)
        let waiting: Vec<_> = (0..12)
            .map(|_| {
                let mut s = std::net::TcpStream::connect(d.addr).unwrap();
                writeln!(s, r#"{{"q":"melanie camping","k":5,"channels":["fts5"]}}"#).unwrap();
                s
            })
            .collect();
        std::thread::sleep(Duration::from_millis(200));
        let took = d.shutdown();
        assert!(took < Duration::from_millis(1200), "shutdown drained the backlog: {took:?}");
        drop(waiting);
    }

    /// Codex conc r1 #4: the protocol is one request line, then the client waits. Trailing input is
    /// refused (so a later peek can only see "nothing yet" or the close), and a half-close reads as
    /// the client leaving.
    #[test]
    fn trailing_input_is_refused_and_a_half_close_is_a_departure() {
        let _env = crate::query::tests::env_lock();
        let mut d = TestDaemon::start("socket");
        let (out, _) = d.ask("{\"q\":\"melanie\",\"k\":5,\"channels\":[\"fts5\"]}\nmore");
        assert!(out.contains("unexpected data after the request line"), "{out}");
        QUERY_DELAY_MS.store(300, Ordering::SeqCst);
        let dropped = QUERIES_DROPPED.load(Ordering::SeqCst) + QUERIES_STARTED.load(Ordering::SeqCst);
        let mut s = std::net::TcpStream::connect(d.addr).unwrap();
        writeln!(s, r#"{{"q":"melanie camping","k":5,"channels":["fts5"]}}"#).unwrap();
        s.shutdown(std::net::Shutdown::Write).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut out = String::new();
        let _ = BufReader::new(s).read_line(&mut out);
        assert!(!out.contains("hits"), "a half-closed client got an answer: {out}");
        assert!(QUERIES_DROPPED.load(Ordering::SeqCst) + QUERIES_STARTED.load(Ordering::SeqCst) > dropped);
        d.shutdown();
    }

    /// The CLI waits QUERY_READ_TIMEOUT_SECS for a query answer: a daemon answering in 6 s must not
    /// be abandoned for the ~20 s in-process path (it gave up at 5 s before 2026-11-06).
    #[test]
    fn the_cli_waits_longer_than_five_seconds_for_a_query_answer() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let server = std::thread::spawn(move || {
            let (s, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(s.try_clone().unwrap()).read_line(&mut line).unwrap();
            std::thread::sleep(Duration::from_secs(6));
            let mut s = s;
            writeln!(s, r#"{{"ok":true,"proto":{WIRE_PROTO},"hits":[]}}"#).unwrap();
        });
        assert!(query_daemon(&addr, "anything", 5, &["fts5".into()]).is_some(), "the CLI gave up before 6 s");
        server.join().unwrap();
    }

    /// Codex conc r1 #2: a refresh patch must get the matrix while queries keep overlapping.
    #[test]
    fn a_writer_is_not_starved_by_overlapping_readers() {
        let g = std::sync::Arc::new(Gated::new(0u32));
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let readers: Vec<_> = (0..4)
            .map(|i| {
                let (g, stop) = (g.clone(), stop.clone());
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(20 * i));
                    while !stop.load(Ordering::SeqCst) {
                        let _r = g.read();
                        std::thread::sleep(Duration::from_millis(80)); // a query holding the matrix
                    }
                })
            })
            .collect();
        std::thread::sleep(Duration::from_millis(200)); // readers now always overlap
        let t = std::time::Instant::now();
        *g.write() += 1;
        let took = t.elapsed();
        stop.store(true, Ordering::SeqCst);
        for r in readers {
            r.join().unwrap();
        }
        assert!(took < Duration::from_millis(500), "the writer waited {took:?} behind a stream of readers");
    }

    /// Codex calib r3: a view queued while the store would not open was dropped into a 10-minute
    /// cooldown. It must be calibrated as soon as the store opens, with no further query.
    #[test]
    fn a_view_queued_while_the_store_will_not_open_is_calibrated_once_it_does() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let _env = crate::query::tests::env_lock();
        let path = std::env::temp_dir().join(format!("recall-daemon-reopen-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&path);
        crate::query::tests::add_chat_notes(&Store::open_at(&path).unwrap());
        let (submitted, failed_after) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (s, f, p) = (submitted.clone(), failed_after.clone(), path.clone());
        // Fails until it has failed once after the submit, so the reader arrives while it is closed.
        let q = spawn_calibrator(move || {
            if !s.load(Ordering::SeqCst) || !f.swap(true, Ordering::SeqCst) {
                anyhow::bail!("store busy (test)");
            }
            Store::open_at(&p)
        });
        q.submit(crate::scope::resolve(Some("w1".into()), None, None).unwrap());
        submitted.store(true, Ordering::SeqCst);
        let store = Store::open_at(&path).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while store.calibration_latest("s").unwrap().is_none() {
            assert!(std::time::Instant::now() < deadline, "the reader queued while closed was never calibrated");
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(failed_after.load(Ordering::SeqCst), "the store did fail after the submit");
        drop((q, store));
        for ext in ["db", "db-wal", "db-shm"] {
            let _ = std::fs::remove_file(path.with_extension(ext));
        }
    }

    /// Live hang 2026-11-06: a scoped agent's first near-miss query computed its phantom null (200
    /// judged probes) on the request thread and every client timed out behind it. With the
    /// calibrator, the request thread only looks up; the calibrator computes and stores the row.
    #[test]
    fn a_scoped_near_miss_does_not_block_a_concurrent_operator_query() {
        use crate::query::tests::{CALIBRATED_ON, CALIBRATION_DELAY_MS};
        use std::net::TcpStream;
        use std::sync::atomic::Ordering;
        let _env = crate::query::tests::env_lock();
        /// Undoes the test's global settings and stops the server even when an assertion panics.
        struct Cleanup {
            saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
            addr: Option<std::net::SocketAddr>,
        }
        impl Drop for Cleanup {
            fn drop(&mut self) {
                CALIBRATION_DELAY_MS.store(0, Ordering::SeqCst);
                for (k, v) in self.saved.drain(..) {
                    if let Some(v) = v {
                        std::env::set_var(k, v);
                    }
                }
                if let Some(Ok(mut s)) = self.addr.map(|a| TcpStream::connect(a)) {
                    let _ = writeln!(s, r#"{{"cmd":"shutdown"}}"#);
                }
            }
        }
        // The calibrated path is the default one; an inherited override must not skip it.
        let mut cleanup = Cleanup {
            saved: ["RECALL_ABSTAIN", "RECALL_CHANNELS"].map(|k| (k, std::env::var_os(k))).into(),
            addr: None,
        };
        for (k, _) in &cleanup.saved {
            std::env::remove_var(k);
        }
        let path = std::env::temp_dir().join(format!("recall-daemon-calib-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&path);
        crate::query::tests::add_chat_notes(&Store::open_at(&path).unwrap());
        CALIBRATED_ON.lock().unwrap().clear();
        CALIBRATION_DELAY_MS.store(3000, Ordering::SeqCst); // a "slow vault": each null takes 3 s

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        cleanup.addr = Some(addr);
        let p = path.clone();
        let server = std::thread::Builder::new()
            .name("accept-test".into())
            .spawn(move || {
                let store = Store::open_at(&p).unwrap();
                let p2 = p.clone();
                accept_loop(&listener, &store, Default::default(), None, move || Store::open_at(&p2)).unwrap();
            })
            .unwrap();
        let ask = |line: &str| {
            let t = std::time::Instant::now();
            let mut s = TcpStream::connect(addr).unwrap();
            s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
            writeln!(s, "{line}").unwrap();
            let mut out = String::new();
            BufReader::new(s).read_line(&mut out).unwrap();
            (out, t.elapsed())
        };
        // A scoped agent's near-miss (one unknown word: the calibrated closest-match path).
        let near_miss = r#"{"q":"melanie camping beach adoption zorblat","k":5,"channels":["fts5"],"agent":"w1"}"#;
        let hits = |out: &str| {
            let v: serde_json::Value = serde_json::from_str(out).unwrap_or_else(|e| panic!("{e}: {out}"));
            assert_eq!(v["ok"], true, "{out}");
            v["hits"].as_array().unwrap_or_else(|| panic!("{out}")).len()
        };
        let (out, took) = ask(near_miss);
        assert!(took < Duration::from_millis(1500), "the scoped query waited for its calibration: {took:?}");
        // never calibrated for this view: under RECALL_ABSTAIN_POLICY=refuse (env_lock pins it) the strict
        // verdict (abstain), not a guessed closest match
        assert_eq!(hits(&out), 0, "{out}");
        // Under the default label policy the same cold view never answers with nothing: its hits come back,
        // every one closest_match, without waiting for (or needing) a calibration (answerability-judge).
        std::env::set_var("RECALL_ABSTAIN_POLICY", "label");
        let (out, took) = ask(r#"{"q":"melanie camping beach adoption zorblat","k":5,"channels":["fts5"],"agent":"w2"}"#);
        std::env::set_var("RECALL_ABSTAIN_POLICY", "refuse");
        assert!(took < Duration::from_millis(1500), "the labelled cold view waited: {took:?}");
        assert!(hits(&out) > 0, "label policy: a cold scoped view returns labelled hits: {out}");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(v["hits"].as_array().unwrap().iter().all(|h| h["closest_match"] == true), "{out}");
        // ...and the operator, right behind it, is answered while the calibrator is still busy.
        let (out, took) = ask(r#"{"q":"melanie camping","k":5,"channels":["fts5"]}"#);
        assert!(took < Duration::from_millis(1500), "the operator was blocked behind a calibration: {took:?}");
        assert!(hits(&out) > 0, "{out}");
        // The calibration did run, and only on the calibrator thread; its row lands.
        let store = Store::open_at(&path).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(60);
        while store.calibration_latest("s").unwrap().is_none() {
            assert!(std::time::Instant::now() < deadline, "the calibrator never stored the scoped null");
            std::thread::sleep(Duration::from_millis(50));
        }
        drop(store);
        let on = CALIBRATED_ON.lock().unwrap().clone();
        // (other tests calibrate on their own threads concurrently; only these two names matter)
        assert!(
            on.iter().any(|t| t == "recall-calibrate")
                && !on.iter().any(|t| t == "accept-test" || t.starts_with("recall-query-")),
            "{on:?}"
        );

        ask(r#"{"cmd":"shutdown"}"#);
        cleanup.addr = None;
        server.join().unwrap();
        // The accept loop is gone, so the calibrator's channel is closed and it lets go of the
        // database (Windows refuses to delete a file a connection still holds).
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while let Err(e) = std::fs::remove_file(&path) {
            assert!(std::time::Instant::now() < deadline, "the calibrator still holds the db: {e}");
            std::thread::sleep(Duration::from_millis(100));
        }
        for ext in ["db-wal", "db-shm"] {
            let _ = std::fs::remove_file(path.with_extension(ext));
        }
    }
}
