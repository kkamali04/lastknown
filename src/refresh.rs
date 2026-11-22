// refresh.rs — keep the index honest about files that changed on disk.
//
// Three cooperating pieces, all reusing the existing machinery:
//
//   * a WATCHER (daemon thread, `notify`) that only RECORDS: it flags documents whose file was
//     edited or deleted, and queues created/renamed-in files. It never embeds and never rewrites a
//     body, so it is cheap and cannot fall behind badly;
//   * a SCAN (`Store::scan_freshness`) that does the same comparison by mtime+size, for when the
//     watcher was not running (daemon down, watcher error, files changed while the PC was off);
//   * a REFRESH (`run`) that applies the flags — bodies rewritten IN PLACE so doc ids survive,
//     moves resolved by the same rule `ingest_dir` uses — then re-embeds only the changed
//     documents by handing their paths to `recall embed --path-list --force`.
//
// 🔴 A refresh takes the single-writer index lock and gives up (does not queue) if anything else
// holds it, and it backs off when the machine is busy. Both are skips, reported as skips: a run
// that did not happen must never look like a run that found nothing.

use crate::settings::{RefreshMode, Settings};
use crate::store::freshness::{self, ApplyReport, ScanReport};
use crate::store::{self, Store};
use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ── Single-writer lock ──────────────────────────────────────────────────────────

/// `~/.recall/index.lock`, an OS advisory lock: released by the kernel when the holder exits or
/// crashes, so there is no stale-lock cleanup to get wrong.
pub struct IndexLock {
    _file: std::fs::File,
}

/// Unlock explicitly. Closing the handle also releases the lock, but on Windows only "eventually"
/// (LockFileEx docs: the time it takes depends on system resources), and on Linux a descriptor a
/// concurrent fork inherited keeps it until that child execs or exits. Either way a refresh started
/// right after another one in the same process could find the lock still held and skip, which
/// would explain the `a_failed_embed_is_retried...` test flaking under loaded parallel runs.
impl Drop for IndexLock {
    fn drop(&mut self) {
        let _ = self._file.unlock();
    }
}

/// `~/.recall`, or an explicit directory (tests; never the real one).
fn recall_dir(dir: Option<&Path>) -> PathBuf {
    match dir {
        Some(d) => d.to_path_buf(),
        None => {
            let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
            PathBuf::from(home).join(".recall")
        }
    }
}

pub fn lock_path() -> PathBuf {
    recall_dir(None).join("index.lock")
}

impl IndexLock {
    /// `Ok(None)` when someone else holds it.
    pub fn try_acquire() -> Result<Option<Self>> {
        Self::try_acquire_in(None)
    }

    pub fn try_acquire_in(dir: Option<&Path>) -> Result<Option<Self>> {
        let path = recall_dir(dir).join("index.lock");
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .with_context(|| format!("open {}", path.display()))?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { _file: file })),
            Err(std::fs::TryLockError::WouldBlock) => Ok(None),
            Err(std::fs::TryLockError::Error(e)) => Err(e).context("lock index.lock"),
        }
    }

    /// Wait for the lock; a full index is long, so a refresh that started first is not an excuse
    /// to skip the 02:45 job — but a hung holder must not hang us forever either.
    pub fn acquire_wait(timeout: Duration) -> Result<Self> {
        let start = Instant::now();
        loop {
            if let Some(l) = Self::try_acquire()? {
                return Ok(l);
            }
            if start.elapsed() >= timeout {
                anyhow::bail!(
                    "another recall index/refresh has held {} for {}s; not running a second writer",
                    lock_path().display(),
                    timeout.as_secs()
                );
            }
            std::thread::sleep(Duration::from_secs(5));
        }
    }
}

// ── Back-off when the PC is busy ────────────────────────────────────────────────

const DEFAULT_MAX_LOAD: f64 = 0.75;

fn max_load() -> f64 {
    std::env::var("RECALL_REFRESH_MAX_LOAD")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(DEFAULT_MAX_LOAD)
}

#[cfg(windows)]
fn cpu_busy_ratio() -> Option<f64> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::GetSystemTimes;
    fn sample() -> Option<(u64, u64, u64)> {
        let z = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
        let (mut i, mut k, mut u) = (z, z, z);
        // SAFETY: three valid out-pointers to FILETIME structs on our stack.
        if unsafe { GetSystemTimes(&mut i, &mut k, &mut u) } == 0 {
            return None;
        }
        let f = |t: FILETIME| ((t.dwHighDateTime as u64) << 32) | t.dwLowDateTime as u64;
        Some((f(i), f(k), f(u)))
    }
    let a = sample()?;
    std::thread::sleep(Duration::from_millis(500));
    let b = sample()?;
    // Kernel time includes idle time, so total = kernel + user.
    let total = (b.1 - a.1) + (b.2 - a.2);
    let idle = b.0 - a.0;
    (total > 0).then(|| 1.0 - (idle as f64 / total as f64))
}

#[cfg(not(windows))]
fn cpu_busy_ratio() -> Option<f64> {
    let cores = std::thread::available_parallelism().map(|n| n.get() as f64).unwrap_or(1.0);
    let text = std::fs::read_to_string("/proc/loadavg").ok().or_else(|| {
        // macOS: "{ 1.23 1.50 1.60 }"
        let out = std::process::Command::new("sysctl").args(["-n", "vm.loadavg"]).output().ok()?;
        Some(String::from_utf8_lossy(&out.stdout).replace(['{', '}'], ""))
    })?;
    let one: f64 = text.split_whitespace().next()?.parse().ok()?;
    Some(one / cores)
}

/// `Some(reason)` when the machine is too busy to start background work.
/// An unreadable load is NOT busy: the user opted into refresh, and a platform we cannot measure
/// should still refresh (the schedule and the lock already bound it).
pub fn system_busy() -> Option<String> {
    let ratio = cpu_busy_ratio()?;
    (ratio > max_load()).then(|| format!("system busy ({:.0}% CPU > {:.0}% limit)", ratio * 100.0, max_load() * 100.0))
}

// ── Persistent state ────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct RefreshState {
    pub last_ok_unix: Option<i64>,
    pub last_attempt_unix: Option<i64>,
    pub last_summary: Option<String>,
    pub watcher_pid: Option<u32>,
    pub watcher_started_unix: Option<i64>,
    /// Vault-relative paths whose body is current but whose re-embed has not succeeded yet. The
    /// next refresh retries them; without this, a failed embed would be forgotten as soon as the
    /// body update cleared the file's flag.
    #[serde(default)]
    pub pending_embed: Vec<String>,
}

impl RefreshState {
    pub fn load() -> Self {
        Self::load_in(None)
    }
    pub fn load_in(dir: Option<&Path>) -> Self {
        std::fs::read_to_string(recall_dir(dir).join("refresh-state.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }
    pub fn save(&self) {
        self.save_in(None)
    }
    pub fn save_in(&self, dir: Option<&Path>) {
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let d = recall_dir(dir);
            let _ = std::fs::create_dir_all(&d);
            let _ = std::fs::write(d.join("refresh-state.json"), text);
        }
    }
}

// ── Schedule ────────────────────────────────────────────────────────────────────

/// Is a scheduled refresh due? Pure, so every mode is testable with a fake clock.
/// `has_pending`/`quiet` only matter to `on-change`: pending work, and no file event for a while.
pub fn is_due(s: &Settings, last_ok: Option<i64>, now: i64, has_pending: bool, quiet: bool) -> bool {
    match s.refresh {
        RefreshMode::OnChange => has_pending && quiet,
        RefreshMode::Every(m) => last_ok.is_none_or(|t| now - t >= (m as i64) * 60),
        RefreshMode::Nightly => {
            let (h, m) = s.nightly_at_utc;
            let today = now.div_euclid(86_400) * 86_400 + (h as i64) * 3_600 + (m as i64) * 60;
            let trigger = if now >= today { today } else { today - 86_400 };
            last_ok.is_none_or(|t| t < trigger)
        }
    }
}

/// When [`is_due`] next turns true (unix secs; `now` if it already is). `None` for `on-change`,
/// which fires on file events rather than a clock. A scheduled run starts at the next check after.
pub fn next_due(s: &Settings, last_ok: Option<i64>, now: i64) -> Option<i64> {
    if is_due(s, last_ok, now, false, true) && s.refresh != RefreshMode::OnChange {
        return Some(now);
    }
    match s.refresh {
        RefreshMode::OnChange => None,
        RefreshMode::Every(m) => last_ok.map(|t| t + (m as i64) * 60),
        // Not due means last_ok >= the latest trigger, so the next due second is the first daily
        // trigger strictly after last_ok (also right when last_ok is in the future: clock rollback).
        RefreshMode::Nightly => {
            let (h, m) = s.nightly_at_utc;
            let t = last_ok?;
            let trig = t.div_euclid(86_400) * 86_400 + (h as i64) * 3_600 + (m as i64) * 60;
            Some(if trig > t { trig } else { trig + 86_400 })
        }
    }
}

// ── The refresh ─────────────────────────────────────────────────────────────────

pub struct RefreshOpts {
    /// Re-embed changed documents by spawning `recall embed` (false in tests and `--no-embed`).
    pub embed: bool,
    /// Ignore the busy check (an explicit `recall refresh` from a human).
    pub ignore_busy: bool,
    pub find_new: bool,
    pub legacy_cap: usize,
    pub busy: fn() -> Option<String>,
    /// Where the lock and state files live. `None` = `~/.recall` (always, outside tests).
    pub dir: Option<PathBuf>,
    /// The binary to run for the embed step. `None` = this executable.
    pub embed_exe: Option<PathBuf>,
}

impl Default for RefreshOpts {
    fn default() -> Self {
        Self {
            embed: true,
            ignore_busy: false,
            find_new: true,
            legacy_cap: 20_000,
            busy: system_busy,
            dir: None,
            embed_exe: None,
        }
    }
}

#[derive(Debug, Default, serde::Serialize)]
pub struct RefreshReport {
    pub scan_checked: usize,
    pub scan_changed: usize,
    pub scan_deleted: usize,
    pub scan_new: usize,
    pub legacy_stamped: usize,
    pub legacy_remaining: usize,
    pub updated: usize,
    pub touched: usize,
    pub moved: usize,
    pub inserted: usize,
    pub revived: usize,
    pub tombstoned: usize,
    pub deletes_deferred: usize,
    pub reembedded_docs: usize,
    pub embed_note: String,
    /// The embed step ran and did not succeed (the documents stay queued for the next pass).
    pub embed_failed: bool,
    #[serde(skip)]
    pub reembed_ids: Vec<i64>,
    /// Ids whose vectors the warm daemon must re-read: re-embedded documents and tombstoned ones.
    #[serde(skip)]
    pub patch_ids: Vec<i64>,
}

#[derive(Debug)]
pub enum Outcome {
    Ran(RefreshReport),
    /// Did not run, and why. NOT a "found nothing".
    Skipped(String),
}

impl RefreshReport {
    fn absorb(&mut self, s: &ScanReport, a: &ApplyReport) {
        self.scan_checked = s.checked;
        self.scan_changed = s.changed;
        self.scan_deleted = s.deleted;
        self.scan_new = s.new_files;
        self.legacy_stamped = s.legacy_stamped;
        self.legacy_remaining = s.legacy_remaining;
        self.updated = a.updated;
        self.touched = a.touched;
        self.moved = a.moved;
        self.inserted = a.inserted;
        self.revived = a.revived;
        self.tombstoned = a.tombstoned;
        self.deletes_deferred = a.deletes_deferred;
        self.reembed_ids = a.reembed_ids.clone();
    }

    pub fn summary(&self) -> String {
        format!(
            "{} edited, {} touched-only, {} renamed, {} new, {} revived, {} deleted ({} deletes held back); {} doc(s) re-embedded; embed: {}",
            self.updated, self.touched, self.moved, self.inserted, self.revived, self.tombstoned,
            self.deletes_deferred, self.reembedded_docs, self.embed_note
        )
    }
}

/// Spawn `recall embed --path-list <file> --force` at low priority and wait for it.
/// `--force` is deliberate: without it the path-list queue means "documents with NO chunk rows",
/// which an edited document already has, so a plain path-list embed would silently do nothing.
///
/// Returns a note and the paths that could not be sent (a newline in a name would split the
/// newline-delimited list into two wrong entries; the broad `recall embed` picks those up).
fn embed_paths(exe: Option<&Path>, dir: Option<&Path>, paths: &[String]) -> Result<(String, Vec<String>)> {
    let exe = match exe {
        Some(e) => e.to_path_buf(),
        None => std::env::current_exe().context("locate recall binary for the embed step")?,
    };
    let (sendable, unsendable): (Vec<&String>, Vec<&String>) =
        paths.iter().partition(|p| !p.contains('\n') && !p.contains('\r'));
    if sendable.is_empty() {
        return Ok(("nothing sendable".into(), unsendable.into_iter().cloned().collect()));
    }
    let work = recall_dir(dir);
    std::fs::create_dir_all(&work).with_context(|| format!("create {}", work.display()))?;
    let list = work.join(format!("refresh-embed-{}.txt", std::process::id()));
    let errlog = work.join(format!("refresh-embed-{}.err", std::process::id()));
    std::fs::write(&list, sendable.iter().map(|p| p.as_str()).collect::<Vec<_>>().join("\n"))
        .with_context(|| format!("write {}", list.display()))?;
    let cleanup = || {
        let _ = std::fs::remove_file(&list);
        let _ = std::fs::remove_file(&errlog);
    };
    let mut cmd = if cfg!(windows) {
        std::process::Command::new(&exe)
    } else {
        let mut c = std::process::Command::new("nice");
        c.args(["-n", "19"]).arg(&exe);
        c
    };
    cmd.arg("embed").arg("--path-list").arg(&list).arg("--force");
    // stderr goes to a FILE: a pipe nobody drains fills at ~64 KB and blocks the child while we
    // poll `try_wait`, which would hold the index lock until the timeout.
    cmd.stdout(std::process::Stdio::null()).stderr(match std::fs::File::create(&errlog) {
        Ok(f) => std::process::Stdio::from(f),
        Err(_) => std::process::Stdio::null(),
    });
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0000_4000 | 0x0800_0000); // BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            cleanup();
            return Err(e).context("spawn recall embed");
        }
    };
    let minutes: u64 = std::env::var("RECALL_REFRESH_EMBED_TIMEOUT_MIN")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(120)
        .clamp(1, 24 * 60);
    let limit = Duration::from_secs(minutes * 60);
    let start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) => {}
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                cleanup();
                return Err(e).context("wait for recall embed");
            }
        }
        if start.elapsed() > limit {
            let _ = child.kill(); // our own child, by handle - never by image name
            let _ = child.wait();
            cleanup();
            anyhow::bail!("recall embed exceeded {minutes} min and was stopped; the documents stay queued");
        }
        std::thread::sleep(Duration::from_secs(2));
    };
    let result = if status.success() {
        Ok((
            format!("ok ({} path(s))", sendable.len()),
            unsendable.into_iter().cloned().collect(),
        ))
    } else {
        let err = std::fs::read_to_string(&errlog).unwrap_or_default();
        let tail: Vec<&str> = err.lines().rev().take(3).collect();
        let tail = tail.into_iter().rev().collect::<Vec<_>>().join(" | ");
        Err(anyhow::anyhow!("recall embed exited {status}: {tail}"))
    };
    cleanup();
    result
}


/// Tell whichever daemon is (or later becomes) resident which documents' vectors changed. A file,
/// not a socket: the CLI `recall refresh` has no connection to the daemon, and without this a
/// refresh run by hand would leave the daemon's in-RAM matrix on the old vectors until restart.
///
/// One file PER WRITER, written whole and then renamed into place: a producer never holds a handle
/// on a file the consumer is about to delete, so no append can land in an unlinked file.
pub fn queue_patch(dir: Option<&Path>, ids: &[i64]) {
    if ids.is_empty() {
        return;
    }
    let d = recall_dir(dir);
    let _ = std::fs::create_dir_all(&d);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|t| t.as_nanos())
        .unwrap_or(0);
    let stem = format!("vector-patch-{}-{nanos}", std::process::id());
    let tmp = d.join(format!("{stem}.part"));
    let text: String = ids.iter().map(|i| format!("{i}\n")).collect();
    if std::fs::write(&tmp, text).is_ok() {
        let _ = std::fs::rename(&tmp, d.join(format!("{stem}.ids")));
    }
}

/// Collect every queued id. A file is deleted only after it was read successfully, so a read
/// error (or a crash) leaves it for the next call.
pub fn take_patch(dir: Option<&Path>) -> Vec<i64> {
    let d = recall_dir(dir);
    let Ok(rd) = std::fs::read_dir(&d) else { return vec![] };
    let mut ids = Vec::new();
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !(name.starts_with("vector-patch-") && name.ends_with(".ids")) {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(e.path()) {
            ids.extend(text.lines().filter_map(|l| l.trim().parse::<i64>().ok()));
            let _ = std::fs::remove_file(e.path());
        }
    }
    ids
}

/// One refresh pass. Safe to call from the daemon thread, the CLI, or a scheduled task.
pub fn run(store: &Store, root: &Path, opts: &RefreshOpts) -> Result<Outcome> {
    let Some(_lock) = IndexLock::try_acquire_in(opts.dir.as_deref())? else {
        return Ok(Outcome::Skipped("another index/refresh holds the index lock".into()));
    };
    if !opts.ignore_busy {
        if let Some(why) = (opts.busy)() {
            return Ok(Outcome::Skipped(why));
        }
    }
    let mut state = RefreshState::load_in(opts.dir.as_deref());
    state.last_attempt_unix = Some(freshness::now_secs());

    let scan = store.scan_freshness(root, opts.find_new, opts.legacy_cap)?;
    // A move is only decidable when the created-file queue came from a full enumeration.
    let apply = store.apply_flagged(root, opts.find_new)?;
    let mut report = RefreshReport::default();
    report.absorb(&scan, &apply);
    report.patch_ids = apply.tombstoned_ids.clone();

    // This pass's changes plus anything an earlier pass could not finish embedding.
    let mut to_embed: Vec<String> = state.pending_embed.clone();
    to_embed.extend(apply.reembed.iter().cloned());
    to_embed.sort();
    to_embed.dedup();

    if to_embed.is_empty() {
        report.embed_note = "nothing to re-embed".into();
        state.pending_embed.clear();
    } else if !opts.embed {
        report.embed_note = format!("skipped by request; {} doc(s) stay queued", to_embed.len());
        state.pending_embed = to_embed;
    } else {
        match embed_paths(opts.embed_exe.as_deref(), opts.dir.as_deref(), &to_embed) {
            Ok((note, unsendable)) => {
                report.reembedded_docs = to_embed.len() - unsendable.len();
                report.embed_note = if unsendable.is_empty() {
                    note
                } else {
                    format!("{note}; {} path(s) with a newline in the name are left to `recall embed`", unsendable.len())
                };
                report.reembed_ids = store.live_ids_for_paths(&to_embed)?;
                report.patch_ids.extend(report.reembed_ids.iter().copied());
                state.pending_embed.clear();
            }
            // Not fatal and not hidden: the bodies are already current and still in the stale-hash
            // queue, and these paths are retried by the next refresh.
            Err(e) => {
                report.embed_failed = true;
                report.embed_note = format!("NOT DONE - {e:#}");
                state.pending_embed = to_embed;
            }
        }
    }
    queue_patch(opts.dir.as_deref(), &report.patch_ids);

    // A run whose embed step failed is not a success: the nightly clock must not treat it as done.
    if !report.embed_failed {
        state.last_ok_unix = Some(freshness::now_secs());
    }
    state.last_summary = Some(report.summary());
    state.save_in(opts.dir.as_deref());
    Ok(Outcome::Ran(report))
}

// ── Daemon worker ───────────────────────────────────────────────────────────────

fn skip_component(c: &str) -> bool {
    c.starts_with('.') || c == "node_modules" || c == "target"
}

/// Event paths that are worth a database look. Hidden directories (.git, .obsidian) and build
/// output generate event storms that can never be indexed.
pub fn event_rel(root: &Path, p: &Path, ci: bool) -> Option<String> {
    let rel = freshness::rel_from_event(root, p, ci)?;
    (!rel.split('/').any(skip_component)).then_some(rel)
}

/// [`event_rel`] against the configured root AND its canonical form. FSEvents reports resolved
/// paths (`/private/var/...` for a vault configured as `/var/...`), and Windows can hand back a
/// `\?\` prefix or a resolved junction; against the configured spelling alone those events
/// would be dropped without a word.
pub fn event_rel_any(roots: &[PathBuf], p: &Path, ci: bool) -> Option<String> {
    roots.iter().find_map(|r| event_rel(r, p, ci))
}

fn case_insensitive_fs() -> bool {
    cfg!(any(windows, target_os = "macos"))
}

fn flush(store: &Store, root: &Path, roots: &[PathBuf], pending: &mut HashSet<PathBuf>) -> bool {
    let ci = case_insensitive_fs();
    let batch: Vec<PathBuf> = pending.drain().collect();
    let mut failed = false;
    for p in batch {
        let Some(rel) = event_rel_any(roots, &p, ci) else { continue };
        let res = if freshness::is_indexable_name(&rel) {
            store.note_path_event(root, &rel, ci).map(|_| ())
        } else if std::fs::metadata(&p).map(|m| m.is_dir()).unwrap_or(true) {
            // A directory, or a path that is already gone: it may have been a folder (`notes.v2/`
            // has a dot), and the LIKE lookup under it is cheap and harmless when it was a file.
            store.note_dir_event(root, &rel, ci).map(|_| ())
        } else {
            Ok(())
        };
        if let Err(e) = res {
            // Most likely SQLITE_BUSY behind a long index. Keep the path and retry next tick;
            // the periodic scan is the backstop if it never succeeds.
            eprintln!("recall watcher: {rel}: {e:#}");
            pending.insert(p);
            failed = true;
        }
    }
    !failed
}

/// Start the freshness worker on its own thread and return the channel that carries the doc ids a
/// refresh re-embedded, so the accept loop can patch its in-RAM vector matrix. The worker opens
/// its own `Store`: the daemon's connection (and its non-`Send` model) never leaves the main thread.
pub fn spawn_worker() -> std::sync::mpsc::Receiver<Vec<i64>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("recall-freshness".into())
        .spawn(move || {
            if let Err(e) = worker_main(tx) {
                eprintln!("recall freshness worker stopped: {e:#}");
            }
        })
        .ok();
    rx
}

fn worker_main(reload_tx: std::sync::mpsc::Sender<Vec<i64>>) -> Result<()> {
    use notify::{EventKind, RecursiveMode, Watcher};
    let store = Store::open().context("freshness worker: open store")?;
    store.conn.busy_timeout(Duration::from_secs(30)).ok();
    // A missing root would make every indexed file look deleted, so no scans until it exists.
    // Meanwhile keep forwarding vector patches: a `recall refresh` run with an explicit
    // RECALL_VAULT still has to reach this daemon's in-RAM matrix.
    let mut warned = false;
    let root = loop {
        match store::vault_root_checked() {
            Ok(r) => break r,
            Err(e) => {
                if !warned {
                    eprintln!("recall freshness worker waiting, no scans: {e:#}");
                    warned = true;
                }
                let patch = take_patch(None);
                if !patch.is_empty() && reload_tx.send(patch).is_err() {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_secs(30));
            }
        }
    };
    let mut roots = vec![root.clone()];
    if let Ok(c) = root.canonicalize() {
        let c = PathBuf::from(c.to_string_lossy().trim_start_matches("\\\\?\\").to_string());
        if c != root {
            roots.push(c);
        }
    }
    let mut state = RefreshState::load();
    state.watcher_pid = Some(std::process::id());
    state.watcher_started_unix = Some(freshness::now_secs());
    state.save();

    let (ev_tx, ev_rx) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |res| {
        let _ = ev_tx.send(res);
    })
    .context("create file watcher")?;
    let watching = match watcher.watch(&root, RecursiveMode::Recursive) {
        Ok(()) => {
            eprintln!("recall watcher: watching {}", root.display());
            true
        }
        Err(e) => {
            eprintln!("recall watcher: cannot watch {} ({e}); falling back to periodic scans", root.display());
            false
        }
    };

    // Watcher-down catch-up: anything that changed while no watcher was running.
    match store.scan_freshness(&root, true, 5_000) {
        Ok(r) => eprintln!("recall watcher: startup scan flagged {} changed, {} deleted, {} new", r.changed, r.deleted, r.new_files),
        Err(e) => eprintln!("recall watcher: startup scan failed: {e:#}"),
    }

    let quiet_for = Duration::from_secs(2);
    let tick_every = Duration::from_secs(30);
    let fallback_scan_every = Duration::from_secs(600);
    let mut pending: HashSet<PathBuf> = HashSet::new();
    let mut last_event = Instant::now();
    let mut last_tick = Instant::now();
    let mut last_scan = Instant::now();
    let mut force_scan = false;
    let mut backoff = Duration::from_secs(60);
    let mut not_before = Instant::now();

    loop {
        match ev_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(Ok(ev)) => {
                if !matches!(ev.kind, EventKind::Access(_)) {
                    pending.extend(ev.paths);
                    last_event = Instant::now();
                }
            }
            Ok(Err(e)) => {
                // An overflowed or errored watch cannot be trusted to have seen everything.
                eprintln!("recall watcher: error {e}; scheduling a full scan");
                force_scan = true;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
        if !pending.is_empty() && last_event.elapsed() >= quiet_for {
            flush(&store, &root, &roots, &mut pending);
        }
        if force_scan
            || (!watching && last_scan.elapsed() >= fallback_scan_every)
            || last_scan.elapsed() >= fallback_scan_every * 6
        {
            force_scan = false;
            let _ = store.scan_freshness(&root, true, 5_000).map_err(|e| eprintln!("recall watcher: scan: {e:#}"));
            last_scan = Instant::now();
        }
        // Vectors re-embedded by ANY refresh (this thread's, or a `recall refresh` run by hand).
        let patch = take_patch(None);
        if !patch.is_empty() {
            let _ = reload_tx.send(patch);
        }
        if last_tick.elapsed() < tick_every || Instant::now() < not_before {
            continue;
        }
        last_tick = Instant::now();

        let settings = Settings::load();
        let counts = store.freshness_counts().unwrap_or_default();
        let has_pending = counts.changed + counts.deleted + counts.new_pending > 0;
        let state_now = RefreshState::load();
        let due = is_due(
            &settings,
            state_now.last_ok_unix,
            freshness::now_secs(),
            has_pending,
            pending.is_empty() && last_event.elapsed() >= Duration::from_secs(60),
        );
        if !due {
            continue;
        }
        match run(&store, &root, &RefreshOpts::default()) {
            Ok(Outcome::Ran(rep)) => {
                eprintln!("recall refresh: {}", rep.summary());
                if rep.embed_failed {
                    // Retry later, not every tick: a persistent failure (model missing, benchmark
                    // gate) must not respawn the embedder every 30 s.
                    not_before = Instant::now() + backoff;
                    backoff = (backoff * 2).min(Duration::from_secs(1_800));
                } else {
                    backoff = Duration::from_secs(60);
                }
            }
            Ok(Outcome::Skipped(why)) => {
                eprintln!("recall refresh: skipped ({why}); retrying in {}s", backoff.as_secs());
                not_before = Instant::now() + backoff;
                backoff = (backoff * 2).min(Duration::from_secs(1_800));
            }
            Err(e) => {
                eprintln!("recall refresh: failed: {e:#}");
                not_before = Instant::now() + backoff;
                backoff = (backoff * 2).min(Duration::from_secs(1_800));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(mode: &str) -> Settings {
        let mut x = Settings::default();
        x.apply("refresh", mode).unwrap();
        x
    }

    #[test]
    fn every_n_is_due_only_after_n_minutes() {
        let st = s("every 30 minutes");
        assert!(is_due(&st, None, 1_000_000, false, true), "never ran: due");
        assert!(!is_due(&st, Some(1_000_000 - 29 * 60), 1_000_000, false, true));
        assert!(is_due(&st, Some(1_000_000 - 30 * 60), 1_000_000, false, true));
    }

    #[test]
    fn nightly_fires_once_per_day_after_the_configured_time() {
        let st = s("nightly"); // 07:00 UTC
        let day = 20_000 * 86_400;
        let at = |h: i64, m: i64| day + h * 3_600 + m * 60;
        assert!(!is_due(&st, Some(at(7, 5)), at(23, 0), false, true), "already ran today");
        assert!(is_due(&st, Some(at(7, 5) - 86_400), at(7, 1), false, true), "ran yesterday, 07:00 passed");
        assert!(!is_due(&st, Some(at(7, 5) - 86_400), at(6, 59), false, true), "before today's trigger, yesterday's ran");
        assert!(is_due(&st, Some(at(6, 0) - 86_400), at(6, 59), false, true), "missed yesterday's trigger");
    }

    #[test]
    fn next_due_is_the_first_second_is_due_turns_true() {
        let day = 20_000 * 86_400;
        let at = |h: i64, m: i64| day + h * 3_600 + m * 60;
        // nightly, already ran today after 07:00 -> tomorrow 07:00; ran yesterday, before 07:00 -> today 07:00
        // incl. last_ok in the future (clock rolled back) and last_ok exactly on a trigger
        for (last, now) in [(at(7, 5), at(23, 0)), (at(7, 5) - 86_400, at(6, 59)), (at(8, 0), at(9, 0)),
                            (at(8, 0) + 86_400, at(9, 0)), (at(7, 0), at(7, 30))] {
            let st = s("nightly");
            let t = next_due(&st, Some(last), now).unwrap();
            assert!(t > now && !is_due(&st, Some(last), t - 1, false, true) && is_due(&st, Some(last), t, false, true));
        }
        let st = s("every 30 minutes");
        let t = next_due(&st, Some(1_000_000), 1_000_100).unwrap();
        assert_eq!(t, 1_000_000 + 30 * 60);
        assert!(!is_due(&st, Some(1_000_000), t - 1, false, true) && is_due(&st, Some(1_000_000), t, false, true));
        assert_eq!(next_due(&st, None, 5), Some(5), "never ran: due now");
        assert_eq!(next_due(&s("on-change"), Some(1), 5), None);
    }

    #[test]
    fn on_change_needs_pending_work_and_a_quiet_period() {
        let st = s("on-change");
        assert!(!is_due(&st, None, 1, false, true), "nothing pending");
        assert!(!is_due(&st, None, 1, true, false), "files still being written");
        assert!(is_due(&st, None, 1, true, true));
    }

    #[test]
    fn event_paths_in_hidden_or_build_dirs_are_ignored() {
        let root = Path::new("/v");
        assert_eq!(event_rel(root, Path::new("/v/notes/a.md"), false).as_deref(), Some("notes/a.md"));
        assert_eq!(event_rel(root, Path::new("/v/.git/index"), false), None);
        assert_eq!(event_rel(root, Path::new("/v/proj/target/x.md"), false), None);
        assert_eq!(event_rel(root, Path::new("/elsewhere/a.md"), false), None);
    }

    #[test]
    fn an_event_under_the_canonical_root_is_not_dropped() {
        let roots = vec![PathBuf::from("/var/vault"), PathBuf::from("/private/var/vault")];
        assert_eq!(event_rel_any(&roots, Path::new("/private/var/vault/n/a.md"), false).as_deref(), Some("n/a.md"));
        assert_eq!(event_rel_any(&roots, Path::new("/var/vault/n/a.md"), false).as_deref(), Some("n/a.md"));
        assert_eq!(event_rel_any(&roots, Path::new("/other/a.md"), false), None);
    }

    #[test]
    fn a_second_lock_holder_is_refused_not_queued() {
        let dir = std::env::temp_dir().join(format!("recall-lock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("l.lock");
        let a = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(&path).unwrap();
        a.try_lock().unwrap();
        let b = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        assert!(matches!(b.try_lock(), Err(std::fs::TryLockError::WouldBlock)));
        drop(a);
        b.try_lock().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
