// recall — Local-first cross-platform brain memory — FTS5 + in-process BGE vectors + graph + rg fusion, warm daemon, fully offline.
// SECURITY: secrets loaded from a secret store only — NO std::env::var for secrets

#[cfg(feature = "ann")]
mod ann;
mod audit;
mod bench;
mod bge;
mod ce;
mod daemon;
mod dream;
mod eval;
mod gemini;
mod judge;
mod query;
mod qwen3;
mod refresh;
mod reply;
mod scope;
mod scrub;
mod settings;
mod store;
mod ui;
#[cfg(feature = "voyage")]
mod voyage;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use store::Store;

const DEFAULT_EMBED_DOCTOR_TIMEOUT_SECS: u64 = 20;
const DEFAULT_MIN_EMBED_FREE_GB: f64 = 1.0;
const BYTES_PER_GB: f64 = 1_000_000_000.0;

// ── Vault secret loading ───────────────────────────────────────────────────────
// DO NOT use std::env::var() to load secrets.
// DO NOT use dotenv or any env-based secret loader.
// All secrets MUST be retrieved via cogent vault subprocess.
#[allow(dead_code)] // Reserved for future subcommands that require authenticated vault secrets.
fn vault_secret(key: &str) -> Result<String> {
    // Bounded spawn so a wedged cogent daemon can't freeze a caller.
    const VAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);
    let mut child = std::process::Command::new("cogent")
        .args(["secrets", "get", key])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("cogent not found — install cogent CLI: {}", e))?;

    let deadline = std::time::Instant::now() + VAULT_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            _ => {
                let _ = child.kill();
                return Err(anyhow::anyhow!(
                    "cogent secrets get {}: timed out after {}s",
                    key,
                    VAULT_TIMEOUT.as_secs()
                ));
            }
        }
    }

    let output = child
        .wait_with_output()
        .map_err(|e| anyhow::anyhow!("cogent wait_with_output: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!(
            "cogent secrets get {}: {}",
            key,
            stderr.trim()
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

// ── CLI definition ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "recall",
    version = "0.1.0",
    about = "Local-first cross-platform brain memory — FTS5 + in-process BGE vectors + graph + rg fusion, warm daemon, fully offline."
)]
struct Cli {
    /// Colour in terminal output. auto: only on a terminal, off when NO_COLOR is set, on when
    /// CLICOLOR_FORCE is set. Give it before the command (`recall --color never query ...`).
    #[arg(long, global = true, value_enum, default_value_t = reply::Color::Auto, value_name = "WHEN")]
    color: reply::Color,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Search the requested channels (default fts5,bge: keyword + BGE vectors) and print one fused ranking.
    Query {
        // allow_hyphen_values: a SEARCH QUERY is arbitrary user text and may
        // legitimately start with '-'. Without this, clap eats it as a flag:
        //     recall query "--no-daemon"
        // parsed as the --no-daemon FLAG, not as text, which forced the ~59s cold
        // vector load on every call. That is a one-line denial of service typed
        // into any search box wired to this binary, and every other flag the CLI
        // exposes was reachable the same way.
        //
        // Found while wiring a JavaScript frontend to this binary. execFile()
        // stops SHELL injection but does nothing about ARGV injection - the caller
        // must also pass `--` before user text, and callers get that wrong. This
        // makes the binary safe even when they do.
        #[arg(allow_hyphen_values = true)]
        text: String,
        // 20, not 5. This is a product call - five hits is a thin basis to answer from - and
        // the measurements below say only what it COSTS, not that it retrieves better. No
        // relevance claim is made here; recall@k is `Eval`'s job.
        //
        // Measured on the real corpus, denoise off, same query, five runs each (medians):
        //   warm daemon   top 5 = 103 ms   top 20 = 96 ms   top 50 = 104 ms
        //   cold in-proc  top 5 = 9577 ms  top 20 = 9504 ms
        //
        // Read these as exactly what they are: whole-request medians over five runs, with no
        // per-stage timing and no uncertainty estimate. The top-20 medians came out LOWER
        // than the top-5 medians. That is the observation. It does not bound the ranking cost
        // of the extra candidates, and it does not establish that such a cost would have been
        // detectable here - noise or movement in another stage could mask one.
        //
        // `fusion_depth` rises 80 -> 120 candidates, which CAN in principle reorder the top
        // of the ranking. Checked, not assumed: over 6 queries x 2 channels, the first five
        // of a top-20 result were identical to a top-5 result in 12 cases out of 12.
        //
        // One cost IS measured rather than inferred: the response grows 2,227 -> 8,648 bytes
        // for one query, ~3.9x. A caller on a token budget should pass `--top 5` explicitly.
        //
        // Deliberately NOT changed on `Eval`, which keeps k=5: moving it would silently
        // redefine every recall@k number this project has ever recorded.
        #[arg(long, short = 'k', default_value = "20")]
        top: Option<String>,
        // NO clap default_value here on purpose. A default_value makes this
        // Option always Some, so the resolution logic in cmd_query (env override
        // + feature-aware default) becomes unreachable dead code. That is exactly
        // the bug that made `voyage` look like it was ignored: `--channels voyage`
        // worked while the default silently stayed on fts5.
        //
        // The real default now lives in cmd_query::DEFAULT_CHANNELS.
        //
        // Historical note kept: the legacy `bge` channel does an unindexed in-RAM
        // cosine scan over every stored vector (bge.rs BgeSearch) and on the full
        // corpus pinned all cores. `voyage` does not have that problem — it holds
        // a resident copy and scans it in parallel in ~235ms warm.
        #[arg(long, short = 'c')]
        channels: Option<String>,
        /// Force routing through the warm daemon (error if daemon not running).
        /// Without this flag, daemon is used automatically when available.
        #[arg(long)]
        via_daemon: bool,
        /// Force the in-process path even when a daemon marker exists.
        #[arg(long)]
        no_daemon: bool,
        /// Emit results as a JSON array instead of human-readable text.
        /// Fields: doc_id, rank (0-based), score, vault_path, title, snippet, source, ranks,
        /// scores, rrf (per-channel provenance), plus the freshness fields (stale, stale_note, ...).
        /// Zero-cloud: this flag only changes output format, no network calls.
        #[arg(long, short = 'j')]
        json: bool,
        /// When the bge channel cannot load (e.g. the ONNX Runtime library is missing), serve the fts5
        /// rescue tier instead of exiting non-zero. Prints a `RESCUE:` stderr line and labels
        /// those results `source: "fts5-rescue"`. Skips the daemon, which has no per-request
        /// fallback switch.
        #[arg(long)]
        allow_fallback: bool,
        /// Print per-hit scores, missing-term diagnostics and start-up narration, as
        /// RECALL_VERBOSE=1 does. Warnings and errors print either way.
        #[arg(long)]
        verbose: bool,
    },
    /// Capture an observation into the memory store. Keyword-searchable at once; `recall embed` adds its vector.
    Save {
        text: String,
        #[arg(long, short = 's', default_value = "cli")]
        source: Option<String>,
        /// Which box an agent's memory goes to: agent | project | team | global. Default: the
        /// agent's project (its own box when RECALL_PROJECT is unset). The operator saves to global.
        #[arg(long)]
        share: Option<String>,
    },
    /// Read file bytes on stdin, write the exact text `recall index` would STORE for them (the
    /// scrub layers applied when `scrub.store_redacted` is on). Lets a caller confirm the index
    /// holds a file's current text without re-implementing the redaction.
    #[command(hide = true)]
    StoredForm {},
    /// Print the tenancy reader this process runs as (agent and the boxes it may read).
    Whoami {},
    /// Manage the warm resident daemon (holds embedding model + DB handles).
    Daemon { action: String },
    /// Maintainer tool: run the eval harness against a fixture set that is not shipped (needs `bench_fixtures` in ~/.recall/private-rules.json).
    Bench {
        #[arg(long, short = 'o', default_value = "false")]
        overnight: Option<String>,
        #[arg(long, short = 'n', default_value = "50")]
        iterations: Option<String>,
        /// Score only the first N positive fixtures (fast CI smoke). 0 = all.
        #[arg(long, short = 's', default_value = "0")]
        smoke: Option<String>,
        /// Write a redacted machine-readable benchmark artifact under eval/results/bench_release.
        #[arg(long)]
        json_out: bool,
    },
    /// Build or refresh the keyword (FTS5) index from the notes folder. Vectors are a separate step: `recall embed`.
    Index {
        #[arg(long, short = 'r', default_value = "false")]
        reset: Option<String>,
        /// Read vault-relative paths from a newline-delimited file and index only those docs.
        #[arg(long)]
        path_list: Option<PathBuf>,
        /// Count/index candidates without reading bodies or mutating recall.db.
        #[arg(long)]
        dry_run: bool,
        /// Tombstone documents whose file has been deleted: off | dry | apply.
        /// `dry` writes the full would-tombstone list and changes nothing.
        #[arg(long, default_value = "off")]
        prune_missing: String,
    },
    /// Vectorize indexed docs with BGE-small (resumable; only docs with no vector or a vector made from older text).
    Embed {
        #[arg(long, short = 'b', default_value = "256")]
        batch: Option<String>,
        /// Read vault-relative paths from a newline-delimited file and embed only those indexed docs.
        #[arg(long)]
        path_list: Option<PathBuf>,
        /// Count candidates without loading BGE or mutating recall.db.
        #[arg(long)]
        dry_run: bool,
        /// Override the benchmark regression gate for real embedding mutations.
        #[arg(long)]
        allow_benchmark_regression: bool,
        /// Re-embed path-list docs that ALREADY have chunk vectors. Required after a chunker
        /// change: without it every document is already "done" and the fix reaches none of them.
        #[arg(long, requires = "path_list")]
        force: bool,
    },
    /// Maintainer migration: backfill `documents.content_sha256` for rows that lack it.
    ///
    /// Idempotent and resumable: it selects only rows where the column is NULL, so a re-run
    /// continues where an interrupted one stopped and a finished corpus is a no-op. Reads
    /// `body`, writes one column, touches no vector.
    BackfillHashes {
        /// Rows per transaction. Larger is faster but holds a write lock longer.
        #[arg(long, default_value_t = 5_000)]
        batch: usize,
    },
    /// Maintainer migration: record that existing vectors are current, behind a guard. Does NOT embed anything.
    ///
    /// The queue can read "N stale" purely because `embedded_sha256` was never populated —
    /// a missing RECORD of what the vectors were built from, not evidence they are wrong. When
    /// they really are current this writes that record in ~5 seconds instead of re-embedding for
    /// ~20 hours. Because "when they really are current" is a claim this command cannot verify by
    /// itself, it refuses to run unless every precondition holds, and re-checks afterwards.
    StampVectorsCurrent {
        /// Refuse unless the live document count is exactly this (required: state the count you expect).
        #[arg(long)]
        expect_live_docs: i64,
        /// Refuse unless the chunk-vector count is exactly this (required).
        #[arg(long)]
        expect_chunk_vectors: i64,
        /// Refuse unless this backup file exists AND opens AND holds the same document count.
        #[arg(long)]
        backup: PathBuf,
        /// Re-embed this many random documents first and require every vector to come back
        /// byte-identical. 0 skips it. This is the only check that tests the actual claim.
        #[arg(long, default_value_t = 0)]
        verify_sample: usize,
        /// Run every check and print the plan, then stop without writing.
        #[arg(long)]
        dry_run: bool,
    },
    /// Maintainer experiment: load Qwen3-Embedding-0.6B and prove the wiring, before anything depends on it.
    ///
    /// The whole point is the cosine pair: the same sentence twice must come back ~1.0, and two
    /// unrelated sentences must come back well below 0.9. A broken pooling wire or a missing EOS
    /// makes EVERYTHING similar to everything, with no error anywhere, so this is the only cheap
    /// check that catches it.
    Qwen3Check {
        /// Refuse (exit non-zero) unless identical text scores at least this.
        #[arg(long, default_value_t = 0.99)]
        min_self: f32,
        /// Refuse (exit non-zero) unless unrelated text scores below this.
        #[arg(long, default_value_t = 0.90)]
        max_unrelated: f32,
    },
    /// Maintainer experiment: throughput + VRAM bake-off for Qwen3 on REAL chunk text from the live store.
    ///
    /// Reconstructs chunks from `bge_chunk_embeddings` char offsets so the token-length
    /// distribution is that of the indexed notes, not a synthetic string that happens to be short.
    /// Reads only; writes nothing.
    Qwen3Bench {
        /// How many real chunks to embed.
        #[arg(long, default_value_t = 1_000)]
        chunks: usize,
        /// Texts per forward pass.
        #[arg(long, default_value_t = 16)]
        batch: usize,
        /// Total chunk rows the full re-embed would cover, used for the projection (required).
        #[arg(long)]
        project_over: u64,
    },
    /// Maintainer experiment: what does a BIGGER chunk cost? Re-splits real document bodies at a target
    /// size and embeds them, so the answer comes from the indexed notes rather than from arithmetic.
    ///
    /// BGE forced 350-token chunks because it could not see past 512. Qwen3 can, and the whole
    /// point of the swap was to stop cutting sections in half. Total TEXT is fixed, so the real
    /// question is documents/hour and VRAM, not chunks/sec - a number that improves automatically
    /// when chunks get bigger and means nothing on its own.
    Qwen3ChunkSweep {
        /// Documents to run through every arm. The same documents in each, so arms compare.
        #[arg(long, default_value_t = 60)]
        docs: usize,
        /// Target chunk sizes in characters. Repeat the flag or comma-separate.
        #[arg(long, value_delimiter = ',', default_values_t = [1800usize, 3600, 7200])]
        target_chars: Vec<usize>,
        /// Texts per forward pass.
        #[arg(long, default_value_t = 16)]
        batch: usize,
        /// Live documents the full run would cover, for the projection (required).
        #[arg(long)]
        project_docs: u64,
    },
    /// Maintainer experiment: embed documents with Qwen3 into `qwen3_chunk_embeddings`. Resumable; safe to interrupt.
    ///
    /// Writes ONLY that table. BGE's vectors are never touched, so `--channels bge` keeps working
    /// throughout and the swap stays reversible.
    Qwen3Embed {
        /// Stop after this many documents. 0 means the whole queue.
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Chunks per forward pass.
        #[arg(long, default_value_t = 16)]
        batch: usize,
        /// Count the queue and print the plan, then stop without writing.
        #[arg(long)]
        dry_run: bool,
    },
    /// Local health checks for recall subsystems.
    Doctor {
        /// Check to run: embed or health. Maintainer checks (need `recall_dev_dir` in ~/.recall/private-rules.json): memory, coverage, path-probe, embedding-backlog, safe-embed-batch.
        #[arg(default_value = "embed")]
        check: String,
        /// Kill child-process BGE probes if they exceed this many seconds.
        #[arg(long, default_value = "20")]
        timeout_seconds: Option<String>,
        /// Emit machine-readable JSON.
        #[arg(long, short = 'j')]
        json: bool,
        /// Vault-relative approved path-list for `doctor path-probe`.
        #[arg(long)]
        path_list: Option<PathBuf>,
        /// Redacted probe artifact path for `doctor path-probe`.
        #[arg(long)]
        summary_json: Option<PathBuf>,
        /// Require selected paths to have embeddings for `doctor path-probe`.
        #[arg(long)]
        require_embedded: bool,
        /// Batch size for `doctor embedding-backlog`.
        #[arg(long, default_value = "64")]
        batch_size: Option<String>,
        /// Optional per-prefix cap for `doctor embedding-backlog`.
        #[arg(long)]
        max_per_prefix: Option<String>,
        /// Optional unembedded JSONL artifact for `doctor embedding-backlog`.
        #[arg(long)]
        unembedded_jsonl: Option<PathBuf>,
        /// Actually run embedding mutation for `doctor safe-embed-batch`.
        #[arg(long)]
        apply: bool,
        /// Run coverage/RAGAS/enterprise postchecks for `doctor safe-embed-batch`.
        #[arg(long)]
        postcheck: bool,
        /// Redacted scan summary whose rejected path hashes should be skipped by `doctor safe-embed-batch`.
        #[arg(long)]
        skip_scan_rejected_json: Vec<PathBuf>,
        /// Also skip rejected hashes from recent redacted safe batch scan summaries.
        #[arg(long)]
        skip_known_rejected: bool,
        /// Maximum recent rejected-scan summaries to load for `doctor safe-embed-batch`.
        #[arg(long, default_value = "64")]
        known_rejected_scan_limit: Option<String>,
    },
    #[command(name = "__embed-doctor-worker", hide = true)]
    EmbedDoctorWorker {},
    /// Chunk documents and report the chunk statistics WITHOUT embedding or writing anything.
    ///
    /// The pre-flight for a backfill. Chunking is cheap and pure; embedding is neither, so the
    /// properties the chunker is supposed to guarantee — every chunk under `CHUNK_MAX_TOKENS`,
    /// a heading path on every chunk that has a heading above it, summary blocks split out —
    /// are all checkable before a single vector is computed or a single row is written.
    ///
    /// Hidden because it is an operator/CI tool, not part of the user-facing surface.
    #[command(name = "__chunk-report", hide = true)]
    ChunkReport {
        /// Newline-delimited vault-relative paths to chunk (same format as `embed --path-list`).
        #[arg(long)]
        path_list: PathBuf,
        /// Stop after this many documents.
        #[arg(long, default_value = "200")]
        limit: Option<String>,
        /// Write the full JSON report here. Without it only the summary is printed.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// RecallDreamer — offline consolidation: reconcile near-dupes, supersede stale memories.
    Dream {},
    /// Re-sync the index with files that changed on disk: flag edits/deletes/renames/new files,
    /// rewrite changed documents in place (doc ids kept), then re-embed only those documents.
    /// Takes the single-writer index lock and skips (exit 0, says so) if another index holds it.
    Refresh {
        /// Only flag what changed (the watcher-down fallback scan); rewrite and embed nothing.
        #[arg(long)]
        scan_only: bool,
        /// Re-index changed documents but leave their re-embedding to the next `recall embed`.
        #[arg(long)]
        no_embed: bool,
        /// Run even if the machine is busy.
        #[arg(long)]
        force: bool,
        /// Only run when the `refresh` setting says one is due (for a scheduled task).
        #[arg(long)]
        if_due: bool,
        /// Print the pending-change counts and last run, then stop.
        #[arg(long)]
        status: bool,
        #[arg(long, short = 'j')]
        json: bool,
    },
    /// View or change settings in ~/.recall/settings.json: `settings`, `settings get KEY`,
    /// `settings set KEY VALUE` (e.g. `settings set refresh "every 2 hours"`, `settings set scrub.pii off`).
    Settings {
        action: Option<String>,
        key: Option<String>,
        value: Option<String>,
    },
    /// Local web page (127.0.0.1, random port, per-launch token): status, settings, test search.
    Ui {
        /// Print the URL but do not open a browser.
        #[arg(long)]
        no_open: bool,
    },
    /// Store health: doc counts, per-source breakdown, superseded count, DB size.
    Status {
        /// Emit machine-readable JSON instead of human output (for CI smoke tests).
        #[arg(long, short = 'j')]
        json: bool,
        /// Exit non-zero if embedding coverage (%) falls below this threshold (CI gate).
        /// Off by default — without it, status only reports and always exits 0.
        #[arg(long)]
        min_coverage: Option<f64>,
    },
    /// Measure recall@k of the fast vector search path against an exact scan of the
    /// same vectors, over a short list of built-in probe phrases. 1.0 by construction unless
    /// the build has the `ann` feature; it does not measure retrieval quality on
    /// your notes.
    Eval {
        #[arg(long, short = 'k', default_value = "5")]
        top: Option<String>,
        /// Exit non-zero if mean recall@k falls below this threshold (CI gate).
        /// Off by default — without it, eval only reports and always exits 0.
        #[arg(long)]
        min_recall: Option<f32>,
    },
    /// Surface recurring themes across saved memories (offline clustering).
    Insights {
        #[arg(long, short = 'm', default_value = "2")]
        min: Option<String>,
    },
    /// Show the most recent saved observations.
    Recent {
        #[arg(long, short = 'n', default_value = "10")]
        count: Option<String>,
    },
    /// Retire memories matching text (bi-temporal — reversible, not deleted).
    Forget { text: String },
    /// Retire a single memory by id.
    Supersede { id: String },
    /// Dump all valid memories as JSON (backup / piping).
    Export {},
    /// Prune embeddings orphaned from tombstoned/superseded documents.
    /// Dry-run by default — pass --apply to execute deletions.
    Gc {
        /// Execute the deletions (overrides default dry-run mode).
        #[arg(long)]
        apply: bool,
        /// Dry-run count for live embeddings whose path matches the current exclude policy.
        #[arg(long)]
        exclude_aware: bool,
    },
    /// Build (or rebuild) the HNSW ANN index from existing BGE vectors.
    /// Only available when compiled with --features ann.
    /// Does NOT re-embed anything — reads existing vectors from recall.db.
    #[cfg(feature = "ann")]
    IndexAnn {},
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    // Point ort (load-dynamic) at the onnxruntime.dll shipped beside recall.exe,
    // BEFORE any embedding code runs. Without this, ort would search the OS load
    // path and could pick up a stale C:\Windows\System32\onnxruntime.dll and hang.
    // We pin the beside-exe path unless the user already set ORT_DYLIB_PATH.
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let dll = dir.join(bge::ort_lib_filename());
                // Pin ORT at the beside-exe path WHETHER OR NOT it exists. This is
                // the canonical location; the BGE channel checks the same path at
                // load time (bge::ort_runtime_available) and skips itself when the
                // DLL is absent, rather than letting ort fall through to the OS path
                // (stale System32 onnxruntime.dll → silent wedge at session init).
                // (a stale System32 onnxruntime.dll wedges at session init)
                std::env::set_var("ORT_DYLIB_PATH", dll);
            }
        }
    }

    let cli = Cli::parse();
    reply::init(cli.color);

    // `recall query ... | head -1`: the reader closes the pipe and the next println! panics with
    // "failed printing to stdout: Broken pipe". Nothing went wrong, so end quietly. Only that one
    // panic is touched, and not in the long-running commands, whose stdout is not their purpose.
    // SIGPIPE itself stays ignored: a client that hangs up must never kill a daemon.
    let one_shot = !matches!(cli.command, Commands::Daemon { .. } | Commands::Ui { .. });
    if one_shot {
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = info.payload().downcast_ref::<String>().map_or("", String::as_str);
            let closed = ["os error 32)", "os error 109)", "os error 232)"].iter().any(|e| msg.ends_with(e));
            reply::reset();
            if msg.starts_with("failed printing to stdout") && closed {
                std::process::exit(0);
            }
            default_hook(info);
        }));
    }

    // Preflight or refuse: a broken private rules file would silently re-tag documents.
    if let Err(e) = store::check_private_rules() {
        eprintln!("recall: private rules file is invalid, refusing to run: {e:#}");
        std::process::exit(1);
    }

    if let Err(e) = run(cli) {
        // The same closed pipe, reached through a `write!` that returns its error (`recall export | head`).
        let closed = |io: &std::io::Error| io.kind() == std::io::ErrorKind::BrokenPipe;
        if one_shot && e.downcast_ref::<std::io::Error>().is_some_and(closed) {
            std::process::exit(0);
        }
        audit::log_error("main", &e.to_string());
        reply::blocking("main", &e.to_string());
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    // Tenancy: resolve the reader ONCE, before any command touches the store. Fail-closed: a bad
    // RECALL_AGENT / RECALL_PROJECT / RECALL_TEAM stops here with exit 1 and zero hits.
    scope::init_from_env()?;
    // DEFAULT-DENY for scoped readers: only the commands that are scoped end to end are allowed.
    // Every other command (index, refresh, embed, gc, doctor, dream, insights, bench, eval, the
    // hidden reports, ...) reads or rewrites the whole corpus (Codex tenancy b3), and any command
    // added later is refused until it is deliberately listed here.
    if !scope::current().is_operator()
        && !matches!(
            cli.command,
            Commands::Query { .. }
                | Commands::Save { .. }
                | Commands::Recent { .. }
                | Commands::Export {}
                | Commands::Forget { .. }
                | Commands::Supersede { .. }
                | Commands::Status { .. }
                | Commands::Whoami {}
        )
    {
        anyhow::bail!(
            "recall: that command is operator-only (it reads or changes the whole corpus); a scoped \
             reader (RECALL_AGENT set) may run query, save, recent, export, forget, supersede, \
             status, whoami"
        );
    }
    match cli.command {
        Commands::Query {
            text,
            top,
            channels,
            via_daemon,
            no_daemon,
            json,
            allow_fallback,
            verbose,
        } => {
            if verbose {
                std::env::set_var("RECALL_VERBOSE", "1"); // before any thread starts; bge::chatter reads it
            }
            cmd_query(text, top, channels, via_daemon, no_daemon, json, allow_fallback)?;
        }
        Commands::Save { text, source, share } => {
            cmd_save(text, source, share)?;
        }
        Commands::Whoami {} => {
            println!("{}", scope::current().describe());
        }
        Commands::StoredForm {} => {
            use std::io::{Read, Write};
            if !scrub::policy_initialized() {
                anyhow::bail!("stored-form: no redaction policy yet (new profile, nothing indexed); run `recall index` first");
            }
            let mut bytes = Vec::new();
            std::io::stdin().read_to_end(&mut bytes)?;
            // raw bytes out: no newline added, no CRLF translation
            std::io::stdout().write_all(store::stored_form(bytes).as_bytes())?;
        }
        Commands::Daemon { action } => {
            cmd_daemon(action)?;
        }
        Commands::Bench {
            overnight,
            iterations,
            smoke,
            json_out,
        } => {
            cmd_bench(overnight, iterations, smoke, json_out)?;
        }
        Commands::Index {
            reset,
            path_list,
            dry_run,
            prune_missing,
        } => {
            cmd_index(reset, path_list, dry_run, prune_missing)?;
        }
        Commands::Embed {
            batch,
            path_list,
            dry_run,
            allow_benchmark_regression,
            force,
        } => {
            cmd_embed(batch, path_list, dry_run, allow_benchmark_regression, force)?;
        }
        Commands::BackfillHashes { batch } => {
            cmd_backfill_hashes(batch)?;
        }
        Commands::StampVectorsCurrent {
            expect_live_docs,
            expect_chunk_vectors,
            backup,
            verify_sample,
            dry_run,
        } => {
            cmd_stamp_vectors_current(
                expect_live_docs,
                expect_chunk_vectors,
                &backup,
                verify_sample,
                dry_run,
            )?;
        }
        Commands::Qwen3Embed {
            limit,
            batch,
            dry_run,
        } => {
            cmd_qwen3_embed(limit, batch, dry_run)?;
        }
        Commands::Qwen3ChunkSweep {
            docs,
            target_chars,
            batch,
            project_docs,
        } => {
            cmd_qwen3_chunk_sweep(docs, &target_chars, batch, project_docs)?;
        }
        Commands::Qwen3Bench {
            chunks,
            batch,
            project_over,
        } => {
            cmd_qwen3_bench(chunks, batch, project_over)?;
        }
        Commands::Qwen3Check {
            min_self,
            max_unrelated,
        } => {
            cmd_qwen3_check(min_self, max_unrelated)?;
        }
        Commands::Doctor {
            check,
            timeout_seconds,
            json,
            path_list,
            summary_json,
            require_embedded,
            batch_size,
            max_per_prefix,
            unembedded_jsonl,
            apply,
            postcheck,
            skip_scan_rejected_json,
            skip_known_rejected,
            known_rejected_scan_limit,
        } => {
            cmd_doctor(
                check,
                timeout_seconds,
                json,
                path_list,
                summary_json,
                require_embedded,
                batch_size,
                max_per_prefix,
                unembedded_jsonl,
                apply,
                postcheck,
                skip_scan_rejected_json,
                skip_known_rejected,
                known_rejected_scan_limit,
            )?;
        }
        Commands::EmbedDoctorWorker {} => {
            cmd_embed_doctor_worker()?;
        }
        Commands::ChunkReport {
            path_list,
            limit,
            out,
        } => {
            cmd_chunk_report(&path_list, &limit, out.as_deref())?;
        }
        Commands::Dream {} => {
            cmd_dream()?;
        }
        Commands::Status { json, min_coverage } => {
            cmd_status(json, min_coverage)?;
        }
        Commands::Refresh { scan_only, no_embed, force, if_due, status, json } => {
            cmd_refresh(scan_only, no_embed, force, if_due, status, json)?;
        }
        Commands::Settings { action, key, value } => {
            cmd_settings(action, key, value)?;
        }
        Commands::Ui { no_open } => {
            ui::run(!no_open)?;
        }
        Commands::Eval { top, min_recall } => {
            cmd_eval_recall(top, min_recall)?;
        }
        Commands::Insights { min } => {
            cmd_insights(min)?;
        }
        Commands::Recent { count } => {
            cmd_recent(count)?;
        }
        Commands::Forget { text } => {
            cmd_forget(text)?;
        }
        Commands::Supersede { id } => {
            cmd_supersede(id)?;
        }
        Commands::Export {} => {
            cmd_export()?;
        }
        Commands::Gc {
            apply,
            exclude_aware,
        } => {
            cmd_gc(apply, exclude_aware)?;
        }
        #[cfg(feature = "ann")]
        Commands::IndexAnn {} => {
            cmd_index_ann()?;
        }
    }
    Ok(())
}

// ── JSON output type for `recall query --json` ────────────────────────────────

/// Wire format emitted by `recall query --json`. Field names match what
/// multi_rag_router.py's `_LocalHit` / `_parse_recall_text` expects, letting
/// callers switch from text-parsing to `json.loads()` with no schema change.
///   doc_id     — database row id (i64)
///   rank       — 0-based position in result list
///   score      — BM25/cosine score from the winning channel (higher = better)
///   vault_path — vault-relative path (forward-slash); empty for ad-hoc saves
///   title      — document title
///   snippet    — FTS5/bge excerpt (may be empty for path-only hits)
///   source     — the channel(s) that ranked this hit, '+'-joined: "fts5", "bge", "fts5+bge",
///                "fts5-rescue" (a requested channel failed); "unknown" from a pre-source daemon
///   ranks      — {channel: 0-based position in that channel's list}, one per channel in source
///   scores     — {channel: RAW score}: fts5 / fts5-rescue -bm25 (higher = better), bge cosine.
///                A channel with no raw score (rg, graph, qwen3, gemini, voyage) has no key.
///   rrf        — sum of 1/(k + rank + 1) over `ranks` (the default fusion's sort value)
/// `score` is the legacy single value from the first channel that surfaced the doc; mixed scales.
#[derive(serde::Serialize)]
struct QueryJsonHit {
    doc_id: i64,
    rank: usize,
    score: f64,
    vault_path: String,
    title: String,
    snippet: String,
    source: String,
    ranks: std::collections::BTreeMap<String, usize>,
    scores: std::collections::BTreeMap<String, f64>,
    rrf: f64,
    /// Query subject terms this hit lacks (`query::Provenance::lacks`); omitted when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    lacks: Vec<String>,
    /// Same-file doc ids collapsed into this hit (`query::Provenance::merged`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    merged: Vec<i64>,
    /// `query::Provenance::labels_truncated`; omitted when false.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    labels_truncated: bool,
    /// Where the scrub layers redacted something in the ORIGINAL file: `{kind, line}`, never the
    /// value. Omitted when nothing is redacted. Open `vault_path` at `line` for the original.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    redactions: Vec<scrub::Redaction>,
    /// File-change freshness: `stale` (null = could not check), `deleted`, `modified_at`,
    /// `indexed_at`, `file_modified_at`, `stale_note`. See `store::freshness::FreshInfo`.
    #[serde(flatten)]
    fresh: store::freshness::FreshInfo,
    /// `query::Provenance::closest_match`: the query failed an abstain rule; not a confident
    /// answer. Omitted when false.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    closest_match: bool,
    /// `query::Provenance::absent`: main query words no note holds. Omitted when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    absent: Vec<String>,
}

/// Human output for a closest-match answer (lme-improve R2): stated above the list, whatever the
/// per-hit `lacks` say.
fn print_closest_banner(closest: bool) {
    if closest {
        reply::warn("No confident match: these are the closest matches, not answers.");
    }
}

/// The head of a human result list: the closest-match banner only under the `closest_labels` setting
/// (cli-quality addendum A), and the "Not found in the indexed text: ..." line whenever a main query word
/// is in no note (addendum B). Returns whether to show the closest-match label per hit.
fn print_list_head(closest: &[bool], absent: &[String], sc: &scrub::Scrubber) -> bool {
    let show = query::closest_labels_on() && closest.iter().any(|c| *c);
    print_closest_banner(show);
    if let Some(l) = query::absent_line(absent) {
        println!("  {}\n", reply::paint(reply::NOTE, &human_text(sc, &l)));
    }
    show
}

/// Re-order `v` by `order` (indices into `v`, each used once).
fn permute<T>(v: Vec<T>, order: &[usize]) -> Vec<T> {
    let mut slots: Vec<Option<T>> = v.into_iter().map(Some).collect();
    order.iter().filter_map(|&i| slots[i].take()).collect()
}

/// Check each hit's file against the disk and apply the optional stale/recency ranking knobs
/// (both default OFF). Every query surface calls this, so every surface reports freshness.
fn with_freshness<T>(
    hits: Vec<T>,
    id_of: impl Fn(&T) -> i64,
    excerpt_of: impl Fn(&T) -> &str,
    query: &str,
) -> (Vec<T>, Vec<store::freshness::FreshInfo>) {
    let settings = settings::Settings::load();
    let ids: Vec<i64> = hits.iter().map(&id_of).collect();
    let excerpts: Vec<&str> = hits.iter().map(&excerpt_of).collect();
    let infos = if settings.freshness_check {
        let t = std::time::Instant::now();
        let infos = store::freshness::annotate_ids_for(&ids, &excerpts, query);
        if query::prof::on() {
            eprintln!("recall profile: client freshness check of {} hit(s) {:.1}ms", ids.len(), t.elapsed().as_secs_f64() * 1000.0);
        }
        infos
    } else {
        vec![store::freshness::FreshInfo::unchecked("freshness_check is off"); ids.len()]
    };
    if settings.stale_penalty == 0.0 && settings.recency_boost == 0.0 {
        return (hits, infos);
    }
    let order = store::freshness::rerank_order(
        &infos,
        settings.stale_penalty,
        settings.recency_boost,
        store::freshness::now_secs(),
    );
    (permute(hits, &order), permute(infos, &order))
}

/// Banner above the result list. One line, plain words; the per-hit line says which files.
fn freshness_banner(infos: &[store::freshness::FreshInfo]) -> Option<String> {
    let stale = infos.iter().filter(|f| f.is_stale()).count();
    if stale > 0 {
        return Some(format!(
            "{stale} of {} results point at files that changed on disk after recall indexed them; read the actual files, not these excerpts.",
            infos.len()
        ));
    }
    infos
        .iter()
        .find(|f| f.stale.is_none())
        .and_then(|f| f.stale_note.clone())
}

/// The per-hit line, or `None` for a fresh hit.
fn freshness_line(f: &store::freshness::FreshInfo) -> Option<String> {
    match (f.stale, &f.stale_note) {
        (Some(true), Some(n)) => Some(reply::paint(if f.deleted { reply::BAD } else { reply::WARN }, &format!("!! STALE: {n}"))),
        (None, Some(n)) => Some(reply::paint(reply::NOTE, &format!("? {n}"))),
        // Changed file, unchanged passage: not stale, but say so.
        (Some(false), Some(n)) => Some(reply::paint(reply::NOTE, &format!("~ {n}"))),
        _ => None,
    }
}

/// One text line of provenance: `via fts5+bge · fts5 12.3456 · bge 0.7812 · rank-sum 0.03252`; a
/// channel with a rank but no raw score prints as `rg #3`. `rank-sum` is the RRF sum of the channel ranks
/// (`Provenance::rrf`), always computed; it is NOT what ordered the list under the default convex fusion or a
/// reranker, so it is not called "rrf" (the UI chip says the same since PR #42).
fn provenance_line(
    source: &str,
    ranks: &std::collections::BTreeMap<String, usize>,
    scores: &std::collections::BTreeMap<String, f64>,
    rrf: f64,
) -> String {
    let mut parts = vec![format!("via {source}")];
    for ch in source.split('+').filter(|c| !c.is_empty()) {
        match (scores.get(ch), ranks.get(ch)) {
            (Some(s), _) => parts.push(format!("{ch} {s:.4}")),
            (None, Some(r)) => parts.push(format!("{ch} #{}", r + 1)),
            (None, None) => {}
        }
    }
    parts.push(format!("rank-sum {rrf:.5}"));
    parts.join(" · ")
}

/// Scrub before normalization to retain ANSI/hidden-character matching, and again
/// afterwards: removing a control must not join fragments into a visible secret.
fn human_text(sc: &scrub::Scrubber, text: &str) -> String {
    sc.scrub(&reply::plain(&sc.scrub(text))).into_owned()
}

/// One hit of the human result list.
struct HitText<'a> {
    title: &'a str,
    path: &'a str,
    snippet: &'a str,
    lacks: &'a [String],
    closest: bool,
    absent: &'a [String],
    provenance: String,
}

/// The human result list; the daemon path and the in-process path both print through here. Per hit:
/// title, path, the file-change warning, notes, the excerpt, then the scoring line.
fn print_hit_list(
    text: &str,
    sc: &scrub::Scrubber,
    vault: &Path,
    fresh: &[store::freshness::FreshInfo],
    truncated: Option<&str>,
    hits: &[HitText],
) {
    let head = format!("recall query: top {} for: {}", hits.len(), human_text(sc, text));
    println!("\n{}\n", reply::paint(reply::BOLD, &reply::plain(&head)));
    let show = print_list_head(
        &hits.iter().map(|h| h.closest).collect::<Vec<_>>(),
        hits.first().map(|h| h.absent).unwrap_or(&[]),
        sc,
    );
    if let Some(b) = freshness_banner(fresh) {
        reply::warn(&b);
    }
    if let Some(n) = truncated {
        reply::warn(n);
    }
    for (i, h) in hits.iter().enumerate() {
        if i > 0 {
            println!();
        }
        let title = if h.title.is_empty() { h.path } else { h.title };
        let (weak, note) = query::hit_note(i, show && h.closest, h.lacks);
        let diagnostics = query::verbose_on();
        println!(
            "  {}. {}{}",
            i + 1,
            if weak && (show || diagnostics) { reply::paint(reply::DIM, "(weak) ") } else { String::new() },
            reply::paint(reply::BOLD, &human_text(sc, title))
        );
        if !h.path.is_empty() {
            println!("     {}", reply::paint(reply::DIM, &human_text(sc, h.path)));
        }
        if let Some(l) = freshness_line(&fresh[i]) {
            println!("     {l}");
        }
        if let Some(l) = note.filter(|_| show || diagnostics) {
            println!("     {}", human_text(sc, &l));
        }
        let ptrs = if i < scrub::MAX_POINTER_HITS { scrub::pointers(sc, vault, h.path) } else { Vec::new() };
        if !ptrs.is_empty() {
            println!("     redacted in original: {}", scrub::pointer_line(&ptrs));
        }
        if !h.snippet.is_empty() {
            println!("     {}", reply::marks(&human_text(sc, h.snippet).replace('\n', " "), text));
        }
        if diagnostics {
            println!("     {}", reply::paint(reply::DIM, &h.provenance));
        }
    }
    println!();
}

// ── Subcommand handlers ───────────────────────────────────────────────────────
/// A guarded mutation: preconditions -> plan -> mutate -> postconditions.
///
/// 🔴 Every check REFUSES. None of them warns. An exit code is not a result, so a check that
/// prints a warning and continues is worse than no check at all — it produces a green run with a
/// scrolled-past warning, which is the exact shape of every expensive failure this project has
/// had: the Vertex 403 that kept answering from keywords, the rename fix that reported
/// "0 moved (renamed)" because it was inert, the queue that claimed 20 hours of work for a job
/// that took 5 seconds.
///
/// Reused deliberately. The Qwen3 re-embed gets the same shape with different numbers: it is a
/// long, expensive, irreversible-in-practice mutation of the same tables, and the argument for
/// guarding it is identical.
struct GuardedRun<'a> {
    /// Shown in every message so a failure says which operation refused.
    what: &'a str,
    expect_live_docs: i64,
    expect_chunk_vectors: i64,
    /// A backup that must exist AND open AND agree. `None` only for operations that cannot lose
    /// data; the stamp backfill is not one of those.
    backup: Option<&'a Path>,
}

impl<'a> GuardedRun<'a> {
    /// Everything that must be true BEFORE the mutation. Returns the counts it verified, so the
    /// caller compares against the same snapshot the guard used rather than re-reading.
    fn preflight(&self, store: &Store) -> Result<store::GuardCounts> {
        let c = store.guard_counts()?;
        println!("\n{}", reply::paint(reply::BOLD, &format!("GUARD: {}", self.what)));
        println!("   live documents      {:>9}   (expect {})", c.live_docs, self.expect_live_docs);
        println!(
            "   chunk vectors       {:>9}   (expect {})",
            c.chunk_vectors, self.expect_chunk_vectors
        );
        println!("   orphaned vectors    {:>9}   (expect 0)", c.orphan_vectors);
        println!("   rows w/o a hash     {:>9}", c.content_sha_null);
        println!("   embedding queue     {:>9}", c.queue);

        if c.live_docs != self.expect_live_docs {
            anyhow::bail!(
                "REFUSING: live document count is {} but {} was expected.\n\
                 The database is not the one this run was authorised against. If the corpus really \
                 has changed, re-run with --expect-live-docs {} AFTER confirming why.",
                c.live_docs,
                self.expect_live_docs,
                c.live_docs
            );
        }
        if c.chunk_vectors != self.expect_chunk_vectors {
            anyhow::bail!(
                "REFUSING: chunk-vector count is {} but {} was expected.\n\
                 Something has written vectors since this run was authorised. Find out what before \
                 overriding with --expect-chunk-vectors {}.",
                c.chunk_vectors,
                self.expect_chunk_vectors,
                c.chunk_vectors
            );
        }
        if c.orphan_vectors != 0 {
            anyhow::bail!(
                "REFUSING: {} vectors point at documents that no longer exist. Fix that first — \
                 a guard that runs on a corrupt index certifies corruption.",
                c.orphan_vectors
            );
        }

        match self.backup {
            None => {}
            Some(path) => {
                if !path.exists() {
                    anyhow::bail!(
                        "REFUSING: backup {} does not exist. Make one first:\n  \
                         sqlite3 <db> \".backup '<path>'\"",
                        path.display()
                    );
                }
                // 🔴 Existing is not enough — OPEN it and COUNT. A 7 GB file that answers
                // `PRAGMA quick_check: ok` can still contain no tables: a `file:C:\path` URI with
                // BACKSLASHES silently opens a different, empty database. That exact mistake
                // reported a good backup as empty during this work.
                let bak = rusqlite::Connection::open_with_flags(
                    path,
                    rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
                )
                .with_context(|| format!("REFUSING: backup {} will not open", path.display()))?;
                let live: i64 = bak
                    .query_row(
                        "SELECT count(*) FROM documents WHERE valid_to IS NULL",
                        [],
                        |r| r.get(0),
                    )
                    .with_context(|| {
                        format!(
                            "REFUSING: backup {} opened but has no readable `documents` table",
                            path.display()
                        )
                    })?;
                println!("   backup live docs    {:>9}   {}", live, path.display());
                if live != self.expect_live_docs {
                    anyhow::bail!(
                        "REFUSING: the backup holds {} live documents, not {}. It is not a backup \
                         of this database state, so it cannot be relied on to undo this run.",
                        live,
                        self.expect_live_docs
                    );
                }
            }
        }
        println!("   ok: preconditions hold");
        Ok(c)
    }

    /// Everything that must be true AFTER. `extra` lets a caller add its own postconditions
    /// without the guard having to know about them.
    fn postflight(
        &self,
        store: &Store,
        before: store::GuardCounts,
        extra: &dyn Fn(store::GuardCounts) -> Result<()>,
    ) -> Result<store::GuardCounts> {
        let c = store.guard_counts()?;
        println!("\n{}", reply::paint(reply::BOLD, &format!("AFTER: {}", self.what)));
        println!("   live documents      {:>9}   (was {})", c.live_docs, before.live_docs);
        println!("   chunk vectors       {:>9}   (was {})", c.chunk_vectors, before.chunk_vectors);
        println!("   orphaned vectors    {:>9}", c.orphan_vectors);
        println!("   rows w/o a hash     {:>9}   (was {})", c.content_sha_null, before.content_sha_null);
        println!("   embedding queue     {:>9}   (was {})", c.queue, before.queue);

        if c.live_docs != before.live_docs {
            anyhow::bail!(
                "FAILED: live documents went {} -> {}. This operation must not change that.",
                before.live_docs,
                c.live_docs
            );
        }
        if c.chunk_vectors != before.chunk_vectors {
            anyhow::bail!(
                "FAILED: chunk vectors went {} -> {}. This operation must not change that.",
                before.chunk_vectors,
                c.chunk_vectors
            );
        }
        if c.orphan_vectors != 0 {
            anyhow::bail!("FAILED: {} orphaned vectors after the run", c.orphan_vectors);
        }
        extra(c)?;
        println!("   ok: postconditions hold");
        Ok(c)
    }
}

/// `recall stamp-vectors-current` — the guarded stamp backfill.
fn cmd_stamp_vectors_current(
    expect_live_docs: i64,
    expect_chunk_vectors: i64,
    backup: &Path,
    verify_sample: usize,
    dry_run: bool,
) -> Result<()> {
    let store = Store::open()?;
    let guard = GuardedRun {
        what: "stamp existing vectors as current",
        expect_live_docs,
        expect_chunk_vectors,
        backup: Some(backup),
    };
    let before = guard.preflight(&store)?;

    if before.content_sha_null > 0 {
        anyhow::bail!(
            "REFUSING: {} rows have no content_sha256. Run `recall backfill-hashes` first — \
             stamping a row whose content hash is NULL writes NULL and leaves it queued forever.",
            before.content_sha_null
        );
    }

    println!("\n{}", reply::paint(reply::BOLD, "PLAN"));
    println!("   stamp {} live document(s) whose vectors exist but are unrecorded", before.stampable);
    println!("   this writes ONE column. It does not embed, delete, or move anything.");
    // 🔴 Do NOT subtract these two. They are DIFFERENT predicates: `queue` is tier-gated (it
    // excludes tiers the query path will never serve), `stampable` is not (a document is worth
    // recording regardless of whether search returns it). Subtracting them printed
    // `queue should fall 52131 -> -7377` on the live corpus - a negative queue, which is exactly
    // the class of nonsense counter this command exists to stop producing.
    //
    // Report both, predict neither. The AFTER block measures the real number seconds later.
    println!(
        "   {} document(s) are stampable; the tier-gated queue is currently {}",
        before.stampable, before.queue
    );

    if before.stampable == 0 {
        println!("\n   nothing to stamp — every document with vectors already has its record.");
        // Not a skip: the postconditions still have to hold, and they are what the caller cares
        // about. Falling through re-verifies rather than exiting on an assumption.
    }

    if verify_sample > 0 {
        println!(
            "\n--verify-sample {} was requested but is not implemented in this command.",
            verify_sample
        );
        anyhow::bail!(
            "REFUSING: --verify-sample is not wired yet. Verify by hand with \
             `recall embed --path-list <sample> --force` and compare vectors, or omit the flag \
             and accept the sample already measured (290 documents, 8,306 chunk rows, 0 differences)."
        );
    }

    if dry_run {
        println!("\n--dry-run: every precondition passed and NOTHING was written.");
        anyhow::bail!("dry run: exiting non-zero so this can never be mistaken for a completed run");
    }

    println!("\napplying...");
    let started = std::time::Instant::now();
    let stamped = store.backfill_embedded_stamp()?;
    println!("   stamped {} row(s) in {:.1}s", stamped, started.elapsed().as_secs_f64());

    let after = guard.postflight(&store, before, &|c: store::GuardCounts| {
        if c.content_sha_null != 0 {
            anyhow::bail!(
                "FAILED: {} rows have no content_sha256 after the run. Every document must have \
                 one, or the next run stamps NULL over it.",
                c.content_sha_null
            );
        }
        if c.stampable != 0 {
            anyhow::bail!(
                "FAILED: {} document(s) still have vectors with no record. The run was PARTIAL — \
                 do not treat this as done.",
                c.stampable
            );
        }
        Ok(())
    })?;

    println!(
        "\nDONE: {} document(s) stamped, queue {} -> {}",
        stamped, before.queue, after.queue
    );
    if after.queue > 0 {
        println!(
            "   {} document(s) remain queued: those genuinely need embedding (no vectors, or a \
             body that changed since they were embedded).",
            after.queue
        );
    }
    Ok(())
}

/// `recall backfill-hashes` — fill `content_sha256` wherever it is NULL.
///
/// Loops the store's batched backfill until it reports 0 rows. Progress goes to stderr so the
/// final count on stdout stays machine-readable.
fn cmd_qwen3_embed(limit: usize, batch: usize, dry_run: bool) -> Result<()> {
    use std::time::Instant;

    anyhow::ensure!(batch > 0, "--batch must be at least 1");
    let mut store = store::Store::open()?;

    let (before_docs, before_rows, queued) = store.qwen3_progress()?;
    println!("BEFORE");
    println!("   documents with Qwen3 vectors : {before_docs}");
    println!("   Qwen3 chunk rows             : {before_rows}");
    println!("   documents queued             : {queued}");
    let target = if limit == 0 { queued as usize } else { limit };
    println!("   this run will embed          : {target}");
    println!("   chunk size                   : {} chars", qwen3::chunk_chars());
    if dry_run {
        println!();
        println!("--dry-run: nothing written.");
        return Ok(());
    }
    if queued == 0 {
        println!();
        println!("Queue is empty. Nothing to do.");
        return Ok(());
    }

    let mut e = qwen3::Qwen3Embedder::new()?;
    println!("backend={}", e.backend());

    let t0 = Instant::now();
    let mut docs_done = 0usize;
    let mut rows_written = 0usize;
    let target_chars = qwen3::chunk_chars();

    // 🔴 Batch ACROSS documents, never within one.
    //
    // The first version did `for group in doc_chunks.chunks(batch)`, which looks right and is
    // catastrophic here: the median vault document is 1,604 chars and therefore ONE chunk, so
    // almost every forward pass was a batch of 1 and the GPU sat idle between them. Measured on
    // the live queue: 1.44 chunks/sec, a 25-hour projection, against 5.5 chunks/sec in the
    // benchmark that fed it full batches. Same model, same card, same text - only the packing
    // differed.
    //
    // So: flatten a slice of documents into one chunk list, embed in FULL batches, then scatter
    // the vectors back to their documents. A document is still written in a single transaction.
    // 🔴 A slice is bounded by CHARACTERS, not by a document count.
    //
    // Document length in this vault spans four orders of magnitude - median 1,604 chars, largest
    // 491,003. A fixed 64-document slice was therefore anywhere from 64 to 2,674 chunks: measured
    // on the real queue, the first 64 documents held 9.6 MB of text, which pinned VRAM at 7,811
    // MiB of 8,192 and wrote nothing for over four minutes. A count-based slice cannot bound
    // either memory or commit latency when the thing being counted varies that much.
    //
    // 400k chars is ~110 chunks: several full batches, a commit every ~20s, and VRAM well clear
    // of the ceiling.
    const SLICE_CHARS: usize = 400_000;
    // Still cap the document count so a slice of tiny notes does not build a huge Vec.
    const SLICE_DOCS: usize = 512;
    let mut pending: Vec<(i64, String, Vec<qwen3::Chunk>)> = Vec::new();
    let mut flat: Vec<String> = Vec::new();

    'outer: loop {
        let slice = store.docs_needing_qwen3(SLICE_DOCS)?;
        if slice.is_empty() {
            break;
        }

        pending.clear();
        flat.clear();
        let mut slice_chars = 0usize;
        for (doc_id, title, body) in slice {
            // Take at least one document even when it alone exceeds the budget - a 491k-char
            // note must still be embeddable, just on its own.
            if slice_chars > 0 && slice_chars + body.len() > SLICE_CHARS {
                break;
            }
            slice_chars += body.len();
            let chunks = qwen3::split_document(&title, &body, target_chars);
            let sha = {
                use sha2::{Digest, Sha256};
                let mut h = Sha256::new();
                h.update(body.as_bytes());
                format!("{:x}", h.finalize())
            };
            for c in &chunks {
                flat.push(c.text.clone());
            }
            pending.push((doc_id, sha, chunks));
        }

        let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(flat.len());
        for group in flat.chunks(batch) {
            vecs.extend(e.embed_documents(group)?);
        }
        anyhow::ensure!(
            vecs.len() == flat.len(),
            "got {} vectors for {} chunks",
            vecs.len(),
            flat.len()
        );

        let mut cursor = 0usize;
        for (doc_id, sha, chunks) in pending.drain(..) {
            let n = chunks.len();
            let rows: Vec<(String, i64, i64, Vec<f32>)> = chunks
                .into_iter()
                .zip(vecs[cursor..cursor + n].iter().cloned())
                .map(|(c, v)| (c.heading, c.start, c.end, v))
                .collect();
            cursor += n;
            rows_written += store.replace_qwen3_chunks(doc_id, qwen3::DIM, &rows, &sha)?;
            docs_done += 1;

            if docs_done % 25 == 0 {
                let secs = t0.elapsed().as_secs_f64();
                let rate = docs_done as f64 / secs;
                let left = (target.saturating_sub(docs_done)) as f64 / rate / 3600.0;
                println!(
                    "   {docs_done}/{target} docs, {rows_written} rows, {rate:.2} docs/sec, {:.1} chunks/sec, ~{left:.2}h left",
                    rows_written as f64 / secs
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
            if limit > 0 && docs_done >= limit {
                break 'outer;
            }
        }
    }

    let (after_docs, after_rows, after_queued) = store.qwen3_progress()?;
    println!();
    println!("AFTER  ({:.1}s)", t0.elapsed().as_secs_f64());
    println!("   documents with Qwen3 vectors : {before_docs} -> {after_docs}");
    println!("   Qwen3 chunk rows             : {before_rows} -> {after_rows}");
    println!("   documents queued             : {queued} -> {after_queued}");
    anyhow::ensure!(
        docs_done > 0,
        "embedded nothing while the queue was non-empty - refusing to exit 0 on a no-op"
    );
    Ok(())
}

/// Split on character count at a paragraph boundary where one is near, else hard-cut.
/// Deliberately crude: this measures COST, not retrieval quality, and a smarter splitter would
/// only change the token distribution by a few percent.
fn rough_split(body: &str, target: usize) -> Vec<String> {
    let chars: Vec<char> = body.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let hard = (i + target).min(chars.len());
        let mut end = hard;
        if hard < chars.len() {
            // look back up to 15% of the window for a blank line
            let floor = i + target * 85 / 100;
            let mut j = hard;
            while j > floor && j > i + 1 {
                if chars[j - 1] == '\n' && j >= 2 && chars[j - 2] == '\n' {
                    end = j;
                    break;
                }
                j -= 1;
            }
        }
        if end <= i {
            end = hard.max(i + 1);
        }
        out.push(chars[i..end].iter().collect());
        i = end;
    }
    out
}

fn cmd_qwen3_chunk_sweep(
    docs: usize,
    target_chars: &[usize],
    batch: usize,
    project_docs: u64,
) -> Result<()> {
    use std::time::Instant;

    anyhow::ensure!(batch > 0, "--batch must be at least 1");
    anyhow::ensure!(!target_chars.is_empty(), "--target-chars must not be empty");

    let store = store::Store::open()?;
    let bodies = store.sample_doc_bodies(docs)?;
    anyhow::ensure!(!bodies.is_empty(), "no document bodies readable");
    let total_chars: usize = bodies.iter().map(|(_, b)| b.chars().count()).sum();
    println!(
        "corpus: {} real documents, {} chars total, mean {:.0} chars/doc",
        bodies.len(),
        total_chars,
        total_chars as f64 / bodies.len() as f64
    );

    let mut e = qwen3::Qwen3Embedder::new()?;
    println!("backend={}", e.backend());

    // Warm the kernels once, outside every arm, so arm 1 is not taxed for all of them.
    e.embed_documents(&["warmup".repeat(50)])?;

    println!();
    println!(
        "{:>8}  {:>8}  {:>10}  {:>11}  {:>12}  {:>14}",
        "chars", "chunks", "chunks/sec", "docs/sec", "hours(docs)", "chunks(corpus)"
    );
    for &target in target_chars {
        let mut texts: Vec<String> = Vec::new();
        for (title, body) in &bodies {
            for piece in rough_split(body, target) {
                texts.push(if title.is_empty() {
                    piece
                } else {
                    format!("{title}\n{piece}")
                });
            }
        }
        let n = texts.len();
        let t = Instant::now();
        for group in texts.chunks(batch) {
            e.embed_documents(group)?;
        }
        let secs = t.elapsed().as_secs_f64();
        let chunks_sec = n as f64 / secs;
        let docs_sec = bodies.len() as f64 / secs;
        let hours = project_docs as f64 / docs_sec / 3600.0;
        let corpus_chunks = (n as f64 / bodies.len() as f64) * project_docs as f64;
        println!(
            "{target:>8}  {n:>8}  {chunks_sec:>10.1}  {docs_sec:>11.2}  {hours:>12.2}  {corpus_chunks:>14.0}"
        );
    }

    println!();
    println!("Read the docs/sec and hours columns. chunks/sec RISES with smaller chunks while the");
    println!("run gets SLOWER - it is the metric that lies here, because the text is constant.");
    Ok(())
}

fn cmd_qwen3_bench(chunks: usize, batch: usize, project_over: u64) -> Result<()> {
    use std::time::Instant;

    anyhow::ensure!(batch > 0, "--batch must be at least 1");
    let store = store::Store::open()?;
    let texts = store.sample_chunk_texts(chunks)?;
    anyhow::ensure!(
        !texts.is_empty(),
        "no chunk rows readable - nothing to benchmark"
    );
    let total_chars: usize = texts.iter().map(|t| t.len()).sum();
    println!(
        "corpus: {} real chunks, {} chars, mean {:.0} chars/chunk",
        texts.len(),
        total_chars,
        total_chars as f64 / texts.len() as f64
    );

    let t0 = Instant::now();
    let mut e = qwen3::Qwen3Embedder::new()?;
    let load = t0.elapsed();
    println!("load: {load:?}  backend={}", e.backend());

    // One warm batch first. The first forward on DirectML compiles kernels, and folding that
    // into the throughput number understates the GPU by a large factor on a short run.
    let warm = texts[..batch.min(texts.len())].to_vec();
    let tw = Instant::now();
    e.embed_documents(&warm)?;
    println!("warmup batch ({} texts): {:?}", warm.len(), tw.elapsed());

    let t1 = Instant::now();
    let mut done = 0usize;
    let mut dim = 0usize;
    for group in texts.chunks(batch) {
        let v = e.embed_documents(group)?;
        if dim == 0 {
            dim = v.first().map(|x| x.len()).unwrap_or(0);
        }
        done += v.len();
    }
    let elapsed = t1.elapsed();
    let secs = elapsed.as_secs_f64();
    let rate = done as f64 / secs;

    println!();
    println!("backend              : {}", e.backend());
    println!("dim                  : {dim}");
    println!("batch size           : {batch}");
    println!("chunks embedded      : {done}");
    println!("elapsed              : {secs:.2}s");
    println!("throughput           : {rate:.1} chunks/sec");
    let hours = project_over as f64 / rate / 3600.0;
    println!("projected over {project_over} chunk rows : {hours:.2} hours");
    Ok(())
}

fn cmd_qwen3_check(min_self: f32, max_unrelated: f32) -> Result<()> {
    use std::time::Instant;

    if !qwen3::available() {
        anyhow::bail!(
            "Qwen3 model not present at {}",
            qwen3::model_dir()?.display()
        );
    }

    let t0 = Instant::now();
    let mut e = qwen3::Qwen3Embedder::new()?;
    println!("load: {:?}  backend={}", t0.elapsed(), e.backend());

    // One batch, so a pooling bug that reads the wrong ROW shows up too - if pooling were
    // per-batch-position rather than per-row, rows 0 and 1 (identical text) would still agree
    // but rows 2 and 3 would collide with them.
    let texts: Vec<String> = vec![
        "The backup job uploads encrypted snapshots to Backblaze B2 every night.".into(),
        "The backup job uploads encrypted snapshots to Backblaze B2 every night.".into(),
        "Qwen3 uses last-token pooling over a causal decoder.".into(),
        "My cat refuses to eat anything but tuna on a Tuesday.".into(),
    ];
    let t1 = Instant::now();
    let v = e.embed_documents(&texts)?;
    println!("embed: {} texts in {:?}", v.len(), t1.elapsed());

    anyhow::ensure!(v.len() == 4, "expected 4 vectors, got {}", v.len());
    for (i, x) in v.iter().enumerate() {
        anyhow::ensure!(
            x.len() == qwen3::DIM,
            "vector {i} has width {}, expected {}",
            x.len(),
            qwen3::DIM
        );
    }

    let same = qwen3::cosine(&v[0], &v[1]);
    let un_a = qwen3::cosine(&v[0], &v[2]);
    let un_b = qwen3::cosine(&v[0], &v[3]);
    let un_c = qwen3::cosine(&v[2], &v[3]);

    println!();
    println!("dim                        : {}", qwen3::DIM);
    println!("identical text             : {same:.4}   (must be >= {min_self})");
    println!("backup vs pooling          : {un_a:.4}   (must be <  {max_unrelated})");
    println!("backup vs cat              : {un_b:.4}   (must be <  {max_unrelated})");
    println!("pooling vs cat             : {un_c:.4}   (must be <  {max_unrelated})");

    // The query prefix must MOVE the vector. If it does not, the prefix is being dropped and
    // asymmetric retrieval is silently symmetric.
    let q = e.embed_query("where do the nightly backups go")?;
    let d = qwen3::cosine(&q, &v[0]);
    let q_raw = e.embed_documents(&["where do the nightly backups go".to_string()])?;
    let drift = qwen3::cosine(&q, &q_raw[0]);
    println!("query->doc (related)       : {d:.4}");
    println!("prefixed vs unprefixed query: {drift:.4}   (must be < 0.999, else the prefix is inert)");

    let mut bad: Vec<String> = Vec::new();
    if !(same >= min_self) {
        bad.push(format!("identical text scored {same:.4}, below {min_self}"));
    }
    for (name, c) in [
        ("backup vs pooling", un_a),
        ("backup vs cat", un_b),
        ("pooling vs cat", un_c),
    ] {
        if !(c < max_unrelated) {
            bad.push(format!("{name} scored {c:.4}, not below {max_unrelated}"));
        }
    }
    if !(drift < 0.999) {
        bad.push(format!(
            "the query prefix changed the vector by nothing ({drift:.4}) - it is not being applied"
        ));
    }

    println!();
    if bad.is_empty() {
        println!("PASS - pooling, EOS and the instruction prefix are all wired correctly.");
        Ok(())
    } else {
        for b in &bad {
            println!("FAIL - {b}");
        }
        anyhow::bail!("{} qwen3 wiring check(s) failed", bad.len())
    }
}

fn cmd_backfill_hashes(batch: usize) -> Result<()> {
    if batch == 0 {
        anyhow::bail!("--batch must be at least 1");
    }
    let store = Store::open()?;
    let started = std::time::Instant::now();
    let mut total = 0usize;
    loop {
        let n = store.backfill_content_hashes(batch)?;
        if n == 0 {
            break;
        }
        total += n;
        eprintln!("backfilled {total} rows ({:.1}s)", started.elapsed().as_secs_f64());
    }
    println!(
        "done: {total} rows hashed in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmd_query(
    text: String,
    top: Option<String>,
    channels: Option<String>,
    via_daemon: bool,
    no_daemon: bool,
    json_out: bool,
    allow_fallback: bool,
) -> Result<()> {
    let top: usize = top.as_deref().unwrap_or("5").parse().unwrap_or(5);
    // Mirror the daemon's server-side guard on the in-process path: `query::fused`
    // computes `top * 4` (multiply-overflow risk for a pathological `top`) and makes
    // every channel over-allocate, while `top == 0` silently returns nothing.
    let top = top.clamp(1, daemon::MAX_QUERY_K);
    // Default channels. `voyage` (Voyage-4-large chunk vectors, 1024-dim) is the
    // default because it recalls far more than fts5 alone on the maintainer's
    // private query set.
    //
    // DELIBERATELY NOT "voyage,fts5". Plain RRF weights every channel equally, so
    // fusing a 0.71 channel with a 0.32 one lets the weak channel outvote the
    // strong one. Measured 2026-07-27, top-1, same warm daemon:
    //   query                                     voyage         voyage,fts5
    //   "what did I decide about secrets storage"  crypt       -> E2E Test: env-migrate
    //   "how do I stop the model making things up" Prevent LLM Hallucinations -> Best practices
    //   "cheapest way to host a background worker" Parallel Agents + VPS Infra -> Executive Verdict
    //   "how much did the embeddings cost me"      03 - Cost Models -> Dictation 2026-04-18
    // 4 of 5 regressed. Re-enable fusion only once RRF is weighted by measured
    // per-channel quality, or fts5 is used as a rescue tier when voyage returns
    // nothing. An earlier rerank experiment showed the same thing: reranking a
    // weak pool also lost (MRR 0.5837 -> 0.5567).
    //
    // Fails open: with no API key, no network, or a build without the `voyage`
    // feature the channel contributes nothing, so fts5 must stay in the fallback.
    // Override with --channels or RECALL_CHANNELS.
    // Default is `gemini`, NOT `voyage`. Voyage is a paid API on the owner's own card;
    // Gemini runs on Google credit. The two are interchangeable at this layer - both are
    // single dense channels with an fts5 rescue tier - so defaulting to the one that does
    // not bill him is free. `voyage` remains selectable via --channels for A/B work.
    #[cfg(feature = "gemini")]
    const DEFAULT_CHANNELS: &str = "gemini";
    #[cfg(all(feature = "voyage", not(feature = "gemini")))]
    const DEFAULT_CHANNELS: &str = "voyage";
    // Plain build (no paid-API features): the free, offline pair. bge is always compiled in; if
    // ONNX Runtime / the model are missing the query exits 1 loudly rather than serving keywords.
    #[cfg(not(any(feature = "voyage", feature = "gemini")))]
    const DEFAULT_CHANNELS: &str = "fts5,bge";

    // Output-time scrub (secrets / pii / custom layers); see src/scrub.rs. Settings read once per query.
    let sc = scrub::Scrubber::new(&scrub::Config::load());
    let vault = store::vault_root();
    let channels: Vec<String> = channels
        .or_else(|| std::env::var("RECALL_CHANNELS").ok())
        .unwrap_or_else(|| DEFAULT_CHANNELS.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    // Before the daemon fast path, so a bad name fails here with exit 1 whichever path runs.
    query::validate_channels(&channels)?;
    // Said here, in the caller's terminal: a daemon would print it to its own stderr only.
    if channels.iter().any(|c| matches!(c.to_lowercase().as_str(), "rg" | "ripgrep")) && !store::rg_available() {
        store::warn_rg_missing();
    }

    // ── Daemon fast-path ──────────────────────────────────────────────────────
    // If a daemon.json exists (or --via-daemon is forced), try the warm path.
    // On any error we fall through to the in-process path below — the user sees
    // no difference; they just lose the latency win for that call.
    // A scoped reader's identity travels in the request (proto 19); an older daemon's reply is
    // refused by version and the query falls through to the in-process path, which is scoped.
    if !no_daemon && !allow_fallback && (via_daemon || daemon::DaemonInfo::read().is_some()) {
        if let Some(info) = daemon::DaemonInfo::read() {
            match daemon::query_daemon(&info.addr, &text, top, &channels) {
                Some(daemon_hits) => {
                    // Print using the same format as the in-process path.
                    if daemon_hits.is_empty() {
                        if json_out {
                            println!("[]");
                        } else {
                            reply::section("WHAT I DID", &format!("recall query: no matches for: {}", human_text(&sc, &text)));
                        }
                        return Ok(());
                    }
                    // Labels come from the hits themselves, so a daemon answer carries them too.
                    let truncated = query::truncation_note(daemon_hits.iter().any(|h| h.labels_truncated));
                    if let Some(n) = truncated {
                        eprintln!("{n}");
                    }
                    let (daemon_hits, fresh) = with_freshness(daemon_hits, |h| h.id, |h| h.snippet.as_str(), &text);
                    if json_out {
                        let out: Vec<QueryJsonHit> = daemon_hits
                            .iter()
                            .zip(fresh.iter())
                            .enumerate()
                            .map(|(rank, (h, fr))| QueryJsonHit {
                                doc_id: h.id,
                                rank,
                                score: h.score,
                                vault_path: h.path.clone(),
                                title: sc.scrub(&h.title).into_owned(),
                                snippet: sc.scrub(&h.snippet).into_owned(),
                                // A pre-2026-10-29 daemon sends no source; never claim "local".
                                source: if h.source.is_empty() {
                                    "unknown".to_string()
                                } else {
                                    h.source.clone()
                                },
                                ranks: h.ranks.clone(),
                                scores: h.scores.clone(),
                                rrf: h.rrf,
                                lacks: h.lacks.clone(),
                                merged: h.merged.clone(),
                                labels_truncated: h.labels_truncated,
                                redactions: if rank < scrub::MAX_POINTER_HITS { scrub::pointers(&sc, &vault, &h.path) } else { Vec::new() },
                                fresh: fr.clone(),
                                closest_match: h.closest_match,
                                absent: h.absent.clone(),
                            })
                            .collect();
                        println!("{}", serde_json::to_string_pretty(&out)?);
                    } else {
                        let list: Vec<HitText> = daemon_hits
                            .iter()
                            .map(|h| HitText {
                                title: &h.title,
                                path: &h.path,
                                snippet: &h.snippet,
                                lacks: &h.lacks,
                                closest: h.closest_match,
                                absent: &h.absent,
                                provenance: provenance_line(&h.source, &h.ranks, &h.scores, h.rrf),
                            })
                            .collect();
                        print_hit_list(&text, &sc, &vault, &fresh, truncated, &list);
                    }
                    return Ok(());
                }
                None => {
                    if via_daemon {
                        // Force flag set — error out rather than silently fall through.
                        anyhow::bail!(
                            "daemon at {} is not responding (use `recall daemon status` to check)",
                            info.addr
                        );
                    }
                    // Auto-detect mode: daemon file exists but connect failed (stale).
                    // Fall through to in-process path silently.
                }
            }
        } else if via_daemon {
            anyhow::bail!(
                "--via-daemon requested but no daemon is running (start with `recall daemon up`)"
            );
        }
    }

    // ── In-process path (unchanged from original) ─────────────────────────────
    let store = Store::open()?;
    // Load the BGE channel only when the caller actually asked for it — keyword-
    // only queries shouldn't pay the model-load cost.
    let bge = if channels.iter().any(|c| {
        let c = c.to_lowercase();
        c == "bge" || c == "vector" || c == "lancedb"
    }) {
        match bge::BgeSearch::load(&store) {
            Err(e) if allow_fallback => {
                eprintln!("RESCUE: bge unavailable ({e:#}); --allow-fallback given, serving fts5");
                None
            }
            r => r?,
        }
    } else {
        None
    };
    // Same rule as BGE: only pay the model-load cost when the caller asked for this channel.
    let qwen = if channels
        .iter()
        .any(|c| matches!(c.to_lowercase().as_str(), "qwen3" | "qwen"))
    {
        qwen3::Qwen3Search::load(&store)?
    } else {
        None
    };
    let hits = query::fused_sourced(
        &store,
        bge.as_ref(),
        qwen.as_ref(),
        &text,
        top,
        &channels,
    )?;

    if hits.is_empty() {
        if json_out {
            println!("[]");
        } else {
            reply::section("WHAT I DID", &format!("recall query: no matches for: {}", human_text(&sc, &text)));
        }
        return Ok(());
    }

    let (hits, fresh) = with_freshness(hits, |(h, _)| h.id, |(h, _)| h.snippet.as_str(), &text);
    let truncated = query::truncation_note(hits.iter().any(|(_, p)| p.labels_truncated));
    if let Some(n) = truncated {
        eprintln!("{n}");
    }
    if json_out {
        let out: Vec<QueryJsonHit> = hits
            .iter()
            .zip(fresh.iter())
            .enumerate()
            .map(|(rank, ((h, p), fr))| QueryJsonHit {
                doc_id: h.id,
                rank,
                score: h.score,
                vault_path: h.path.clone(),
                title: sc.scrub(&h.title).into_owned(),
                snippet: sc.scrub(&h.snippet).into_owned(),
                source: p.source.clone(),
                ranks: p.ranks.clone(),
                scores: p.scores.clone(),
                rrf: p.rrf,
                lacks: p.lacks.clone(),
                merged: p.merged.clone(),
                labels_truncated: p.labels_truncated,
                redactions: if rank < scrub::MAX_POINTER_HITS { scrub::pointers(&sc, &vault, &h.path) } else { Vec::new() },
                fresh: fr.clone(),
                closest_match: p.closest_match,
                absent: p.absent.clone(),
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        let list: Vec<HitText> = hits
            .iter()
            .map(|(h, p)| HitText {
                title: &h.title,
                path: &h.path,
                snippet: &h.snippet,
                lacks: &p.lacks,
                closest: p.closest_match,
                absent: &p.absent,
                provenance: provenance_line(&p.source, &p.ranks, &p.scores, p.rrf),
            })
            .collect();
        print_hit_list(&text, &sc, &vault, &fresh, truncated, &list);
    }
    Ok(())
}

fn cmd_save(text: String, source: Option<String>, share: Option<String>) -> Result<()> {
    let source = source.unwrap_or_else(|| "cli".into());
    // Resolved BEFORE anything is written, so a refused --share changes nothing.
    let (tag, agent) = scope::save_target(&scope::current(), share.as_deref())?;
    let store = Store::open()?;
    let id = store.insert_scoped("", "", &scrub::scrub_stored(text), &source, &tag, agent.as_deref())?;
    reply::section(
        "WHAT I DID",
        &format!("Saved observation #{} (source={})", id, source),
    );
    Ok(())
}

fn cmd_daemon(action: String) -> Result<()> {
    match action.as_str() {
        "up" => {
            // The daemon serves every reader from one process; it must hold the operator view
            // and scope per request, never inherit one agent's identity for all callers.
            if !scope::current().is_operator() {
                anyhow::bail!("recall daemon must run as the operator: unset RECALL_AGENT/RECALL_PROJECT/RECALL_TEAM");
            }
            // Start the warm daemon. This blocks in the accept loop until a
            // `{"cmd":"shutdown"}` is received. Run with `recall daemon up &`
            // or via `mem.py ensure_daemon` to detach.
            daemon::run_daemon()?;
        }
        "down" => match daemon::DaemonInfo::read() {
            None => {
                reply::section("WHAT I DID", "No daemon running (daemon.json absent).");
            }
            Some(info) => {
                daemon::shutdown_daemon(&info.addr)?;
                reply::section(
                    "WHAT I DID",
                    &format!("Daemon at {} (pid={}) shut down.", info.addr, info.pid),
                );
            }
        },
        "status" => {
            match daemon::DaemonInfo::read() {
                Some(info) => {
                    match daemon::ping_daemon(&info.addr) {
                        Some(details) => {
                            reply::section(
                                "WHAT I DID",
                                &format!("Daemon is RUNNING — {}", details),
                            );
                        }
                        None => {
                            // State file exists but daemon not responding — stale.
                            let store = Store::open()?;
                            reply::card(&[(
                                "WHAT I DID",
                                &[
                                    "daemon.json exists but daemon is NOT responding (stale).",
                                    &format!("Store online — {} valid docs", store.count_valid()?),
                                ][..],
                            )]);
                        }
                    }
                }
                None => {
                    // No daemon file — fall back to store-count as before.
                    let store = Store::open()?;
                    reply::section(
                        "WHAT I DID",
                        &format!(
                            "No daemon running. Store online — {} valid docs",
                            store.count_valid()?
                        ),
                    );
                }
            }
        }
        other => {
            reply::blocking(
                "daemon",
                &format!("unknown action '{}': use up|down|status", other),
            );
        }
    }
    Ok(())
}

fn cmd_bench(
    overnight: Option<String>,
    iterations: Option<String>,
    smoke: Option<String>,
    json_out: bool,
) -> Result<()> {
    if !bench::fixtures_exist() {
        reply::blocking(
            "bench",
            "bench fixtures not found: set `bench_fixtures` in ~/.recall/private-rules.json",
        );
        return Ok(());
    }
    let overnight = matches!(overnight.as_deref(), Some("true") | Some("1"));
    let iterations: usize = iterations
        .as_deref()
        .unwrap_or("50")
        .parse()
        .unwrap_or(50)
        .clamp(1, 10_000);
    // 0 (or unparsable) = score all positives; N>0 = first N only (CI smoke).
    let smoke: usize = smoke.as_deref().unwrap_or("0").parse().unwrap_or(0);
    let limit = (smoke > 0).then_some(smoke);
    if overnight {
        if json_out {
            reply::blocking(
                "bench",
                "--json-out is supported for single benchmark runs only; run without --overnight",
            );
            return Ok(());
        }
        bench::overnight(iterations, 5, limit)?;
    } else if json_out {
        bench::run_and_write(5, limit)?;
    } else {
        bench::run(5, limit)?;
    }
    Ok(())
}

fn read_path_list(path_list: &PathBuf) -> Result<Vec<PathBuf>> {
    let text = std::fs::read_to_string(path_list)?;
    Ok(text
        .lines()
        .map(|line| line.trim().trim_start_matches('\u{feff}').trim())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(PathBuf::from)
        .collect())
}

/// Parse `--prune-missing`. Unknown values are an ERROR, never a silent `off`: a typo
/// (`--prune-missing=apply-please`) must not quietly turn the mutation into a no-op, and it
/// must not quietly turn a no-op into a mutation either.
fn parse_prune_mode(raw: &str) -> Result<Option<store::PruneMode>> {
    match raw.trim().to_lowercase().as_str() {
        "off" | "false" | "no" => Ok(None),
        "dry" | "dry-run" => Ok(Some(store::PruneMode::Dry)),
        "apply" => Ok(Some(store::PruneMode::Apply)),
        other => anyhow::bail!(
            "--prune-missing={other} is not a valid mode (expected: off | dry | apply)"
        ),
    }
}

fn cmd_index(
    reset: Option<String>,
    path_list: Option<PathBuf>,
    dry_run: bool,
    prune_missing: String,
) -> Result<()> {
    let reset = matches!(reset.as_deref(), Some("true") | Some("1"));
    let prune_mode = parse_prune_mode(&prune_missing)?;
    // Preflight or refuse, before the store is opened: a missing vault root makes every listed
    // path look absent and every enumeration empty, and both used to end in exit 0.
    let root = store::vault_root_checked()?;
    let store = Store::open()?;

    // `--prune-missing=dry` is a pure read: it must be usable to inspect the list WITHOUT
    // running an ingest pass first, so nothing is written before the owner has seen what would go.
    if prune_mode == Some(store::PruneMode::Dry) {
        let report = store.prune_missing(
            &root,
            store::PruneMode::Dry,
            &prune_list_dir()?,
        )?;
        print_prune_report(&report, store::PruneMode::Dry);
        return Ok(());
    }

    // Preflight before anything is written: an apply prune that would refuse for a missing rg must
    // not first ingest (or reset) and then report "nothing was changed".
    index_prune_preflight(prune_mode, || store::rg_available())?;

    // One writer at a time: a background refresh may be mid-run. Waits (a refresh is short),
    // then refuses rather than run a second writer. Held until this function returns.
    let _index_lock = if dry_run {
        None
    } else {
        Some(refresh::IndexLock::acquire_wait(std::time::Duration::from_secs(20 * 60))?)
    };

    if reset {
        // DESTRUCTIVE — gate behind explicit env confirmation to honour the
        // "never --reset without confirmation" rule.
        if std::env::var("RECALL_CONFIRM_RESET").as_deref() != Ok("yes") {
            reply::card(&[
                (
                    "BLOCKING",
                    &["--reset is destructive (wipes the index)"][..],
                ),
                (
                    "YOU NEED TO DO",
                    &["Re-run with: RECALL_CONFIRM_RESET=yes recall index --reset true"][..],
                ),
            ]);
            return Ok(());
        }
        store.reset()?;
    }
    let (n, counts) = if let Some(path_list) = path_list.as_ref() {
        let paths = read_path_list(path_list)?;
        (store.ingest_paths(&root, &paths, dry_run)?, store::IngestCounts::default())
    } else if dry_run {
        reply::card(&[(
            "BLOCKING",
            &["--dry-run requires --path-list so broad indexing cannot be simulated as a safe batch"][..],
        )]);
        return Ok(());
    } else {
        store.ingest_dir_counted(&root)?
    };
    let verb = if dry_run { "Would index" } else { "Indexed" };
    let scope = if path_list.is_some() {
        "from path list".to_string()
    } else {
        format!("from {}", root.display())
    };
    let l1 = format!("{} {} docs {}", verb, n, scope);
    if dry_run {
        reply::card(&[("WHAT I DID", &[l1.as_str(), "Dry-run only: recall.db was not mutated"][..])]);
    } else {
        // Everything the pass did, on one stream and in one order. A zero row is not printed.
        reply::summary(
            &l1,
            &[
                ("new", counts.new as i64, false),
                ("updated", counts.updated as i64, false),
                ("moved or renamed", counts.moved as i64, false),
                ("restored", counts.restored as i64, false),
                ("unchanged", counts.unchanged as i64, false),
                ("docs in index", store.count_valid()?, true),
            ],
        );
    }

    if prune_mode == Some(store::PruneMode::Apply) {
        let report = store.prune_missing(&root, store::PruneMode::Apply, &prune_list_dir()?)?;
        print_prune_report(&report, store::PruneMode::Apply);
    }
    // feat/abstain-calibrated: calibrate the abstain judge on this corpus now (phantom probes,
    // stored in `abstain_calibration`), so the first query that needs it does not pay for it.
    if !dry_run {
        let t = std::time::Instant::now();
        match query::phantom_null(&store) {
            Ok(v) if v.is_empty() => {}
            Ok(v) => eprintln!(
                "recall index: calibrated the no-answer check on {} sample queries in {:.1} s",
                v.len(),
                t.elapsed().as_secs_f64()
            ),
            Err(e) => eprintln!("recall index: no-answer calibration skipped: {e}"),
        }
    }
    Ok(())
}

/// Where prune lists land: beside `recall.db` in `~/.recall`.
fn prune_list_dir() -> Result<PathBuf> {
    let db = store::default_db_path()?;
    Ok(db
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(".")))
}

fn print_prune_report(report: &store::PruneReport, mode: store::PruneMode) {
    let enumerated = format!(
        "Enumerated on disk: {}{}",
        report.enumerated,
        if report.used_fallback {
            "  (FALLBACK walk — rg did not run)"
        } else {
            ""
        }
    );
    let baseline = match report.baseline {
        Some(b) => format!(
            "Baseline: {b} (max of recent {} enumerations)",
            if report.used_fallback {
                "fallback-walk"
            } else {
                "rg"
            }
        ),
        None => "Baseline: NONE recorded — the shrink gate had nothing to compare against; \
                 the filesystem check was the only guard"
            .to_string(),
    };
    let live = format!("Live vault docs in index: {}", report.live_vault_docs);
    let candidates = format!(
        "Not enumerated: {} (candidates — excluded/gitignored/oversized files are in here and are NOT pruned)",
        report.not_enumerated
    );
    let gone = format!(
        "FILE IS GONE (filesystem-confirmed): {}",
        report.missing.len()
    );
    let action = match mode {
        store::PruneMode::Dry => {
            "Dry run — recall.db was NOT mutated. Review the list, then re-run with --prune-missing=apply."
                .to_string()
        }
        store::PruneMode::Apply => format!("Tombstoned (valid_to set): {}", report.tombstoned),
    };
    let list = format!("Full list: {}", report.list_path.display());

    reply::card(&[(
        "WHAT I DID",
        &[
            enumerated.as_str(),
            baseline.as_str(),
            live.as_str(),
            candidates.as_str(),
            gone.as_str(),
            action.as_str(),
            list.as_str(),
        ][..],
    )]);
}

/// Guard the embed loop's per-batch invariant: the embedder must return
/// exactly one vector per input. A short return would let `zip` silently
/// truncate, leaving the tail un-embedded and re-queued forever (infinite
/// loop). Returns Err so the caller fails loud instead of spinning.
fn embed_batch_len_ok(got: usize, want: usize) -> Result<()> {
    if got != want {
        anyhow::bail!(
            "embedder returned {} vectors for {} inputs; refusing to loop \
             (some docs would never embed). Check for empty/untokenizable bodies.",
            got,
            want
        );
    }
    Ok(())
}

fn min_embed_free_bytes_from_env() -> u64 {
    std::env::var("RECALL_MIN_EMBED_FREE_GB")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(DEFAULT_MIN_EMBED_FREE_GB)
        .mul_add(BYTES_PER_GB, 0.0)
        .ceil() as u64
}

fn embed_disk_space_decision(free_bytes: u64, min_bytes: u64) -> std::result::Result<(), String> {
    if free_bytes < min_bytes {
        return Err(format!(
            "free disk {:.3} GB is below embed apply floor {:.3} GB",
            free_bytes as f64 / BYTES_PER_GB,
            min_bytes as f64 / BYTES_PER_GB
        ));
    }
    Ok(())
}

#[cfg(windows)]
fn disk_free_bytes(path: &Path) -> Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide.push(0);
    let mut free_bytes = 0u64;
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut free_bytes,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        anyhow::bail!("could not read free disk space for embed preflight");
    }
    Ok(free_bytes)
}

#[cfg(not(windows))]
fn disk_free_bytes(_path: &Path) -> Result<u64> {
    Ok(u64::MAX)
}

fn ensure_embed_disk_space(root: &Path) -> Result<()> {
    let free_bytes = disk_free_bytes(root)?;
    let min_bytes = min_embed_free_bytes_from_env();
    if let Err(reason) = embed_disk_space_decision(free_bytes, min_bytes) {
        reply::card(&[(
            "BLOCKING",
            &[
                "Embedding disk-space gate failed; recall.db was not mutated.",
                reason.as_str(),
                "Free disk space or set RECALL_MIN_EMBED_FREE_GB only for an explicit local override.",
            ][..],
        )]);
        anyhow::bail!("embedding disk-space gate failed: {reason}");
    }
    Ok(())
}

fn embed_disk_space_gate_from_result(
    free_bytes: std::result::Result<u64, String>,
    min_bytes: u64,
) -> EmbedDiskSpaceGateJson {
    match free_bytes {
        Ok(free) => match embed_disk_space_decision(free, min_bytes) {
            Ok(()) => EmbedDiskSpaceGateJson {
                present: true,
                ok: true,
                status: "pass".to_string(),
                blocks_embedding_apply: false,
                reason: None,
                free_bytes: Some(free),
                free_gb: Some(free as f64 / BYTES_PER_GB),
                min_required_free_bytes: min_bytes,
                min_required_free_gb: min_bytes as f64 / BYTES_PER_GB,
                no_env_file_read: true,
                no_llm: true,
                no_paid_calls: true,
            },
            Err(reason) => EmbedDiskSpaceGateJson {
                present: true,
                ok: false,
                status: "blocking".to_string(),
                blocks_embedding_apply: true,
                reason: Some(reason),
                free_bytes: Some(free),
                free_gb: Some(free as f64 / BYTES_PER_GB),
                min_required_free_bytes: min_bytes,
                min_required_free_gb: min_bytes as f64 / BYTES_PER_GB,
                no_env_file_read: true,
                no_llm: true,
                no_paid_calls: true,
            },
        },
        Err(reason) => EmbedDiskSpaceGateJson {
            present: true,
            ok: false,
            status: "unknown".to_string(),
            blocks_embedding_apply: true,
            reason: Some(reason),
            free_bytes: None,
            free_gb: None,
            min_required_free_bytes: min_bytes,
            min_required_free_gb: min_bytes as f64 / BYTES_PER_GB,
            no_env_file_read: true,
            no_llm: true,
            no_paid_calls: true,
        },
    }
}

fn embed_disk_space_gate(root: &Path) -> EmbedDiskSpaceGateJson {
    let free_bytes = disk_free_bytes(root).map_err(|err| err.to_string());
    embed_disk_space_gate_from_result(free_bytes, min_embed_free_bytes_from_env())
}

fn parse_timeout_seconds(timeout_seconds: Option<String>) -> u64 {
    timeout_seconds
        .as_deref()
        .unwrap_or("20")
        .parse()
        .unwrap_or(DEFAULT_EMBED_DOCTOR_TIMEOUT_SECS)
        .clamp(1, 600)
}

fn parse_positive_usize(value: Option<String>, default: usize, name: &str) -> Result<usize> {
    let parsed = value
        .as_deref()
        .unwrap_or("")
        .trim()
        .parse::<usize>()
        .unwrap_or(default);
    if parsed == 0 {
        anyhow::bail!("{name} must be positive");
    }
    Ok(parsed)
}

fn run_embed_doctor_probe(timeout_secs: u64) -> Result<u128> {
    let exe = std::env::current_exe()?;
    let started = Instant::now();
    let mut child = Command::new(exe)
        .arg("__embed-doctor-worker")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let deadline = started + Duration::from_secs(timeout_secs);

    loop {
        match child.try_wait()? {
            Some(status) => {
                let output = child.wait_with_output()?;
                if status.success() {
                    return Ok(started.elapsed().as_millis());
                }
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                anyhow::bail!(
                    "embed doctor failed: status={} stdout={} stderr={}",
                    status,
                    stdout.trim(),
                    stderr.trim()
                );
            }
            None if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100));
            }
            None => {
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!(
                    "embed doctor timed out after {}s; refusing to load BGE in the main process",
                    timeout_secs
                );
            }
        }
    }
}

fn ensure_embed_doctor_ready() -> Result<()> {
    run_embed_doctor_probe(DEFAULT_EMBED_DOCTOR_TIMEOUT_SECS).map(|_| ())
}

fn embed_doctor_json(timeout_secs: u64) -> EmbedDoctorJson {
    let preflight = match bge::ort_runtime_preflight() {
        Ok(preflight) => preflight,
        Err(err) => {
            return EmbedDoctorJson {
                ok: false,
                timeout_seconds: timeout_secs,
                elapsed_ms: None,
                ort_version: None,
                error: Some(err.to_string()),
            };
        }
    };
    match run_embed_doctor_probe(timeout_secs) {
        Ok(elapsed_ms) => EmbedDoctorJson {
            ok: true,
            timeout_seconds: timeout_secs,
            elapsed_ms: Some(elapsed_ms),
            ort_version: Some(preflight.version),
            error: None,
        },
        Err(err) => EmbedDoctorJson {
            ok: false,
            timeout_seconds: timeout_secs,
            elapsed_ms: None,
            ort_version: Some(preflight.version),
            error: Some(err.to_string()),
        },
    }
}

/// recall's development folder inside the vault (private rules `recall_dev_dir`). Required by the
/// eval-script doctor checks; never guessed.
fn recall_dev_dir() -> Result<PathBuf> {
    store::recall_dev_dir().context(
        "this command needs `recall_dev_dir` in ~/.recall/private-rules.json (the vault folder holding recall's eval scripts)",
    )
}

fn default_path_probe_summary_path() -> Result<PathBuf> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_secs();
    Ok(recall_dev_dir()?
        .join("eval")
        .join("results")
        .join("coverage_repair")
        .join(format!("doctor-path-probe-{now}"))
        .join("coverage_path_probe.json"))
}

fn default_embedding_backlog_out_dir() -> Result<PathBuf> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis();
    Ok(recall_dev_dir()?
        .join("eval")
        .join("results")
        .join("embedding_backlog")
        .join(format!("doctor-embedding-backlog-{now}")))
}

fn run_targeted_path_probe_command(
    path_list: PathBuf,
    summary_json: Option<PathBuf>,
    require_embedded: bool,
    timeout_secs: u64,
) -> Result<TargetedPathProbeArtifactJson> {
    if !path_list.exists() {
        anyhow::bail!("path-list not found at {}", path_list.display());
    }
    let summary_path = match summary_json {
        Some(p) => p,
        None => default_path_probe_summary_path()?,
    };
    if summary_path.exists() {
        anyhow::bail!(
            "summary-json already exists at {}; choose a new artifact path",
            summary_path.display()
        );
    }
    let script = recall_dev_dir()?
        .join("eval")
        .join("coverage_path_probe.py");
    if !script.exists() {
        anyhow::bail!("coverage_path_probe.py not found at {}", script.display());
    }

    let started = Instant::now();
    let mut child = Command::new("py")
        .arg(script)
        .arg("--path-list")
        .arg(path_list)
        .arg("--summary-json")
        .arg(&summary_path)
        .arg("--json")
        .args(if require_embedded {
            vec!["--require-embedded"]
        } else {
            Vec::new()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let deadline = started + Duration::from_secs(timeout_secs.clamp(1, 600));

    loop {
        match child.try_wait()? {
            Some(status) => {
                let output = child.wait_with_output()?;
                let stdout = String::from_utf8_lossy(&output.stdout);
                let Some(json_start) = stdout.find('{') else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    anyhow::bail!(
                        "path probe returned no JSON: status={} stderr={}",
                        status,
                        stderr.trim()
                    );
                };
                let value: serde_json::Value = serde_json::from_str(&stdout[json_start..])?;
                if status.code().unwrap_or(2) > 1 {
                    let err = json_string(&value, "error")
                        .unwrap_or_else(|| "path probe failed".to_string());
                    anyhow::bail!("{err}");
                }
                return Ok(targeted_path_probe_artifact_from(Some((
                    summary_path,
                    value,
                ))));
            }
            None if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100));
            }
            None => {
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!("path probe timed out after {}s", timeout_secs.clamp(1, 600));
            }
        }
    }
}

fn run_embedding_backlog_command(
    unembedded_jsonl: Option<PathBuf>,
    batch_size: usize,
    max_per_prefix: Option<usize>,
    timeout_secs: u64,
) -> Result<EmbeddingBacklogArtifactJson> {
    let recall_root = recall_dev_dir()?;
    let script = recall_root.join("eval").join("embedding_backlog_plan.py");
    if !script.exists() {
        anyhow::bail!(
            "embedding_backlog_plan.py not found at {}",
            script.display()
        );
    }
    if let Some(unembedded_jsonl) = unembedded_jsonl.as_ref() {
        if !unembedded_jsonl.exists() {
            anyhow::bail!(
                "unembedded-jsonl not found at {}",
                unembedded_jsonl.display()
            );
        }
    }
    let out_dir = default_embedding_backlog_out_dir()?;
    let summary_path = out_dir.join("embedding_backlog_summary.json");

    let started = Instant::now();
    let mut command = Command::new("py");
    command.arg(script);
    if let Some(unembedded_jsonl) = unembedded_jsonl {
        command
            .arg("--unembedded-jsonl")
            .arg(unembedded_jsonl)
            .arg("--filter-current-recall-db");
    } else {
        command.arg("--from-current-recall-db");
    }
    command
        .arg("--out-dir")
        .arg(&out_dir)
        .arg("--batch-size")
        .arg(batch_size.to_string())
        .arg("--json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(max_per_prefix) = max_per_prefix {
        command
            .arg("--max-per-prefix")
            .arg(max_per_prefix.to_string());
    }
    let mut child = command.spawn()?;
    let deadline = started + Duration::from_secs(timeout_secs.clamp(1, 600));

    loop {
        match child.try_wait()? {
            Some(status) => {
                let output = child.wait_with_output()?;
                let stdout = String::from_utf8_lossy(&output.stdout);
                let Some(json_start) = stdout.find('{') else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    anyhow::bail!(
                        "embedding backlog planner returned no JSON: status={} stderr={}",
                        status,
                        stderr.trim()
                    );
                };
                let value: serde_json::Value = serde_json::from_str(&stdout[json_start..])?;
                if !status.success() {
                    let err = json_string(&value, "error")
                        .unwrap_or_else(|| "embedding backlog planner failed".to_string());
                    anyhow::bail!("{err}");
                }
                return Ok(embedding_backlog_artifact_from(Some((summary_path, value))));
            }
            None if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100));
            }
            None => {
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!(
                    "embedding backlog planner timed out after {}s",
                    timeout_secs.clamp(1, 600)
                );
            }
        }
    }
}

fn run_safe_embedding_batch_command(
    batch_size: usize,
    max_per_prefix: Option<usize>,
    apply: bool,
    postcheck: bool,
    skip_scan_rejected_json: Vec<PathBuf>,
    skip_known_rejected: bool,
    known_rejected_scan_limit: usize,
    timeout_secs: u64,
) -> Result<serde_json::Value> {
    let recall_root = recall_dev_dir()?;
    let script = recall_root.join("eval").join("safe_embedding_batch.py");
    if !script.exists() {
        anyhow::bail!("safe_embedding_batch.py not found at {}", script.display());
    }
    for path in skip_scan_rejected_json.iter() {
        if !path.exists() {
            anyhow::bail!("skip-scan-rejected-json not found at {}", path.display());
        }
    }
    let recall_exe = std::env::current_exe()?;
    let effective_timeout_secs = if timeout_secs == DEFAULT_EMBED_DOCTOR_TIMEOUT_SECS {
        600
    } else {
        timeout_secs
    }
    .clamp(1, 600);

    let started = Instant::now();
    let mut command = Command::new("py");
    command
        .arg(script)
        .arg("--batch-size")
        .arg(batch_size.to_string())
        .arg("--recall-exe")
        .arg(recall_exe)
        .arg("--json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(max_per_prefix) = max_per_prefix {
        command
            .arg("--max-per-prefix")
            .arg(max_per_prefix.to_string());
    }
    if apply {
        command.arg("--apply");
    }
    if postcheck {
        command.arg("--postcheck");
    }
    if skip_known_rejected {
        command.arg("--skip-known-rejected");
    }
    command
        .arg("--known-rejected-scan-limit")
        .arg(known_rejected_scan_limit.to_string());
    for path in skip_scan_rejected_json {
        command.arg("--skip-scan-rejected-json").arg(path);
    }

    let mut child = command.spawn()?;
    let deadline = started + Duration::from_secs(effective_timeout_secs);
    loop {
        match child.try_wait()? {
            Some(status) => {
                let output = child.wait_with_output()?;
                let stdout = String::from_utf8_lossy(&output.stdout);
                let Some(json_start) = stdout.find('{') else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    anyhow::bail!(
                        "safe embedding batch returned no JSON: status={} stderr={}",
                        status,
                        stderr.trim()
                    );
                };
                let value: serde_json::Value = serde_json::from_str(&stdout[json_start..])?;
                if !status.success() {
                    let err = json_string(&value, "error")
                        .unwrap_or_else(|| "safe embedding batch failed".to_string());
                    anyhow::bail!("{err}");
                }
                return Ok(value);
            }
            None if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100));
            }
            None => {
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!(
                    "safe embedding batch timed out after {}s",
                    effective_timeout_secs
                );
            }
        }
    }
}

fn run_coverage_delta_refresh_command(timeout_secs: u64) -> Result<serde_json::Value> {
    let recall_root = recall_dev_dir()?;
    let script = recall_root.join("eval").join("coverage_delta_refresh.py");
    if !script.exists() {
        anyhow::bail!(
            "coverage_delta_refresh.py not found at {}",
            script.display()
        );
    }
    let effective_timeout_secs = if timeout_secs == DEFAULT_EMBED_DOCTOR_TIMEOUT_SECS {
        600
    } else {
        timeout_secs
    }
    .clamp(1, 600);

    let started = Instant::now();
    let surrogate_manifests = latest_surrogate_manifests(&recall_root, 16);
    let mut command = Command::new("py");
    command.arg(script).arg("--json");
    for manifest in surrogate_manifests {
        command.arg("--surrogate-manifest").arg(manifest);
    }
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let deadline = started + Duration::from_secs(effective_timeout_secs);
    loop {
        match child.try_wait()? {
            Some(status) => {
                let output = child.wait_with_output()?;
                let stdout = String::from_utf8_lossy(&output.stdout);
                let Some(json_start) = stdout.find('{') else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    anyhow::bail!(
                        "coverage doctor returned no JSON: status={} stderr={}",
                        status,
                        stderr.trim()
                    );
                };
                let value: serde_json::Value = serde_json::from_str(&stdout[json_start..])?;
                if !status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    anyhow::bail!(
                        "coverage doctor failed: status={} stderr={}",
                        status,
                        stderr.trim()
                    );
                }
                return Ok(value);
            }
            None if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100));
            }
            None => {
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!(
                    "coverage doctor timed out after {}s",
                    effective_timeout_secs
                );
            }
        }
    }
}

fn latest_surrogate_manifests(recall_root: &Path, limit: usize) -> Vec<PathBuf> {
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
    let roots = [
        recall_root
            .join("eval")
            .join("results")
            .join("secret_surrogate"),
        recall_root
            .join("eval")
            .join("results")
            .join("secret_surrogates"),
    ];
    for root in roots {
        collect_surrogate_manifests(&root, &mut candidates);
    }
    candidates.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    candidates
        .into_iter()
        .take(limit)
        .map(|(_, path)| path)
        .collect()
}

fn collect_surrogate_manifests(root: &Path, out: &mut Vec<(std::time::SystemTime, PathBuf)>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_surrogate_manifests(&path, out);
            continue;
        }
        if path.file_name().and_then(|name| name.to_str()) != Some("surrogate_manifest.json") {
            continue;
        }
        let modified = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        out.push((modified, path));
    }
}

#[cfg(feature = "ops-health")]
fn memory_broker_db_health_json(timeout_secs: u64) -> MemoryBrokerDbHealthJson {
    match run_memory_broker_db_health_probe(timeout_secs) {
        Ok(value) => memory_broker_db_health_from_value(value),
        Err(err) => {
            let (error_class, error) = redact_memory_broker_error(Some(err.to_string()));
            MemoryBrokerDbHealthJson {
                ok: false,
                source: None,
                events: None,
                chunks: None,
                cards: None,
                embeddings: None,
                pending_embeddings: None,
                latest_event_at: None,
                latest_chunk_at: None,
                latest_card_at: None,
                control_plane_http_status: None,
                postgres_direct_ok: None,
                error_class,
                error,
            }
        }
    }
}

#[cfg(feature = "ops-health")]
fn run_memory_broker_db_health_probe(timeout_secs: u64) -> Result<serde_json::Value> {
    let script = store::vault_root()
        .join("_system")
        .join("operator-status-bus")
        .join("operator-status-bus.mjs");
    if !script.exists() {
        anyhow::bail!(
            "operator-status-bus script not found at {}",
            script.display()
        );
    }

    let started = Instant::now();
    let mut child = Command::new("node")
        .arg(script)
        .args(["memory-broker", "db-health", "--read-only"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let deadline = started + Duration::from_secs(timeout_secs.clamp(1, 60));

    loop {
        match child.try_wait()? {
            Some(status) => {
                let output = child.wait_with_output()?;
                let stdout = String::from_utf8_lossy(&output.stdout);
                let Some(json_start) = stdout.find('{') else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    anyhow::bail!(
                        "memory-broker db-health returned no JSON: status={} stderr={}",
                        status,
                        stderr.trim()
                    );
                };
                let value: serde_json::Value = serde_json::from_str(&stdout[json_start..])?;
                return Ok(value);
            }
            None if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100));
            }
            None => {
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!(
                    "memory-broker db-health timed out after {}s",
                    timeout_secs.clamp(1, 60)
                );
            }
        }
    }
}

#[cfg(feature = "ops-health")]
fn memory_broker_db_health_from_value(value: serde_json::Value) -> MemoryBrokerDbHealthJson {
    let body = value
        .get("body")
        .filter(|body| body.is_object())
        .unwrap_or(&serde_json::Value::Null);
    let raw_error = if json_bool(&value, "ok").unwrap_or(false) {
        None
    } else {
        json_string(&value, "error")
            .or_else(|| json_string(&value, "reason"))
            .or_else(|| {
                json_path(&value, &["control_plane_http", "error"])
                    .and_then(|item| item.as_str())
                    .map(ToString::to_string)
            })
            .or_else(|| {
                json_path(&value, &["control_plane_http", "reason"])
                    .and_then(|item| item.as_str())
                    .map(ToString::to_string)
            })
    };
    let (error_class, error) = redact_memory_broker_error(raw_error);
    MemoryBrokerDbHealthJson {
        ok: json_bool(&value, "ok").unwrap_or(false),
        source: json_string(&value, "source"),
        events: json_i64(body, "events"),
        chunks: json_i64(body, "chunks"),
        cards: json_i64(body, "cards"),
        embeddings: json_i64(body, "embeddings"),
        pending_embeddings: json_i64(body, "pending_embeddings"),
        latest_event_at: json_string(body, "latest_event_at"),
        latest_chunk_at: json_string(body, "latest_chunk_at"),
        latest_card_at: json_string(body, "latest_card_at"),
        control_plane_http_status: json_path_i64(&value, &["control_plane_http", "status"]),
        postgres_direct_ok: json_path_bool(&value, &["postgres_direct", "ok"]),
        error_class,
        error,
    }
}

#[cfg(feature = "ops-health")]
fn redact_memory_broker_error(raw_error: Option<String>) -> (Option<String>, Option<String>) {
    let Some(raw_error) = raw_error else {
        return (None, None);
    };
    let class = classify_memory_broker_error(&raw_error).to_string();
    (Some(class.clone()), Some(format!("redacted:{class}")))
}

#[cfg(feature = "ops-health")]
fn classify_memory_broker_error(error: &str) -> &'static str {
    let lower = error.to_ascii_lowercase();
    if lower.contains("10013")
        || (lower.contains("connection to server")
            && lower.contains("port")
            && lower.contains("failed: permission denied"))
    {
        "postgres_network_permission_denied"
    } else if lower.contains("permission denied") {
        "postgres_permission_denied"
    } else if lower.contains("timed out") || lower.contains("timeout") {
        "memory_broker_timeout"
    } else if lower.contains("operator-status-bus script not found") {
        "operator_status_bus_missing"
    } else if lower.contains("returned no json") {
        "memory_broker_db_health_no_json"
    } else if lower.contains("control-plane fetch failed") || lower.contains("fetch failed") {
        "control_plane_fetch_failed"
    } else {
        "memory_broker_unavailable"
    }
}

#[cfg(feature = "ops-health")]
fn tailscale_local_health_json(timeout_secs: u64) -> TailscaleLocalHealthJson {
    match run_tailscale_status_probe(timeout_secs) {
        Ok(value) => tailscale_local_health_from_value(value),
        Err(err) => TailscaleLocalHealthJson {
            ok: false,
            installed: false,
            backend_state: None,
            self_online: None,
            peer_count: None,
            online_peer_count: None,
            has_local_tailscale_ip: None,
            error: Some(err.to_string()),
        },
    }
}

#[cfg(feature = "ops-health")]
fn run_tailscale_status_probe(timeout_secs: u64) -> Result<serde_json::Value> {
    let started = Instant::now();
    let mut child = Command::new("tailscale")
        .args(["status", "--json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let deadline = started + Duration::from_secs(timeout_secs.clamp(1, 15));

    loop {
        match child.try_wait()? {
            Some(status) => {
                let output = child.wait_with_output()?;
                if !status.success() {
                    anyhow::bail!("tailscale status failed: {}", status);
                }
                let stdout = String::from_utf8_lossy(&output.stdout);
                let Some(json_start) = stdout.find('{') else {
                    anyhow::bail!("tailscale status returned no JSON");
                };
                let value: serde_json::Value = serde_json::from_str(&stdout[json_start..])?;
                return Ok(value);
            }
            None if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100));
            }
            None => {
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!(
                    "tailscale status timed out after {}s",
                    timeout_secs.clamp(1, 15)
                );
            }
        }
    }
}

#[cfg(feature = "ops-health")]
fn tailscale_local_health_from_value(value: serde_json::Value) -> TailscaleLocalHealthJson {
    let backend_state = json_string(&value, "BackendState");
    let self_online = json_path_bool(&value, &["Self", "Online"]);
    let has_local_tailscale_ip = json_path(&value, &["Self", "TailscaleIPs"])
        .and_then(|item| item.as_array())
        .map(|items| !items.is_empty());
    let peers = value
        .get("Peer")
        .and_then(|item| item.as_object())
        .map(|items| {
            let peer_count = items.len() as i64;
            let online_peer_count = items
                .values()
                .filter(|peer| peer.get("Online").and_then(|online| online.as_bool()) == Some(true))
                .count() as i64;
            (peer_count, online_peer_count)
        });
    let ok = backend_state.as_deref() == Some("Running")
        && self_online == Some(true)
        && has_local_tailscale_ip == Some(true);
    TailscaleLocalHealthJson {
        ok,
        installed: true,
        backend_state,
        self_online,
        peer_count: peers.map(|(peer_count, _)| peer_count),
        online_peer_count: peers.map(|(_, online_peer_count)| online_peer_count),
        has_local_tailscale_ip,
        error: None,
    }
}

#[cfg(feature = "ops-health")]
fn gateway_local_health_json() -> GatewayLocalHealthJson {
    // Optional gateway probe: only meaningful when COGNESIA_GATEWAY_ROOT is set.
    let gateway_root = std::env::var_os("COGNESIA_GATEWAY_ROOT")
        .map(PathBuf::from)
        .unwrap_or_default();
    let mcp_code_present = gateway_root.join("mcp_server.py").exists();
    let rest_code_present = gateway_root.join("rest_face.py").exists();
    let mcp_probes: Vec<GatewayMcpProbe> = gateway_mcp_candidate_ports()
        .into_iter()
        .map(probe_gateway_mcp_port)
        .collect();
    let selected_mcp_probe =
        select_gateway_mcp_probe(&mcp_probes).expect("gateway MCP candidate ports are never empty");
    let mcp_port = selected_mcp_probe.port;
    let rest_port = parse_u16_env("COG_GATEWAY_REST_PORT")
        .or_else(|| parse_u16_env("COG_GW_PORT"))
        .unwrap_or(8810);
    let mcp_status = selected_mcp_probe.mcp_status;
    let mcp_health_status = selected_mcp_probe.mcp_health_status;
    let events_status = selected_mcp_probe.events_status;
    let rest_health_status =
        local_http_status("127.0.0.1", rest_port, "/health", Duration::from_secs(2)).ok();
    let mcp_port_listening = gateway_mcp_port_listening(selected_mcp_probe);
    let rest_port_listening = rest_health_status.is_some();
    let mcp_auth_required = mcp_status.map(|status| status == 401 || status == 403);
    let events_auth_required = events_status.map(|status| status == 401 || status == 403);
    let sse_bearer_configured =
        env_var_present("COG_SSE_BEARER") || events_auth_required == Some(true);
    let gateway_routes_observed = gateway_mcp_routes_observed(selected_mcp_probe);
    let rest_health_ok = rest_health_status.map(|status| status == 200);
    let ok = mcp_code_present
        && rest_code_present
        && mcp_port_listening
        && gateway_routes_observed
        && sse_bearer_configured
        && events_auth_required == Some(true);
    let error = if ok {
        None
    } else if !mcp_code_present || !rest_code_present {
        Some("gateway_code_missing".to_string())
    } else if !mcp_port_listening {
        Some("gateway_mcp_http_not_listening".to_string())
    } else if !gateway_routes_observed {
        Some("gateway_mcp_expected_routes_not_observed".to_string())
    } else if !sse_bearer_configured {
        Some("gateway_sse_bearer_not_configured".to_string())
    } else if events_auth_required != Some(true) {
        Some("gateway_events_auth_gate_not_observed".to_string())
    } else {
        Some("gateway_local_not_ready".to_string())
    };
    GatewayLocalHealthJson {
        ok,
        mcp_port,
        rest_port,
        mcp_code_present,
        rest_code_present,
        mcp_port_listening,
        mcp_status,
        mcp_auth_required,
        mcp_health_status,
        events_status,
        events_auth_required,
        sse_bearer_configured,
        rest_port_listening,
        rest_health_status,
        rest_health_ok,
        error,
    }
}

#[cfg(feature = "ops-health")]
#[derive(Clone, Copy)]
struct GatewayMcpProbe {
    port: u16,
    mcp_status: Option<i64>,
    mcp_health_status: Option<i64>,
    events_status: Option<i64>,
}

#[cfg(feature = "ops-health")]
fn gateway_mcp_candidate_ports() -> Vec<u16> {
    gateway_mcp_candidate_ports_from_env_port(parse_u16_env("COG_GATEWAY_MCP_PORT"))
}

#[cfg(feature = "ops-health")]
fn gateway_mcp_candidate_ports_from_env_port(env_port: Option<u16>) -> Vec<u16> {
    let mut ports = Vec::new();
    if let Some(port) = env_port {
        push_unique_port(&mut ports, port);
    }
    push_unique_port(&mut ports, 9876);
    push_unique_port(&mut ports, 9986);
    ports
}

#[cfg(feature = "ops-health")]
fn select_gateway_mcp_probe(probes: &[GatewayMcpProbe]) -> Option<&GatewayMcpProbe> {
    probes
        .iter()
        .find(|probe| gateway_mcp_auth_gate_observed(probe))
        .or_else(|| {
            probes
                .iter()
                .find(|probe| gateway_mcp_routes_observed(probe))
        })
        .or_else(|| {
            probes
                .iter()
                .find(|probe| gateway_mcp_port_listening(probe))
        })
        .or_else(|| probes.first())
}

fn push_unique_port(ports: &mut Vec<u16>, port: u16) {
    if !ports.contains(&port) {
        ports.push(port);
    }
}

#[cfg(feature = "ops-health")]
fn probe_gateway_mcp_port(port: u16) -> GatewayMcpProbe {
    GatewayMcpProbe {
        port,
        mcp_status: local_http_status("127.0.0.1", port, "/mcp", Duration::from_secs(2)).ok(),
        mcp_health_status: local_http_status("127.0.0.1", port, "/health", Duration::from_secs(2))
            .ok(),
        events_status: local_http_status("127.0.0.1", port, "/events", Duration::from_secs(2)).ok(),
    }
}

#[cfg(feature = "ops-health")]
fn gateway_mcp_port_listening(probe: &GatewayMcpProbe) -> bool {
    probe.mcp_status.is_some() || probe.mcp_health_status.is_some() || probe.events_status.is_some()
}

#[cfg(feature = "ops-health")]
fn gateway_mcp_routes_observed(probe: &GatewayMcpProbe) -> bool {
    matches!(probe.events_status, Some(200 | 401 | 403))
        || matches!(probe.mcp_health_status, Some(200))
}

#[cfg(feature = "ops-health")]
fn gateway_mcp_auth_gate_observed(probe: &GatewayMcpProbe) -> bool {
    matches!(probe.events_status, Some(401 | 403))
}

#[cfg(feature = "ops-health")]
fn ops_health_ready(g: &MemoryDoctorGatesJson) -> bool {
    g.memory_broker_db_proven && g.tailscale_local_ready && g.gateway_local_ready
}

#[cfg(not(feature = "ops-health"))]
fn ops_health_ready(_g: &MemoryDoctorGatesJson) -> bool {
    true
}

fn env_var_present(name: &str) -> bool {
    std::env::var_os(name)
        .map(|value| !value.is_empty())
        .unwrap_or(false)
}

fn parse_u16_env(name: &str) -> Option<u16> {
    std::env::var_os(name)?.to_string_lossy().parse().ok()
}

fn local_http_status(host: &str, port: u16, path: &str, timeout: Duration) -> Result<i64> {
    let mut addrs = (host, port).to_socket_addrs()?;
    let addr = addrs
        .next()
        .ok_or_else(|| anyhow::anyhow!("no socket address resolved"))?;
    let mut stream = TcpStream::connect_timeout(&addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let request =
        format!("GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes())?;
    let mut buffer = [0_u8; 64];
    let read = stream.read(&mut buffer)?;
    let head = std::str::from_utf8(&buffer[..read]).unwrap_or("");
    let Some(status_text) = head.split_whitespace().nth(1) else {
        anyhow::bail!("no HTTP status in response");
    };
    Ok(status_text.parse::<i64>()?)
}

fn health_doctor_json(timeout_secs: u64) -> Result<HealthDoctorJson> {
    let store = Store::open()?;
    let recall_status = status_json_payload_from_store(&store)?;
    let queryable_embeddings = store.load_all_embeddings()?.len() as i64;
    let bge_vectors = BgeVectorStatsJson {
        stored_valid_embeddings: recall_status.embeddings,
        queryable_embeddings,
        excluded_embeddings: recall_status
            .embeddings
            .saturating_sub(queryable_embeddings),
    };
    let embed_doctor = embed_doctor_json(timeout_secs);
    Ok(HealthDoctorJson {
        ok: recall_status.docs_valid > 0 && embed_doctor.ok && recall_status.rg_available,
        recall_status,
        bge_vectors,
        embed_doctor,
        no_llm: true,
        no_paid_calls: true,
    })
}

fn cmd_doctor(
    check: String,
    timeout_seconds: Option<String>,
    json: bool,
    path_list: Option<PathBuf>,
    summary_json: Option<PathBuf>,
    require_embedded: bool,
    batch_size: Option<String>,
    max_per_prefix: Option<String>,
    unembedded_jsonl: Option<PathBuf>,
    apply: bool,
    postcheck: bool,
    skip_scan_rejected_json: Vec<PathBuf>,
    skip_known_rejected: bool,
    known_rejected_scan_limit: Option<String>,
) -> Result<()> {
    let timeout_secs = parse_timeout_seconds(timeout_seconds);
    match check.to_lowercase().as_str() {
        "embed" => {
            let result = embed_doctor_json(timeout_secs);
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
                return Ok(());
            }
            if result.ok {
                let l1 = format!(
                    "Embed doctor passed in {}ms",
                    result.elapsed_ms.unwrap_or(0)
                );
                let l2 =
                    "BGE loaded and embedded one local probe string in a watchdog child process";
                reply::card(&[("WHAT I DID", &[l1.as_str(), l2][..])]);
                Ok(())
            } else {
                anyhow::bail!(
                    "{}",
                    result
                        .error
                        .unwrap_or_else(|| "embed doctor failed".to_string())
                )
            }
        }
        "health" => {
            let result = health_doctor_json(timeout_secs)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
                return Ok(());
            }
            let status_line = if result.ok { "healthy" } else { "degraded" };
            let mut l1 = format!("Recall health is {status_line}");
            if !result.recall_status.rg_available {
                l1 = format!("{l1}. {}", store::RG_MISSING);
            }
            let l2 = format!(
                "Docs valid: {}  embeddings: {}  queryable BGE: {}  coverage: {:.4}%",
                result.recall_status.docs_valid,
                result.recall_status.embeddings,
                result.bge_vectors.queryable_embeddings,
                result.recall_status.coverage_pct
            );
            let l3 = if result.embed_doctor.ok {
                format!(
                    "Embed doctor passed in {}ms",
                    result.embed_doctor.elapsed_ms.unwrap_or(0)
                )
            } else {
                format!(
                    "Embed doctor failed: {}",
                    result
                        .embed_doctor
                        .error
                        .unwrap_or_else(|| "unknown error".to_string())
                )
            };
            reply::card(&[("WHAT I DID", &[l1.as_str(), l2.as_str(), l3.as_str()][..])]);
            Ok(())
        }
        "memory" => {
            let result = memory_doctor_json()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
                return Ok(());
            }
            let l1 = format!("Memory doctor status: {}", result.status);
            let l2 = format!(
                "Recall docs: {} valid / {} total; embeddings: {}; coverage: {:.4}%",
                result.recall_status.docs_valid,
                result.recall_status.docs_total,
                result.recall_status.embeddings,
                result.recall_status.coverage_pct
            );
            let l3 = format!(
                "Active retrieval: denoise={} fusion={} rrf_k={} min_depth={} depth_multiplier={}",
                result.active_retrieval_modes.denoise_mode,
                result.active_retrieval_modes.fusion_mode,
                result.active_retrieval_modes.rrf_k,
                result.active_retrieval_modes.min_fusion_depth,
                result.active_retrieval_modes.fusion_depth_multiplier
            );
            let l4 = format!(
                "Latest bench: fused={}/{} mode={}",
                result.latest_release_benchmark.fused_hits.unwrap_or(0),
                result
                    .latest_release_benchmark
                    .positive_fixtures
                    .unwrap_or(0),
                result
                    .latest_release_benchmark
                    .denoise_mode
                    .unwrap_or_else(|| "unknown".to_string())
            );
            let l5 = format!(
                "RAGAS smoke: passed={} leak_count={}",
                result.ragas_smoke.passed.unwrap_or(false),
                result.ragas_smoke.secret_leak_count.unwrap_or(-1)
            );
            #[cfg(feature = "ops-health")]
            let l6 = format!(
                "MemoryBroker DB: ok={} source={} events={} chunks={} embeddings={} http_status={}",
                result.memory_broker_db_health.ok,
                result
                    .memory_broker_db_health
                    .source
                    .as_deref()
                    .unwrap_or("unknown"),
                result.memory_broker_db_health.events.unwrap_or(0),
                result.memory_broker_db_health.chunks.unwrap_or(0),
                result.memory_broker_db_health.embeddings.unwrap_or(0),
                result
                    .memory_broker_db_health
                    .control_plane_http_status
                    .unwrap_or(0)
            );
            #[cfg(feature = "ops-health")]
            let l7 = format!(
                "Tailscale local: ok={} backend={} self_online={} peers_online={}/{}",
                result.tailscale_local_health.ok,
                result
                    .tailscale_local_health
                    .backend_state
                    .as_deref()
                    .unwrap_or("unknown"),
                result.tailscale_local_health.self_online.unwrap_or(false),
                result.tailscale_local_health.online_peer_count.unwrap_or(0),
                result.tailscale_local_health.peer_count.unwrap_or(0)
            );
            #[cfg(feature = "ops-health")]
            let l8 = format!(
                "Gateway local: ok={} mcp_listening={} mcp_status={} mcp_auth_required={} rest_health_status={}",
                result.gateway_local_health.ok,
                result.gateway_local_health.mcp_port_listening,
                result.gateway_local_health.mcp_status.unwrap_or(0),
                result.gateway_local_health.mcp_auth_required.unwrap_or(false),
                result.gateway_local_health.rest_health_status.unwrap_or(0)
            );
            let l9 = format!(
                "Targeted path probe: present={} ok={} selected={} router_missing={} recall_missing={} redacted={}",
                result.targeted_path_probe.present,
                result.targeted_path_probe.ok.unwrap_or(false),
                result.targeted_path_probe.selected_count.unwrap_or(0),
                result.targeted_path_probe.router_missing_count.unwrap_or(0),
                result.targeted_path_probe.recall_missing_count.unwrap_or(0),
                result.gates.targeted_path_probe_redacted
            );
            let l10 = format!(
                "Embedding backlog: present={} ok={} total_unembedded={} selected={} stale_removed={} redacted={}",
                result.embedding_backlog.present,
                result.embedding_backlog.ok.unwrap_or(false),
                result.embedding_backlog.total_unembedded.unwrap_or(0),
                result.embedding_backlog.selected_count.unwrap_or(0),
                result.embedding_backlog.stale_artifact_rows_removed.unwrap_or(0),
                result.gates.embedding_backlog_redacted
            );
            reply::card(&[(
                "WHAT I DID",
                &[
                    l1.as_str(),
                    l2.as_str(),
                    l3.as_str(),
                    l4.as_str(),
                    l5.as_str(),
                    #[cfg(feature = "ops-health")]
                    l6.as_str(),
                    #[cfg(feature = "ops-health")]
                    l7.as_str(),
                    #[cfg(feature = "ops-health")]
                    l8.as_str(),
                    l9.as_str(),
                    l10.as_str(),
                ][..],
            )]);
            Ok(())
        }
        "coverage" | "coverage-delta" | "coverage_delta" => {
            let result = run_coverage_delta_refresh_command(timeout_secs)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
                return Ok(());
            }
            let status = json_string(&result, "status").unwrap_or_else(|| "unknown".to_string());
            let run_id = json_string(&result, "run_id").unwrap_or_else(|| "unknown".to_string());
            let coverage = json_path_f64(
                &result,
                &[
                    "coverage_estimate_on_baseline_inventory",
                    "recall_embedding_coverage_pct",
                ],
            )
            .unwrap_or(0.0);
            let valid_docs = json_path_i64(
                &result,
                &[
                    "coverage_estimate_on_baseline_inventory",
                    "recall_valid_docs",
                ],
            )
            .unwrap_or(0);
            let embeddings = json_path_i64(
                &result,
                &[
                    "coverage_estimate_on_baseline_inventory",
                    "recall_embedded_valid_docs",
                ],
            )
            .unwrap_or(0);
            let unembedded = json_path_i64(
                &result,
                &[
                    "coverage_estimate_on_baseline_inventory",
                    "recall_unembedded_valid_docs",
                ],
            )
            .unwrap_or(0);
            let recall_missing =
                json_path_i64(&result, &["known_gap_counts", "remaining_recall_missing"])
                    .unwrap_or(0);
            let router_missing =
                json_path_i64(&result, &["known_gap_counts", "remaining_router_missing"])
                    .unwrap_or(0);
            let router_stale =
                json_path_i64(&result, &["known_gap_counts", "remaining_router_stale"])
                    .unwrap_or(0);
            let redacted = json_path_bool(&result, &["raw_paths_included"]) == Some(false)
                && json_path_bool(&result, &["raw_contexts_included"]) == Some(false)
                && json_path_bool(&result, &["file_contents_read"]) == Some(false)
                && json_path_bool(
                    &result,
                    &["redacted_surrogate_coverage", "raw_secret_values_included"],
                ) != Some(true);
            let no_spend = json_path_bool(&result, &["no_llm"]) == Some(true)
                && json_path_bool(&result, &["no_paid_calls"]) == Some(true);
            let summary_json = json_path(&result, &["artifacts", "summary_json"])
                .and_then(|item| item.as_str())
                .unwrap_or("none");
            let l1 = format!("Coverage doctor status: {status} run_id={run_id}");
            let l2 = format!(
                "Embeddings={embeddings}/{valid_docs} coverage={coverage:.4}% unembedded={unembedded}"
            );
            let l3 = format!(
                "Known gaps: recall_missing={recall_missing} router_missing={router_missing} router_stale={router_stale}"
            );
            let l4 = format!("Redacted={redacted} no_spend_no_llm={no_spend}");
            let l5 = format!("Artifact={summary_json}");
            reply::card(&[(
                "WHAT I DID",
                &[
                    l1.as_str(),
                    l2.as_str(),
                    l3.as_str(),
                    l4.as_str(),
                    l5.as_str(),
                ][..],
            )]);
            Ok(())
        }
        "embedding-backlog" | "embedding_backlog" => {
            let batch_size = parse_positive_usize(batch_size, 64, "--batch-size")?;
            let max_per_prefix = match max_per_prefix {
                Some(raw) => Some(parse_positive_usize(Some(raw), 1, "--max-per-prefix")?),
                None => None,
            };
            let result = run_embedding_backlog_command(
                unembedded_jsonl,
                batch_size,
                max_per_prefix,
                timeout_secs,
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
                return Ok(());
            }
            let redacted = result.raw_paths_included == Some(false)
                && result.raw_contexts_included == Some(false)
                && result.file_contents_read == Some(false)
                && result.path_list_contains_raw_paths == Some(false);
            let l1 = format!(
                "Embedding backlog status: {}",
                if result.ok == Some(true) {
                    "ready"
                } else {
                    "degraded"
                }
            );
            let l2 = format!(
                "Total_unembedded={} nonrisk={} path_risk={} selected={}/{}",
                result.total_unembedded.unwrap_or(0),
                result.nonrisk_unembedded.unwrap_or(0),
                result.path_risk_unembedded.unwrap_or(0),
                result.selected_count.unwrap_or(0),
                result.requested_size.unwrap_or(0)
            );
            let l3 = format!(
                "Current filter enabled={} current_unembedded={} stale_artifact_rows_removed={}",
                result.current_filter_enabled.unwrap_or(false),
                result.current_unembedded_paths.unwrap_or(0),
                result.stale_artifact_rows_removed.unwrap_or(0)
            );
            let l4 = format!(
                "Redacted={} artifact={}",
                redacted,
                result.summary_json.as_deref().unwrap_or("none")
            );
            reply::card(&[(
                "WHAT I DID",
                &[l1.as_str(), l2.as_str(), l3.as_str(), l4.as_str()][..],
            )]);
            Ok(())
        }
        "safe-embed-batch" | "safe_embed_batch" => {
            let batch_size = parse_positive_usize(batch_size, 8, "--batch-size")?;
            let max_per_prefix = match max_per_prefix {
                Some(raw) => Some(parse_positive_usize(Some(raw), 1, "--max-per-prefix")?),
                None => None,
            };
            let known_rejected_scan_limit =
                parse_positive_usize(known_rejected_scan_limit, 64, "--known-rejected-scan-limit")?;
            let result = run_safe_embedding_batch_command(
                batch_size,
                max_per_prefix,
                apply,
                postcheck,
                skip_scan_rejected_json,
                skip_known_rejected,
                known_rejected_scan_limit,
                timeout_secs,
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
                return Ok(());
            }
            let status = json_string(&result, "status").unwrap_or_else(|| "unknown".to_string());
            let selected = json_path_i64(&result, &["selection", "selected_count"]).unwrap_or(0);
            let approved = json_path_i64(&result, &["selection", "approved_count"]).unwrap_or(0);
            let rejected = json_path_i64(&result, &["selection", "rejected_count"]).unwrap_or(0);
            let hits = json_path_i64(&result, &["selection", "secret_pattern_hits"]).unwrap_or(0);
            let dry_run =
                json_path_i64(&result, &["selection", "dry_run_embed_count"]).unwrap_or(0);
            let embedded_delta = json_path_i64(&result, &["apply", "embedded_delta"]).unwrap_or(0);
            let embeddings_after =
                json_path_i64(&result, &["apply", "embeddings_after"]).unwrap_or(0);
            let coverage = json_path_f64(
                &result,
                &[
                    "postchecks",
                    "coverage_delta_refresh",
                    "embedding_coverage_pct",
                ],
            )
            .unwrap_or(0.0);
            let ragas_passed =
                json_path_bool(&result, &["postchecks", "ragas_smoke", "passed"]).unwrap_or(false);
            let ragas_leaks =
                json_path_i64(&result, &["postchecks", "ragas_smoke", "secret_leak_count"])
                    .unwrap_or(0);
            let enterprise_status =
                json_path(&result, &["postchecks", "enterprise_readiness", "status"])
                    .and_then(|item| item.as_str())
                    .unwrap_or("not_run");
            let l1 = format!("Safe embedding batch status: {status}");
            let l2 = format!(
                "Selected={selected} approved={approved} rejected={rejected} secret_hits={hits} dry_run_embed_count={dry_run}"
            );
            let l3 = format!("Embedded_delta={embedded_delta} embeddings_after={embeddings_after}");
            let l4 = format!(
                "Postchecks: coverage={coverage:.4}% ragas_passed={ragas_passed} ragas_leaks={ragas_leaks} enterprise={enterprise_status}"
            );
            let l5 = "Redacted summary only: raw paths stay in machine path-list artifacts";
            reply::card(&[(
                "WHAT I DID",
                &[l1.as_str(), l2.as_str(), l3.as_str(), l4.as_str(), l5][..],
            )]);
            Ok(())
        }
        "path-probe" | "path_probe" => {
            let Some(path_list) = path_list else {
                anyhow::bail!("doctor path-probe requires --path-list");
            };
            let result = run_targeted_path_probe_command(
                path_list,
                summary_json,
                require_embedded,
                timeout_secs,
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
                return Ok(());
            }
            let l1 = format!(
                "Targeted path probe status: {}",
                result.status.as_deref().unwrap_or("unknown")
            );
            let l2 = format!(
                "Selected={} router_missing={} router_stale={} recall_missing={} recall_unembedded={}",
                result.selected_count.unwrap_or(0),
                result.router_missing_count.unwrap_or(0),
                result.router_stale_count.unwrap_or(0),
                result.recall_missing_count.unwrap_or(0),
                result.recall_unembedded_count.unwrap_or(0)
            );
            let redacted = result.raw_paths_included == Some(false)
                && result.raw_contexts_included == Some(false)
                && result.file_contents_read == Some(false);
            let l3 = format!(
                "Redacted={} no_file_bodies={} artifact={}",
                redacted,
                result.file_contents_read == Some(false),
                result.summary_json.as_deref().unwrap_or("none")
            );
            reply::card(&[("WHAT I DID", &[l1.as_str(), l2.as_str(), l3.as_str()][..])]);
            Ok(())
        }
        _ => {
            reply::blocking(
                "doctor",
                "supported checks: embed, health, memory, coverage, path-probe, embedding-backlog, safe-embed-batch",
            );
            Ok(())
        }
    }
}

fn cmd_embed_doctor_worker() -> Result<()> {
    let preflight = bge::ort_runtime_preflight()?;
    let embedder = bge::Embedder::new()?;
    let texts = vec!["recall embed doctor smoke".to_string()];
    let vecs = embedder.embed(&texts)?;
    embed_batch_len_ok(vecs.len(), texts.len())?;
    println!(
        "{{\"ok\":true,\"vectors\":{},\"dim\":{},\"ort_version\":\"{}\"}}",
        vecs.len(),
        bge::DIM,
        preflight.version
    );
    Ok(())
}

/// Hard ceiling on how many CHUNK ROWS go into one ONNX call, for both embed queues.
///
/// The unit matters: `--batch` counts documents, the tensor counts chunks, and
/// chunks-per-document is unbounded. See the comment at the path-list loop for the measurement
/// that produced this number. 64 since 2026-10-31: with the ORT arena off (`bge::cpu_ep`) a
/// 64-row worst-case batch peaks near 2.5 GB vs 6.3 GB at 192 (measured), ~5% slower.
const MAX_CHUNK_ROWS_PER_BATCH: usize = bge::MAX_ROWS_PER_ONNX_CALL;

fn cmd_embed(
    batch: Option<String>,
    path_list: Option<PathBuf>,
    dry_run: bool,
    allow_benchmark_regression: bool,
    force: bool,
) -> Result<()> {
    // clamp(1, ..): batch=0 makes docs_missing_embeddings return empty every
    // loop, so embed silently no-ops instead of vectorizing anything.
    let batch: usize = batch
        .as_deref()
        .unwrap_or("256")
        .parse()
        .unwrap_or(256)
        .clamp(1, 2048);
    let store = Store::open()?;
    let root = store::vault_root();


    if let Some(path_list) = path_list.as_ref() {
        // PORTED TO PER-CHUNK VECTORS (was the last caller of the pooled write path).
        //
        // This repair path used to embed each document as ONE truncated
        // `title || ' ' || body` string into the legacy `embeddings` table. That made it
        // silently useless the moment the rollout started: the query path prefers
        // `bge_chunk_embeddings` for any document that has chunk rows, and a repair run here
        // never wrote one — so it "fixed" coverage that no search could see, and left the
        // document looking done to a queue that was asking the wrong table.
        //
        // It now shares the chunker, the untruncated tokenizer, the delete-then-insert
        // discipline and the destination table with the broad queue below. The only thing
        // that still differs is WHICH documents are embedded, which is the entire reason
        // the flag exists.
        let paths = read_path_list(path_list)?;
        // `--force` re-embeds documents that already have vectors. Without it a chunker fix
        // reaches nothing, because every document already passes the "is it embedded" test.
        let work = if force {
            store.docs_for_paths(&root, &paths)?
        } else {
            store.docs_missing_chunk_embeddings_for_paths(&root, &paths)?
        };
        if dry_run {
            let l1 = format!("Would embed {} indexed docs from path list", work.len());
            let l2 = "Dry-run only: BGE was not loaded and recall.db was not mutated";
            reply::card(&[("WHAT I DID", &[l1.as_str(), l2][..])]);
            return Ok(());
        }
        if work.is_empty() {
            reply::card(&[(
                "WHAT I DID",
                &["No path-list docs need embeddings; BGE was not loaded"][..],
            )]);
            return Ok(());
        }
        ensure_embedding_benchmark_gate(allow_benchmark_regression)?;
        ensure_embed_disk_space(&root)?;

        // 🔴 NOT `count_chunk_embedded_docs()`. That counts documents with any chunk row,
        // which after the staleness fix includes every STALE document - so it reported the
        // whole corpus as already done and the progress line climbed past the total
        // (`105 / 100`). `already` must mean "needs nothing", which is the corpus
        // minus the queue.
        // ONE statement for both numbers - see the broad branch for why.
        let (total, remaining_now) = store.embedding_progress()?;
        let already = total - remaining_now;
        println!("\n{}", reply::paint(reply::BOLD, &format!("recall embed: {already} of {total} docs already vectorized")));

        ensure_embed_doctor_ready()?;
        // Fill any missing content hashes, AFTER the preflight gates and before the work is
        // done. A row written before these columns existed has `content_sha256 IS NULL`; the
        // write path heals that itself, but doing it here means the queue, the progress counter
        // and the certification agree from the first batch instead of converging over a run.
        //
        // 🔴 Deliberately placed after the disk-space, benchmark and doctor gates: this writes to
        // every unhashed row in the corpus, and a command about to REFUSE to run should not have
        // already rewritten the corpus on its way to refusing.
        //
        // That is a narrower promise than "nothing is mutated before a refusal", and the narrower
        // wording is deliberate: `Store::open` above runs `init_schema`, which migrates the schema
        // - adding columns, and replacing the `documents_au` trigger - before any gate has run. A
        // refused command therefore CAN leave a migrated schema behind. That migration is additive,
        // idempotent and required before any command can address the database at all, so it is not
        // gateable without a second read-only open; the corpus-wide row write below is, and is.
        // Idempotent and cheap - on a hashed corpus it is one SELECT that returns nothing - and it
        // touches neither title nor body, so it fires no FTS work.
        let mut hashed = 0usize;
        loop {
            let n = store.backfill_content_hashes(5_000)?;
            if n == 0 {
                break;
            }
            hashed += n;
        }
        if hashed > 0 {
            eprintln!("recall embed: backfilled {hashed} missing content hashes first");
        }
        let embedder = bge::Embedder::new()?;
        let chunk_tokenizer = bge::BgeTokenizer::load()?;
        let mut done = 0usize;
        let mut chunks_written = 0usize;
        // Rolled-back replacements: an ingest rewrote the document mid-embed. This counts
        // REJECTIONS, not distinct documents - one document retried twice counts twice.
        let mut rejected_writes = 0usize;
        // 🔴 BATCH BY CHUNKS, NOT BY DOCUMENTS. `-b` counts documents, but the tensor handed to
        // ONNX is one ROW PER CHUNK, and chunks-per-document is unbounded — a 200-page note
        // produces hundreds. So a document-counted batch has no upper bound on its tensor size.
        //
        // Measured 2026-08-20 on this backfill: `-b 32` (nominally ~240 rows at the vault's
        // 7.5 chunks/doc average) grew the process from 6.5 GB to 23.8 GB of private bytes in
        // twelve minutes, and throughput COLLAPSED from 42.4 docs/min to ~11 as the box went
        // into memory pressure. ONNX Runtime's arena grows to the largest allocation it has
        // ever seen and does not give it back, so one batch that happens to collect long
        // documents raises the floor for the rest of the run.
        //
        // Capping the row count makes the peak tensor — and therefore the arena — bounded by a
        // constant instead of by the longest document in the corpus. `-b` still applies as a
        // document ceiling, so an explicit small `-b` is still honoured.
        let mut queue = work.iter().peekable();
        while queue.peek().is_some() {
            // The hash of the body being embedded travels WITH the batch. It cannot be
            // read back from the row at write time: embedding happens outside the write
            // transaction, so by then the row may hold a different body, and stamping
            // that body's hash against these vectors is the staleness bug reintroduced.
            let mut per_doc: Vec<(i64, Vec<bge::ChunkSpan>, String)> = Vec::new();
            let mut texts: Vec<String> = Vec::new();
            let mut docs_in_batch = 0usize;
            // Documents whose replacement actually COMMITTED. `docs_in_batch` counts what
            // was attempted; adding attempts to `done` is how a run reports more documents
            // finished than exist.
            let mut certified_in_batch = 0usize;
            let chunks_before_batch = chunks_written;
            while let Some((id, title, body, _)) = queue.peek() {
                let spans = bge::chunk_for_embedding_spans(title, body, &chunk_tokenizer);
                // Always take at least one document, even if it alone blows the cap — otherwise
                // a single very long note would stall the queue forever instead of embedding.
                if !texts.is_empty()
                    && (texts.len() + spans.len() > MAX_CHUNK_ROWS_PER_BATCH
                        || docs_in_batch >= batch)
                {
                    break;
                }
                texts.extend(spans.iter().map(|s| s.text.clone()));
                per_doc.push((*id, spans, store::sha256_hex(body.as_bytes())));
                docs_in_batch += 1;
                queue.next();
            }
            let vecs = embedder.embed(&texts)?;
            embed_batch_len_ok(vecs.len(), texts.len())?;
            let mut cursor = 0usize;
            for (id, spans, embedded_of) in per_doc {
                // Clear first, for the same reason the broad queue does: an upsert only
                // overwrites the indices it is handed, so a document that shrank from 8
                // chunks to 3 would keep chunks 3..8 of its previous body and go on matching
                // text it no longer contains.
                // ONE transaction for the delete, the inserts AND the embedded_sha256
                // stamp. Written as three separate autocommitted statements, a crash
                // between the vectors and the stamp leaves the document looking fresh
                // while carrying vectors built from a body it no longer has - the exact
                // staleness bug, re-created.
                let rows: Vec<(Vec<u8>, store::ChunkPlacement<'_>)> = spans
                    .iter()
                    .zip(vecs[cursor..cursor + spans.len()].iter())
                    .map(|(span, v)| (bge::to_bytes(v), span.into()))
                    .collect();
                // A false return means the row changed under us between the queue read and
                // this write. The ENTIRE replacement is rolled back - the document keeps
                // the vectors it already had - because committing vectors that describe a
                // body the row no longer holds can leave it looking permanently fresh.
                //
                // Whether a rejected document is retried in THIS run depends on the branch:
                // the broad loop re-reads the queue on its outer iteration and can pick it
                // up again, while the path-list branch walks a snapshot of `work` and cannot.
                // Either way the document is still in the queue, so nothing is lost - which
                // is why the summary reports the QUEUE as outstanding work and the rejection
                // count merely as events that happened.
                //
                // 🔴 So `chunks_written` is only incremented on a COMMITTED write. Counting
                // the attempt would report chunks that are not in the database, and count
                // them again on the retry.
                if store.replace_chunk_embeddings(id, bge::DIM, &rows, &embedded_of)? {
                    chunks_written += rows.len();
                    certified_in_batch += 1;
                } else {
                    rejected_writes += 1;
                }
                cursor += spans.len();
            }
            done += certified_in_batch;
            // 🔴 `certified_in_batch`, not `docs_in_batch`: the latter is what was ATTEMPTED,
            // and printing it reports documents embedded when none committed.
            //
            // Completion is read from the live queue rather than accumulated. A running
            // total cannot express it: this branch iterates a SNAPSHOT of `work`, and the
            // broad branch re-reads the queue, so one document can be replaced successfully
            // more than once in a run (embedded, edited by an ingest, embedded again) and
            // would be counted twice. `done` below is therefore a count of committed
            // REPLACEMENTS - work performed - and not a position in the corpus.
            // `texts.len()` is what was SENT to the model; the committed rows are the
            // delta in `chunks_written`, which only advances on a committed replacement.
            // A batch of 16 chunks that was entirely rejected must print 0, not 16.
            let committed_chunks = chunks_written - chunks_before_batch;
            // 🔴 Report THIS RUN'S work set, not the corpus queue. This branch embeds an
            // explicit path list; printing the corpus-wide figure told the operator
            // "N-3 of N still queued" on a three-file run - a number this invocation
            // will never reduce, and with `--force` one it does not even move.
            println!(
                "   embedded {} doc(s) / {} chunk row(s) ({} of {} in this path list)",
                certified_in_batch,
                committed_chunks,
                done,
                work.len()
            );
        }

        print_embed_summary(&store, "path-list", done, chunks_written, rejected_writes)?;
        return Ok(());
    }

    if dry_run {
        reply::card(&[(
            "BLOCKING",
            &["--dry-run requires --path-list so broad embedding cannot be simulated as a safe batch"][..],
        )]);
        return Ok(());
    }

    ensure_embedding_benchmark_gate(allow_benchmark_regression)?;
    ensure_embed_disk_space(&root)?;
    ensure_embed_doctor_ready()?;
    // Fill any missing content hashes, AFTER the preflight gates and before the work is
    // done. A row written before these columns existed has `content_sha256 IS NULL`; the
    // write path heals that itself, but doing it here means the queue, the progress counter
    // and the certification agree from the first batch instead of converging over a run.
    //
    // 🔴 Deliberately placed after the disk-space, benchmark and doctor gates: this writes to
    // every unhashed row in the corpus, and a command about to REFUSE to run should not have
    // already rewritten the corpus on its way to refusing.
    //
    // That is a narrower promise than "nothing is mutated before a refusal", and the narrower
    // wording is deliberate: `Store::open` above runs `init_schema`, which migrates the schema
    // - adding columns, and replacing the `documents_au` trigger - before any gate has run. A
    // refused command therefore CAN leave a migrated schema behind. That migration is additive,
    // idempotent and required before any command can address the database at all, so it is not
    // gateable without a second read-only open; the corpus-wide row write below is, and is.
    // Idempotent and cheap - on a hashed corpus it is one SELECT that returns nothing - and it
    // touches neither title nor body, so it fires no FTS work.
    let mut hashed = 0usize;
    loop {
        let n = store.backfill_content_hashes(5_000)?;
        if n == 0 {
            break;
        }
        hashed += n;
    }
    if hashed > 0 {
        eprintln!("recall embed: backfilled {hashed} missing content hashes first");
    }
    // 🔴 NOT `count_chunk_embedded_docs()`. That counts documents with any chunk row,
    // which after the staleness fix includes every STALE document - so it reported the
    // whole corpus as already done and the progress line climbed past the total
    // (`105 / 100`). `already` must mean "needs nothing", which is the corpus
    // minus the queue.
    // ONE statement for both numbers. Read separately, an index committing between them can make
    // the queue exceed the corpus and print a NEGATIVE "already vectorized" - which is the exact
    // failure `embedding_progress` was added to prevent, and it was not being used here.
    let (total, remaining_now) = store.embedding_progress()?;
    let already = total - remaining_now;
    println!("\n{}", reply::paint(reply::BOLD, &format!("recall embed: {already} of {total} docs already vectorized")));

    let embedder = bge::Embedder::new()?;
    let chunk_tokenizer = bge::BgeTokenizer::load()?;
    let mut done = 0usize;
    let mut chunks_written = 0usize;
    // Rolled-back replacements: an ingest rewrote the document mid-embed. This counts
        // REJECTIONS, not distinct documents - one document retried twice counts twice.
    let mut rejected_writes = 0usize;
    loop {
        let work = store.docs_missing_chunk_embeddings(batch)?;
        if work.is_empty() {
            break;
        }
        // Reset PER PASS. `done` is the run total and is declared outside this loop, so printing
        // it under the label "this pass" made pass 2 report pass 1 + pass 2.
        let pass_start_done = done;
        // PER-CHUNK EMBEDDING. Two bugs deep, and this is the second fix.
        //
        // Originally this handed the whole `title || body` string to a model that reads
        // ~512 tokens, so 76.5% of the vault's characters were never embedded and 58.1% of
        // documents were truncated. That was fixed by chunking — but the chunk vectors were
        // then MEAN-POOLED back into one row, because `embeddings.doc_id` is a PRIMARY KEY
        // and physically cannot hold more. Pooling twenty chunks yields a vector that
        // resembles none of them, so a passage in the middle of a long note stayed
        // unretrievable; the chunking had bought accuracy on paper and almost none in
        // practice.
        //
        // Now every chunk keeps its own row in `bge_chunk_embeddings` and the query path
        // scores a document by its BEST chunk (`bge::best_per_doc`), matching what the
        // gemini and voyage channels have always done.
        //
        // The chunker is TOKEN-bounded, using `chunk_tokenizer` — an untruncated second
        // tokenizer instance. Asking the embedder's own tokenizer how long a document is
        // returns 512 for everything over 512, which is the exact number the chunker has to
        // be right about.
        //
        // 🔴 AND THE TENSOR IS BOUNDED BY CHUNK ROWS, NOT BY DOCUMENTS. `-b` counts documents,
        // but ONNX gets one ROW PER CHUNK and chunks-per-document is unbounded, so a
        // document-counted batch has no ceiling on its tensor. Measured 2026-08-20 on the
        // path-list twin: `-b 32` grew the process 6.5 GB -> 23.8 GB of private bytes in twelve
        // minutes and throughput collapsed from 42.4 to ~11 docs/min. ORT's arena grows to the
        // largest allocation it has ever seen and never returns it, so ONE batch that happens
        // to collect long documents raises the floor for the whole run. The attention matrix is
        // the driver: rows x heads x seq x seq, with seq up to 512.
        //
        // So the fetched batch is embedded in chunk-bounded slices. The outer `loop` still
        // re-queries the work queue, which is what makes a kill cost at most one slice.
        let mut slice_start = 0usize;
        while slice_start < work.len() {
            // Same reason as the path-list loop: the embedded body's hash travels with
            // the batch rather than being read back after the embed.
            let mut per_doc: Vec<(i64, Vec<bge::ChunkSpan>, String)> = Vec::new();
            let mut texts: Vec<String> = Vec::new();
            let mut slice_end = slice_start;
            // See the path-list loop: only committed replacements count as progress.
            let mut certified_in_batch = 0usize;
            while slice_end < work.len() {
                let (id, title, body) = &work[slice_end];
                let spans = bge::chunk_for_embedding_spans(title, body, &chunk_tokenizer);
                // Always take at least one document, even if it alone exceeds the cap —
                // otherwise a single very long note stalls the queue forever.
                if !texts.is_empty() && texts.len() + spans.len() > MAX_CHUNK_ROWS_PER_BATCH {
                    break;
                }
                texts.extend(spans.iter().map(|s| s.text.clone()));
                per_doc.push((*id, spans, store::sha256_hex(body.as_bytes())));
                slice_end += 1;
            }
            let vecs = embedder.embed(&texts)?;
            embed_batch_len_ok(vecs.len(), texts.len())?;
            let mut cursor = 0usize;
            for (id, spans, embedded_of) in per_doc {
                // Clear first: `upsert_chunk_embedding` only overwrites the indices it is
                // given, so a document that shrank from 8 chunks to 3 would keep chunks 3..8
                // of its previous body and go on matching text it no longer contains.
                // ONE transaction for the delete, the inserts AND the embedded_sha256
                // stamp. Written as three separate autocommitted statements, a crash
                // between the vectors and the stamp leaves the document looking fresh
                // while carrying vectors built from a body it no longer has - the exact
                // staleness bug, re-created.
                let rows: Vec<(Vec<u8>, store::ChunkPlacement<'_>)> = spans
                    .iter()
                    .zip(vecs[cursor..cursor + spans.len()].iter())
                    .map(|(span, v)| (bge::to_bytes(v), span.into()))
                    .collect();
                // A false return means the row changed under us between the queue read and
                // this write. The ENTIRE replacement is rolled back - the document keeps
                // the vectors it already had - because committing vectors that describe a
                // body the row no longer holds can leave it looking permanently fresh.
                //
                // Whether a rejected document is retried in THIS run depends on the branch:
                // the broad loop re-reads the queue on its outer iteration and can pick it
                // up again, while the path-list branch walks a snapshot of `work` and cannot.
                // Either way the document is still in the queue, so nothing is lost - which
                // is why the summary reports the QUEUE as outstanding work and the rejection
                // count merely as events that happened.
                //
                // 🔴 So `chunks_written` is only incremented on a COMMITTED write. Counting
                // the attempt would report chunks that are not in the database, and count
                // them again on the retry.
                if store.replace_chunk_embeddings(id, bge::DIM, &rows, &embedded_of)? {
                    chunks_written += rows.len();
                    certified_in_batch += 1;
                } else {
                    rejected_writes += 1;
                }
                cursor += spans.len();
            }
            done += certified_in_batch;
            slice_start = slice_end;
        }
        // Same rule as the path-list loop: report what COMMITTED, and read completion from
        // the live queue rather than accumulating. `work.len()` is what was attempted, and
        // `already + done` double-counts any document this run embedded more than once
        // (embedded, edited by an ingest between queue reads, embedded again).
        let (live_now, remaining) = store.embedding_progress()?;
        println!(
            "   embedded {} doc(s) this pass, {} so far ({} of {} still queued)",
            done - pass_start_done,
            done,
            remaining,
            live_now
        );
    }

    print_embed_summary(&store, "new", done, chunks_written, rejected_writes)
}

/// The closing lines of `recall embed`. The rolled-back count is an event count and is said only
/// when it is not zero; "still queued" is what is actually left.
fn print_embed_summary(store: &Store, which: &str, done: usize, chunks_written: usize, rejected_writes: usize) -> Result<()> {
    reply::summary(
        &format!("Vectorized {done} {which} doc(s) into {chunks_written} chunk vector(s) with BGE-small ({}-dim)", bge::DIM),
        &[
            ("docs with vectors", store.count_chunk_embedded_docs()?, true),
            ("still queued", store.count_docs_needing_chunk_embeddings()?, true),
        ],
    );
    if rejected_writes > 0 {
        println!(
            "  note: {rejected_writes} chunk-write(s) rolled back (document rewritten mid-embed); previous vectors kept. \
             This is an event count, not outstanding work - \"still queued\" above is what is actually left\n"
        );
    }
    Ok(())
}

/// Per-document row of the chunk report.
#[derive(serde::Serialize)]
struct ChunkReportDoc {
    doc_id: i64,
    path: String,
    body_bytes: usize,
    chunks: usize,
    summary_chunks: usize,
    with_heading_path: usize,
    max_tokens: usize,
    over_limit: usize,
}

/// Top-level chunk report — the pre-flight verdict for a backfill.
#[derive(serde::Serialize)]
struct ChunkReportJson {
    docs: usize,
    chunks: usize,
    chunk_max_tokens: usize,
    /// The number that has to be zero. A chunk longer than the model's window is silently
    /// truncated at embed time, which is exactly how 25.74% of the corpus was lost once already.
    chunks_over_limit: usize,
    pct_over_limit: f64,
    max_tokens_seen: usize,
    mean_tokens: f64,
    mean_chunks_per_doc: f64,
    docs_with_summary_chunk: usize,
    summary_chunks: usize,
    chunks_with_heading_path: usize,
    pct_chunks_with_heading_path: f64,
    per_doc: Vec<ChunkReportDoc>,
}

/// Chunk a path list and report the statistics, embedding nothing and writing nothing.
///
/// 🔴 Token lengths are measured with [`bge::BgeTokenizer`] — a SECOND tokenizer instance with
/// truncation OFF. The embedder's own tokenizer is capped at 512 and answers "512" for
/// everything longer, so measuring with it would report a clean zero over-limit count no matter
/// how badly the chunker was broken. That is not a hypothetical: a byte-budget bug truncated
/// 25.74% of real chunks while every length check looked fine.
fn cmd_chunk_report(path_list: &PathBuf, limit: &Option<String>, out: Option<&Path>) -> Result<()> {
    // `count` comes from the ChunkTokenizer trait, which must be in scope to be callable —
    // without it `tok.count(..)` resolves to `Iterator::count` and fails to compile.
    use bge::ChunkTokenizer;
    let limit: usize = limit
        .as_deref()
        .unwrap_or("200")
        .parse()
        .unwrap_or(200)
        .max(1);
    let store = Store::open()?;
    let root = store::vault_root();
    let paths = read_path_list(path_list)?;
    let mut work = store.docs_missing_chunk_embeddings_for_paths(&root, &paths)?;
    work.truncate(limit);
    if work.is_empty() {
        reply::card(&[(
            "WHAT I DID",
            &["No path-list docs are missing chunk embeddings; nothing to report"][..],
        )]);
        return Ok(());
    }

    let tok = bge::BgeTokenizer::load()?;
    let mut per_doc = Vec::with_capacity(work.len());
    let mut total_chunks = 0usize;
    let mut total_tokens = 0usize;
    let mut chunks_over = 0usize;
    let mut max_tokens_seen = 0usize;
    let mut summary_chunks = 0usize;
    let mut docs_with_summary = 0usize;
    let mut with_heading = 0usize;

    for (id, title, body, rel) in work.iter() {
        let spans = bge::chunk_for_embedding_spans(title, body, &tok);
        let mut doc_max = 0usize;
        let mut doc_over = 0usize;
        let mut doc_summary = 0usize;
        let mut doc_heading = 0usize;
        for span in spans.iter() {
            let n = tok.count(&span.text);
            total_tokens += n;
            doc_max = doc_max.max(n);
            if n > bge::CHUNK_MAX_TOKENS {
                doc_over += 1;
            }
            if span.is_summary {
                doc_summary += 1;
            }
            if !span.heading_path.is_empty() {
                doc_heading += 1;
            }
        }
        total_chunks += spans.len();
        chunks_over += doc_over;
        max_tokens_seen = max_tokens_seen.max(doc_max);
        summary_chunks += doc_summary;
        with_heading += doc_heading;
        if doc_summary > 0 {
            docs_with_summary += 1;
        }
        per_doc.push(ChunkReportDoc {
            doc_id: *id,
            path: rel.clone(),
            body_bytes: body.len(),
            chunks: spans.len(),
            summary_chunks: doc_summary,
            with_heading_path: doc_heading,
            max_tokens: doc_max,
            over_limit: doc_over,
        });
    }

    let report = ChunkReportJson {
        docs: work.len(),
        chunks: total_chunks,
        chunk_max_tokens: bge::CHUNK_MAX_TOKENS,
        chunks_over_limit: chunks_over,
        pct_over_limit: pct_of(chunks_over, total_chunks),
        max_tokens_seen,
        mean_tokens: mean_of(total_tokens, total_chunks),
        mean_chunks_per_doc: mean_of(total_chunks, work.len()),
        docs_with_summary_chunk: docs_with_summary,
        summary_chunks,
        chunks_with_heading_path: with_heading,
        pct_chunks_with_heading_path: pct_of(with_heading, total_chunks),
        per_doc,
    };

    if let Some(out) = out {
        std::fs::write(out, serde_json::to_string_pretty(&report)?)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// `numerator / denominator` as a percentage, 0.0 when the denominator is zero.
fn pct_of(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        (numerator as f64 / denominator as f64) * 100.0
    }
}

/// `numerator / denominator` as a mean, 0.0 when the denominator is zero.
fn mean_of(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn cmd_dream() -> Result<()> {
    // Operator-only: consolidation retires rows by its own UPDATE, outside the retire ownership
    // rule, so an agent running it could retire another agent's shared memory (Codex tenancy r2).
    operator_only("dream")?;
    let store = Store::open()?;
    let rep = dream::consolidate(&store)?;
    let l1 = format!("RecallDreamer scanned {} observations", rep.scanned);
    let l2 = format!(
        "Superseded {} near-duplicate(s) (bi-temporal, reversible)",
        rep.superseded
    );
    reply::card(&[("WHAT I DID", &[l1.as_str(), l2.as_str()][..])]);
    Ok(())
}

/// Returns the embeddings coverage percentage (0.0–100.0).
/// Returns 0.0 when `docs == 0` to avoid division-by-zero.
fn coverage_pct(embeddings: usize, docs: usize) -> f64 {
    if docs == 0 {
        0.0
    } else {
        (embeddings as f64 / docs as f64) * 100.0
    }
}

/// The two vector lines of `recall status`: query-path chunk coverage, then the legacy table.
fn status_vector_lines(chunk_embedded: i64, legacy: i64, valid: i64) -> [String; 2] {
    [
        format!(
            "   chunk vectors : {} / {}  ({:.2}%)  [query path]",
            chunk_embedded,
            valid,
            coverage_pct(chunk_embedded as usize, valid as usize)
        ),
        format!(
            "   legacy pooled : {} / {}  [legacy embeddings table, not the query path]",
            legacy, valid
        ),
    ]
}

/// Serializable payload for `recall status --json`.
#[derive(Clone, serde::Serialize)]
struct StatusJson {
    docs_valid: i64,
    docs_total: i64,
    superseded: i64,
    /// Valid docs with at least one row in `bge_chunk_embeddings` — what `coverage_pct` and the
    /// `--min-coverage` gate measure.
    chunk_embedded_docs: i64,
    /// Valid docs in the legacy pooled `embeddings` table.
    legacy_embeddings: i64,
    /// Alias of `legacy_embeddings`, kept for one release so existing readers do not break.
    embeddings: i64,
    coverage_pct: f64,
    db_size_mb: f64,
    /// ripgrep runs. When false the rg query channel is off and `index --prune-missing` refuses.
    rg_available: bool,
}

#[derive(serde::Serialize)]
struct EmbedDoctorJson {
    ok: bool,
    timeout_seconds: u64,
    elapsed_ms: Option<u128>,
    ort_version: Option<String>,
    error: Option<String>,
}

#[derive(serde::Serialize)]
struct BgeVectorStatsJson {
    stored_valid_embeddings: i64,
    queryable_embeddings: i64,
    excluded_embeddings: i64,
}

#[derive(serde::Serialize)]
struct HealthDoctorJson {
    ok: bool,
    recall_status: StatusJson,
    bge_vectors: BgeVectorStatsJson,
    embed_doctor: EmbedDoctorJson,
    no_llm: bool,
    no_paid_calls: bool,
}

#[derive(serde::Serialize)]
struct ActiveRetrievalModesJson {
    denoise_mode: String,
    denoise_mode_set: bool,
    fusion_mode: String,
    fusion_mode_set: bool,
    rrf_k: f64,
    rrf_k_set: bool,
    min_fusion_depth: usize,
    min_fusion_depth_set: bool,
    fusion_depth_multiplier: usize,
    fusion_depth_multiplier_set: bool,
}

#[derive(serde::Serialize)]
struct CoverageArtifactJson {
    present: bool,
    run_id: Option<String>,
    summary_json: Option<String>,
    recall_embedding_coverage_pct: Option<f64>,
    recall_unembedded_valid_docs: Option<i64>,
    recall_missing_target_docs: Option<i64>,
    router_missing_target_docs: Option<i64>,
    router_stale_target_docs: Option<i64>,
    router_stale_hard_target_docs: Option<i64>,
    router_stale_live_write_target_docs: Option<i64>,
    secret_risk_target_docs: Option<i64>,
}

#[derive(serde::Serialize)]
struct EmbeddingCoverageFreshnessJson {
    authoritative_source: String,
    artifact_source: String,
    artifact_run_id: Option<String>,
    current_db_unembedded_valid_docs: i64,
    artifact_unembedded_valid_docs: Option<i64>,
    artifact_unembedded_delta_vs_current_db: Option<i64>,
    artifact_matches_current_db: Option<bool>,
}

#[derive(serde::Serialize)]
struct RagasArtifactJson {
    present: bool,
    run_id: Option<String>,
    summary_json: Option<String>,
    passed: Option<bool>,
    pass_count: Option<i64>,
    failed_count: Option<i64>,
    secret_leak_count: Option<i64>,
    recall_at_5: Option<f64>,
    precision_at_5: Option<f64>,
    mrr: Option<f64>,
    ndcg_at_5: Option<f64>,
    latency_p50_ms: Option<f64>,
    latency_p95_ms: Option<f64>,
}

#[derive(serde::Serialize)]
struct BenchArtifactJson {
    present: bool,
    run_id: Option<String>,
    summary_json: Option<String>,
    denoise_mode: Option<String>,
    fusion_mode: Option<String>,
    rrf_k: Option<f64>,
    top_k: Option<i64>,
    positive_fixtures: Option<i64>,
    fts5_hits: Option<i64>,
    bge_hits: Option<i64>,
    fused_hits: Option<i64>,
    fused_recall: Option<f64>,
    latency_p50_ms: Option<f64>,
    latency_p95_ms: Option<f64>,
    raw_contexts_included: Option<bool>,
}

#[derive(serde::Serialize)]
struct BenchHighWaterJson {
    present: bool,
    run_id: Option<String>,
    summary_json: Option<String>,
    fused_hits: Option<i64>,
    fused_recall: Option<f64>,
    denoise_mode: Option<String>,
}

#[derive(serde::Serialize)]
struct VaultOnlyBenchmarkArtifactJson {
    present: bool,
    run_id: Option<String>,
    summary_json: Option<String>,
    source_bench_run_id: Option<String>,
    source_gold_fixture_count: Option<i64>,
    vault_only_fixture_count: Option<i64>,
    excluded_fixture_count: Option<i64>,
    fts5_hits: Option<i64>,
    fts5_recall: Option<f64>,
    bge_hits: Option<i64>,
    bge_recall: Option<f64>,
    fused_hits: Option<i64>,
    fused_recall: Option<f64>,
    fused_mrr: Option<f64>,
    fused_ndcg_at_5: Option<f64>,
    latency_p50_ms: Option<f64>,
    latency_p95_ms: Option<f64>,
    raw_contexts_included: Option<bool>,
    raw_paths_included: Option<bool>,
    fixture_ids_included: Option<bool>,
    secret_values_included: Option<bool>,
    no_llm: Option<bool>,
    no_paid_calls: Option<bool>,
}

#[derive(serde::Serialize)]
struct EmbeddingBenchmarkGateJson {
    present: bool,
    ok: bool,
    status: String,
    blocks_embedding_apply: bool,
    reason: Option<String>,
    latest_run_id: Option<String>,
    high_water_run_id: Option<String>,
    latest_fused_hits: Option<i64>,
    high_water_fused_hits: Option<i64>,
    hit_gap: Option<i64>,
    latest_fused_recall: Option<f64>,
    high_water_fused_recall: Option<f64>,
    latest_raw_contexts_included: Option<bool>,
    no_llm: bool,
    no_paid_calls: bool,
}

#[cfg(feature = "ops-health")]
#[derive(serde::Serialize)]
struct MemoryBrokerDbHealthJson {
    ok: bool,
    source: Option<String>,
    events: Option<i64>,
    chunks: Option<i64>,
    cards: Option<i64>,
    embeddings: Option<i64>,
    pending_embeddings: Option<i64>,
    latest_event_at: Option<String>,
    latest_chunk_at: Option<String>,
    latest_card_at: Option<String>,
    control_plane_http_status: Option<i64>,
    postgres_direct_ok: Option<bool>,
    error_class: Option<String>,
    error: Option<String>,
}

#[cfg(feature = "ops-health")]
#[derive(serde::Serialize)]
struct TailscaleLocalHealthJson {
    ok: bool,
    installed: bool,
    backend_state: Option<String>,
    self_online: Option<bool>,
    peer_count: Option<i64>,
    online_peer_count: Option<i64>,
    has_local_tailscale_ip: Option<bool>,
    error: Option<String>,
}

#[cfg(feature = "ops-health")]
#[derive(serde::Serialize)]
struct GatewayLocalHealthJson {
    ok: bool,
    mcp_port: u16,
    rest_port: u16,
    mcp_code_present: bool,
    rest_code_present: bool,
    mcp_port_listening: bool,
    mcp_status: Option<i64>,
    mcp_auth_required: Option<bool>,
    mcp_health_status: Option<i64>,
    events_status: Option<i64>,
    events_auth_required: Option<bool>,
    sse_bearer_configured: bool,
    rest_port_listening: bool,
    rest_health_status: Option<i64>,
    rest_health_ok: Option<bool>,
    error: Option<String>,
}

#[derive(serde::Serialize)]
struct TargetedPathProbeArtifactJson {
    present: bool,
    run_id: Option<String>,
    summary_json: Option<String>,
    ok: Option<bool>,
    status: Option<String>,
    mode: Option<String>,
    selected_count: Option<i64>,
    router_missing_count: Option<i64>,
    router_stale_count: Option<i64>,
    recall_missing_count: Option<i64>,
    recall_unembedded_count: Option<i64>,
    raw_paths_included: Option<bool>,
    raw_contexts_included: Option<bool>,
    file_contents_read: Option<bool>,
    require_embedded: Option<bool>,
    recall_freshness_verifiable: Option<bool>,
    elapsed_ms: Option<i64>,
}

#[derive(serde::Serialize)]
struct EmbeddingBacklogArtifactJson {
    present: bool,
    summary_json: Option<String>,
    ok: Option<bool>,
    mode: Option<String>,
    total_unembedded: Option<i64>,
    nonrisk_unembedded: Option<i64>,
    path_risk_unembedded: Option<i64>,
    router_eligible_pct: Option<f64>,
    nonrisk_pct: Option<f64>,
    path_risk_pct: Option<f64>,
    selected_pct_of_total: Option<f64>,
    selected_pct_of_router_eligible: Option<f64>,
    requested_size: Option<i64>,
    selected_count: Option<i64>,
    max_per_prefix: Option<i64>,
    raw_paths_included: Option<bool>,
    raw_contexts_included: Option<bool>,
    file_contents_read: Option<bool>,
    path_list_contains_raw_paths: Option<bool>,
    current_filter_enabled: Option<bool>,
    current_unembedded_paths: Option<i64>,
    stale_artifact_rows_removed: Option<i64>,
}

#[derive(serde::Serialize)]
struct MemoryDoctorGatesJson {
    recall_store_present: bool,
    recall_index_has_docs: bool,
    recall_embeddings_complete: bool,
    coverage_artifact_present: bool,
    coverage_delta_refresh_present: bool,
    coverage_delta_refresh_redacted: bool,
    ragas_smoke_green: bool,
    secret_leak_rate_zero: bool,
    release_benchmark_present: bool,
    release_mode_sweep_present: bool,
    release_mode_sweep_not_blocked: bool,
    latest_bench_no_raw_contexts: bool,
    vault_only_benchmark_present: bool,
    vault_only_benchmark_redacted: bool,
    embedding_benchmark_gate_present: bool,
    embedding_benchmark_gate_passed: bool,
    embed_disk_space_gate_present: bool,
    embed_disk_space_gate_passed: bool,
    #[cfg(feature = "ops-health")]
    memory_broker_db_proven: bool,
    #[cfg(feature = "ops-health")]
    tailscale_local_ready: bool,
    #[cfg(feature = "ops-health")]
    gateway_local_ready: bool,
    targeted_path_probe_present: bool,
    targeted_path_probe_redacted: bool,
    embedding_backlog_present: bool,
    embedding_backlog_redacted: bool,
}

#[derive(serde::Serialize)]
struct MemoryDoctorJson {
    ok: bool,
    status: String,
    recall_status: StatusJson,
    active_retrieval_modes: ActiveRetrievalModesJson,
    coverage_artifact: CoverageArtifactJson,
    coverage_delta_refresh: CoverageDeltaRefreshArtifactJson,
    embedding_coverage_freshness: EmbeddingCoverageFreshnessJson,
    ragas_smoke: RagasArtifactJson,
    latest_release_benchmark: BenchArtifactJson,
    release_benchmark_high_water: BenchHighWaterJson,
    release_mode_sweep: ReleaseModeSweepArtifactJson,
    vault_only_benchmark: VaultOnlyBenchmarkArtifactJson,
    embedding_benchmark_gate: EmbeddingBenchmarkGateJson,
    embed_disk_space_gate: EmbedDiskSpaceGateJson,
    #[cfg(feature = "ops-health")]
    memory_broker_db_health: MemoryBrokerDbHealthJson,
    #[cfg(feature = "ops-health")]
    tailscale_local_health: TailscaleLocalHealthJson,
    #[cfg(feature = "ops-health")]
    gateway_local_health: GatewayLocalHealthJson,
    targeted_path_probe: TargetedPathProbeArtifactJson,
    embedding_backlog: EmbeddingBacklogArtifactJson,
    gates: MemoryDoctorGatesJson,
    raw_contexts_included: bool,
    no_llm: bool,
    no_paid_calls: bool,
}

#[derive(serde::Serialize)]
struct CoverageDeltaRefreshArtifactJson {
    present: bool,
    run_id: Option<String>,
    summary_json: Option<String>,
    status: Option<String>,
    ok: Option<bool>,
    baseline_coverage_run_id: Option<String>,
    full_audit_skipped: Option<bool>,
    full_audit_skip_reason: Option<String>,
    disk_free_gb: Option<f64>,
    min_full_audit_free_gb: Option<f64>,
    router_indexer_target_coverage_pct: Option<f64>,
    recall_indexer_target_coverage_pct: Option<f64>,
    recall_valid_docs: Option<i64>,
    recall_embeddings: Option<i64>,
    recall_unembedded_valid_docs: Option<i64>,
    recall_duplicate_valid_paths: Option<i64>,
    remaining_recall_missing: Option<i64>,
    remaining_router_missing: Option<i64>,
    remaining_router_stale: Option<i64>,
    remaining_router_hard_stale: Option<i64>,
    remaining_router_live_write_stale: Option<i64>,
    repaired_recall_missing: Option<i64>,
    repaired_router_missing: Option<i64>,
    repaired_router_stale: Option<i64>,
    redacted_surrogate_enabled: Option<bool>,
    redacted_surrogate_manifest_count: Option<i64>,
    redacted_surrogate_unsafe_manifest_count: Option<i64>,
    redacted_surrogate_written_count: Option<i64>,
    redacted_surrogate_pattern_hits_redacted: Option<i64>,
    router_missing_docs_with_redacted_surrogate: Option<i64>,
    recall_missing_docs_with_redacted_surrogate: Option<i64>,
    router_missing_target_docs_without_redacted_surrogate: Option<i64>,
    recall_missing_target_docs_without_redacted_surrogate: Option<i64>,
    raw_contexts_included: Option<bool>,
    raw_paths_included: Option<bool>,
    raw_secret_values_included: Option<bool>,
    file_contents_read: Option<bool>,
    no_llm: Option<bool>,
    no_paid_calls: Option<bool>,
    database_mutation_expected: Option<bool>,
}

#[derive(serde::Serialize)]
struct EmbedDiskSpaceGateJson {
    present: bool,
    ok: bool,
    status: String,
    blocks_embedding_apply: bool,
    reason: Option<String>,
    free_bytes: Option<u64>,
    free_gb: Option<f64>,
    min_required_free_bytes: u64,
    min_required_free_gb: f64,
    no_env_file_read: bool,
    no_llm: bool,
    no_paid_calls: bool,
}

#[derive(serde::Serialize)]
struct ReleaseModeSweepArtifactJson {
    present: bool,
    run_id: Option<String>,
    summary_json: Option<String>,
    status: Option<String>,
    blocker: Option<String>,
    reason: Option<String>,
    variant_count: Option<i64>,
    baseline_fused_hits: Option<i64>,
    best_fused_hits: Option<i64>,
    high_water_fused_hits: Option<i64>,
    disk_free_gb: Option<f64>,
    min_required_free_gb: Option<f64>,
    raw_contexts_included: Option<bool>,
    no_llm: Option<bool>,
    no_paid_calls: Option<bool>,
    no_env_file_read: Option<bool>,
    database_mutation_expected: Option<bool>,
}

/// Pure threshold predicate for the `--min-coverage` CI gate.
/// Coverage exactly at the threshold passes; strictly below fails.
fn coverage_meets(pct: f64, threshold: f64) -> bool {
    pct >= threshold
}

fn status_json_payload_from_store(store: &Store) -> Result<StatusJson> {
    let s = store.stats()?;
    Ok(status_json_payload(&s))
}

/// `status --json` from already-computed stats. `coverage_pct` (and so the `--min-coverage`
/// gate) measures chunk vectors — the table the query path reads — not the legacy table.
fn status_json_payload(s: &store::Stats) -> StatusJson {
    StatusJson {
        docs_valid: s.valid,
        docs_total: s.total,
        superseded: s.superseded,
        chunk_embedded_docs: s.chunk_embedded,
        legacy_embeddings: s.embeddings,
        embeddings: s.embeddings,
        coverage_pct: coverage_pct(s.chunk_embedded as usize, s.valid as usize),
        db_size_mb: s.db_bytes as f64 / 1_048_576.0,
        rg_available: store::rg_available(),
    }
}

fn json_string(value: &serde_json::Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(ToString::to_string)
}

fn json_i64(value: &serde_json::Value, key: &str) -> Option<i64> {
    value.get(key)?.as_i64()
}

fn json_f64(value: &serde_json::Value, key: &str) -> Option<f64> {
    value.get(key)?.as_f64()
}

fn json_bool(value: &serde_json::Value, key: &str) -> Option<bool> {
    value.get(key)?.as_bool()
}

fn json_path<'a>(value: &'a serde_json::Value, keys: &[&str]) -> Option<&'a serde_json::Value> {
    let mut cursor = value;
    for key in keys {
        cursor = cursor.get(*key)?;
    }
    Some(cursor)
}

fn json_path_i64(value: &serde_json::Value, keys: &[&str]) -> Option<i64> {
    json_path(value, keys)?.as_i64()
}

fn json_path_f64(value: &serde_json::Value, keys: &[&str]) -> Option<f64> {
    json_path(value, keys)?.as_f64()
}

fn json_path_bool(value: &serde_json::Value, keys: &[&str]) -> Option<bool> {
    json_path(value, keys)?.as_bool()
}

fn read_json_file(path: &PathBuf) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn latest_nested_summary(root: PathBuf, filename: &str) -> Option<(PathBuf, serde_json::Value)> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    let direct_summary = root.join(filename);
    if direct_summary.exists() {
        candidates.push(direct_summary);
    }
    collect_nested_summary_paths(&root, filename, &mut candidates);
    candidates.sort_by_key(|path| {
        path.metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });
    candidates.reverse();
    for path in candidates {
        if let Some(value) = read_json_file(&path) {
            return Some((path, value));
        }
    }
    None
}

fn collect_nested_summary_paths(root: &Path, filename: &str, candidates: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.filter_map(|entry| entry.ok()) {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let summary = path.join(filename);
        if summary.exists() {
            candidates.push(summary);
        }
        collect_nested_summary_paths(&path, filename, candidates);
    }
}

fn latest_nested_summary_by_mtime(
    root: PathBuf,
    filename: &str,
) -> Option<(PathBuf, serde_json::Value)> {
    let direct_summary = root.join(filename);
    let mut candidates: Vec<PathBuf> = Vec::new();
    if direct_summary.exists() {
        candidates.push(direct_summary);
    }
    candidates.extend(
        std::fs::read_dir(root)
            .ok()?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_dir())
            .map(|path| path.join(filename))
            .filter(|path| path.exists()),
    );
    candidates.sort_by_key(|path| {
        path.metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });
    candidates.reverse();
    for path in candidates {
        if let Some(value) = read_json_file(&path) {
            return Some((path, value));
        }
    }
    None
}

fn latest_nested_summary_across_roots_by_mtime(
    roots: Vec<PathBuf>,
    filename: &str,
) -> Option<(PathBuf, serde_json::Value)> {
    let mut candidates = Vec::new();
    for root in roots {
        let direct_summary = root.join(filename);
        if direct_summary.exists() {
            candidates.push(direct_summary);
        }
        let Ok(entries) = std::fs::read_dir(root) else {
            continue;
        };
        candidates.extend(
            entries
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| path.is_dir())
                .map(|path| path.join(filename))
                .filter(|path| path.exists()),
        );
    }
    candidates.sort_by_key(|path| {
        path.metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });
    candidates.reverse();
    for path in candidates {
        if let Some(value) = read_json_file(&path) {
            return Some((path, value));
        }
    }
    None
}

fn coverage_audit_roots(results_root: &std::path::Path) -> Vec<PathBuf> {
    vec![
        results_root.join("coverage_audit"),
    ]
}

fn latest_targeted_path_probe_artifact(root: PathBuf) -> Option<(PathBuf, serde_json::Value)> {
    ["coverage_path_probe.json", "router_stale_probe.json"]
        .iter()
        .filter_map(|filename| latest_nested_summary_by_mtime(root.clone(), filename))
        .max_by_key(|(path, _)| {
            path.metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        })
}

fn is_bench_report_path(path: &PathBuf) -> bool {
    path.is_file()
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| {
                name.strip_prefix("bench_")
                    .and_then(|rest| rest.strip_suffix(".json"))
                    .map(|run| run.chars().all(|ch| ch.is_ascii_digit()))
                    .unwrap_or(false)
            })
            .unwrap_or(false)
}

fn latest_bench_report(root: PathBuf) -> Option<(PathBuf, serde_json::Value)> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(is_bench_report_path)
        .collect();
    candidates.sort_by_key(|path| {
        std::fs::metadata(path)
            .and_then(|meta| meta.modified())
            .ok()
    });
    candidates.reverse();
    for path in candidates {
        if let Some(value) = read_json_file(&path) {
            return Some((path, value));
        }
    }
    None
}

fn bench_high_water(root: PathBuf) -> BenchHighWaterJson {
    let Some(entries) = std::fs::read_dir(root).ok() else {
        return BenchHighWaterJson {
            present: false,
            run_id: None,
            summary_json: None,
            fused_hits: None,
            fused_recall: None,
            denoise_mode: None,
        };
    };
    let mut best: Option<(
        PathBuf,
        serde_json::Value,
        i64,
        Option<std::time::SystemTime>,
    )> = None;
    for path in entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(is_bench_report_path)
    {
        let Some(value) = read_json_file(&path) else {
            continue;
        };
        let Some(fused_hits) = json_i64(&value, "fused_hits") else {
            continue;
        };
        let modified = std::fs::metadata(&path)
            .and_then(|meta| meta.modified())
            .ok();
        let replace = best
            .as_ref()
            .map(|(_, _, current, current_modified)| {
                fused_hits > *current || (fused_hits == *current && modified > *current_modified)
            })
            .unwrap_or(true);
        if replace {
            best = Some((path, value, fused_hits, modified));
        }
    }
    let Some((path, value, fused_hits, _)) = best else {
        return BenchHighWaterJson {
            present: false,
            run_id: None,
            summary_json: None,
            fused_hits: None,
            fused_recall: None,
            denoise_mode: None,
        };
    };
    BenchHighWaterJson {
        present: true,
        run_id: json_string(&value, "run_id"),
        summary_json: Some(path.display().to_string()),
        fused_hits: Some(fused_hits),
        fused_recall: json_f64(&value, "fused_recall"),
        denoise_mode: json_string(&value, "denoise_mode"),
    }
}

fn embedding_benchmark_gate_decision(
    latest: &BenchArtifactJson,
    high_water: &BenchHighWaterJson,
) -> std::result::Result<(), String> {
    if !latest.present {
        return Err("no latest release benchmark artifact found".to_string());
    }
    if !high_water.present {
        return Err("no release benchmark high-water artifact found".to_string());
    }
    if latest.raw_contexts_included != Some(false) {
        return Err("latest benchmark artifact is not redacted".to_string());
    }
    let latest_hits = latest
        .fused_hits
        .ok_or_else(|| "latest benchmark is missing fused_hits".to_string())?;
    let high_hits = high_water
        .fused_hits
        .ok_or_else(|| "high-water benchmark is missing fused_hits".to_string())?;
    if latest_hits < high_hits {
        return Err(format!(
            "latest fused recall is below high-water: {} < {} (latest={}, high_water={})",
            latest_hits,
            high_hits,
            latest.run_id.as_deref().unwrap_or("unknown"),
            high_water.run_id.as_deref().unwrap_or("unknown")
        ));
    }
    Ok(())
}

fn embedding_benchmark_gate_from(
    latest: &BenchArtifactJson,
    high_water: &BenchHighWaterJson,
) -> EmbeddingBenchmarkGateJson {
    let decision = embedding_benchmark_gate_decision(latest, high_water);
    let reason = decision.as_ref().err().cloned();
    let hit_gap = match (latest.fused_hits, high_water.fused_hits) {
        (Some(latest_hits), Some(high_hits)) => Some(latest_hits - high_hits),
        _ => None,
    };
    let present = latest.present && high_water.present;
    let ok = decision.is_ok();
    EmbeddingBenchmarkGateJson {
        present,
        ok,
        status: if ok { "pass" } else { "blocking" }.to_string(),
        blocks_embedding_apply: !ok,
        reason,
        latest_run_id: latest.run_id.clone(),
        high_water_run_id: high_water.run_id.clone(),
        latest_fused_hits: latest.fused_hits,
        high_water_fused_hits: high_water.fused_hits,
        hit_gap,
        latest_fused_recall: latest.fused_recall,
        high_water_fused_recall: high_water.fused_recall,
        latest_raw_contexts_included: latest.raw_contexts_included,
        no_llm: true,
        no_paid_calls: true,
    }
}

fn release_sweep_benchmark_evidence(
    sweep: &ReleaseModeSweepArtifactJson,
) -> Option<BenchArtifactJson> {
    if !sweep.present {
        return None;
    }
    if sweep.status.as_deref() == Some("blocked_environment") {
        return None;
    }
    if sweep.raw_contexts_included != Some(false)
        || sweep.no_llm != Some(true)
        || sweep.no_paid_calls != Some(true)
        || sweep.database_mutation_expected != Some(false)
    {
        return None;
    }
    let best_hits = sweep.best_fused_hits?;
    let high_hits = sweep.high_water_fused_hits?;
    if best_hits < high_hits {
        return None;
    }
    Some(BenchArtifactJson {
        present: true,
        run_id: sweep
            .run_id
            .as_ref()
            .map(|run_id| format!("release_sweep:{run_id}")),
        summary_json: sweep.summary_json.clone(),
        denoise_mode: None,
        fusion_mode: None,
        rrf_k: None,
        top_k: None,
        positive_fixtures: None,
        fts5_hits: None,
        bge_hits: None,
        fused_hits: Some(best_hits),
        fused_recall: None,
        latency_p50_ms: None,
        latency_p95_ms: None,
        raw_contexts_included: Some(false),
    })
}

#[cfg(test)]
fn embedding_benchmark_gate_from_sweep(
    latest: &BenchArtifactJson,
    high_water: &BenchHighWaterJson,
    sweep: &ReleaseModeSweepArtifactJson,
) -> EmbeddingBenchmarkGateJson {
    if let Some(evidence) = release_sweep_benchmark_evidence(sweep) {
        return embedding_benchmark_gate_from(&evidence, high_water);
    }
    embedding_benchmark_gate_from(latest, high_water)
}

fn latest_release_sweep_benchmark_evidence(root: PathBuf) -> Option<BenchArtifactJson> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(root)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .map(|path| path.join("release_mode_sweep_summary.json"))
        .filter(|path| path.exists())
        .collect();
    candidates.sort_by_key(|path| {
        path.metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });
    candidates.reverse();
    for path in candidates {
        let Some(value) = read_json_file(&path) else {
            continue;
        };
        let sweep = release_mode_sweep_artifact_from(Some((path, value)));
        if let Some(evidence) = release_sweep_benchmark_evidence(&sweep) {
            return Some(evidence);
        }
    }
    None
}

fn ensure_embedding_benchmark_gate(allow_regression: bool) -> Result<()> {
    if allow_regression {
        return Ok(());
    }
    // The gate compares against benchmark artifacts in recall's development folder. Without one
    // (any install but the developer's) there is nothing to compare, so there is no gate.
    if store::recall_dev_dir().is_none() {
        return Ok(());
    }
    let results_root = recall_dev_dir()?
        .join("eval")
        .join("results");
    let bench_root = recall_dev_dir()?
        .join("eval")
        .join("results")
        .join("bench_release");
    let latest = bench_artifact_from(latest_bench_report(bench_root.clone()));
    let high_water = bench_high_water(bench_root);
    let evidence = latest_release_sweep_benchmark_evidence(results_root.join("release_mode_sweep"))
        .unwrap_or(latest);
    if let Err(reason) = embedding_benchmark_gate_decision(&evidence, &high_water) {
        reply::card(&[
            (
                "BLOCKING",
                &[
                    "Embedding benchmark gate failed; recall.db was not mutated.",
                    reason.as_str(),
                    "Run a broad benchmark that meets or beats high-water, or pass --allow-benchmark-regression for an explicit override.",
                ][..],
            ),
        ]);
        anyhow::bail!("embedding benchmark gate failed: {reason}");
    }
    Ok(())
}

fn coverage_artifact_from(value: Option<(PathBuf, serde_json::Value)>) -> CoverageArtifactJson {
    let Some((path, value)) = value else {
        return CoverageArtifactJson {
            present: false,
            run_id: None,
            summary_json: None,
            recall_embedding_coverage_pct: None,
            recall_unembedded_valid_docs: None,
            recall_missing_target_docs: None,
            router_missing_target_docs: None,
            router_stale_target_docs: None,
            router_stale_hard_target_docs: None,
            router_stale_live_write_target_docs: None,
            secret_risk_target_docs: None,
        };
    };
    let metrics = value
        .get("metrics")
        .filter(|value| value.is_object())
        .unwrap_or(&serde_json::Value::Null);
    CoverageArtifactJson {
        present: true,
        run_id: json_string(&value, "run_id"),
        summary_json: Some(path.display().to_string()),
        recall_embedding_coverage_pct: json_f64(metrics, "recall_embedding_coverage_pct"),
        recall_unembedded_valid_docs: json_i64(metrics, "recall_unembedded_valid_docs"),
        recall_missing_target_docs: json_i64(metrics, "recall_missing_target_docs"),
        router_missing_target_docs: json_i64(metrics, "router_missing_target_docs"),
        router_stale_target_docs: json_i64(metrics, "router_stale_target_docs"),
        router_stale_hard_target_docs: json_i64(metrics, "router_stale_hard_target_docs"),
        router_stale_live_write_target_docs: json_i64(
            metrics,
            "router_stale_live_write_target_docs",
        ),
        secret_risk_target_docs: json_i64(metrics, "secret_risk_target_docs"),
    }
}

fn ragas_artifact_from(value: Option<(PathBuf, serde_json::Value)>) -> RagasArtifactJson {
    let Some((path, value)) = value else {
        return RagasArtifactJson {
            present: false,
            run_id: None,
            summary_json: None,
            passed: None,
            pass_count: None,
            failed_count: None,
            secret_leak_count: None,
            recall_at_5: None,
            precision_at_5: None,
            mrr: None,
            ndcg_at_5: None,
            latency_p50_ms: None,
            latency_p95_ms: None,
        };
    };
    let metrics = value
        .get("metrics")
        .filter(|value| value.is_object())
        .unwrap_or(&serde_json::Value::Null);
    RagasArtifactJson {
        present: true,
        run_id: json_string(&value, "run_id"),
        summary_json: Some(path.display().to_string()),
        passed: json_bool(&value, "passed"),
        pass_count: json_i64(&value, "pass_count"),
        failed_count: json_i64(&value, "failed_count"),
        secret_leak_count: json_i64(&value, "secret_leak_count"),
        recall_at_5: json_f64(metrics, "recall_at_5"),
        precision_at_5: json_f64(metrics, "precision_at_5"),
        mrr: json_f64(metrics, "mrr"),
        ndcg_at_5: json_f64(metrics, "ndcg_at_5"),
        latency_p50_ms: json_f64(metrics, "latency_p50_ms"),
        latency_p95_ms: json_f64(metrics, "latency_p95_ms"),
    }
}

fn bench_artifact_from(value: Option<(PathBuf, serde_json::Value)>) -> BenchArtifactJson {
    let Some((path, value)) = value else {
        return BenchArtifactJson {
            present: false,
            run_id: None,
            summary_json: None,
            denoise_mode: None,
            fusion_mode: None,
            rrf_k: None,
            top_k: None,
            positive_fixtures: None,
            fts5_hits: None,
            bge_hits: None,
            fused_hits: None,
            fused_recall: None,
            latency_p50_ms: None,
            latency_p95_ms: None,
            raw_contexts_included: None,
        };
    };
    BenchArtifactJson {
        present: true,
        run_id: json_string(&value, "run_id"),
        summary_json: Some(path.display().to_string()),
        denoise_mode: json_string(&value, "denoise_mode"),
        fusion_mode: json_string(&value, "fusion_mode"),
        rrf_k: json_f64(&value, "rrf_k"),
        top_k: json_i64(&value, "top_k"),
        positive_fixtures: json_i64(&value, "positive_fixtures"),
        fts5_hits: json_i64(&value, "fts5_hits"),
        bge_hits: json_i64(&value, "bge_hits"),
        fused_hits: json_i64(&value, "fused_hits"),
        fused_recall: json_f64(&value, "fused_recall"),
        latency_p50_ms: json_f64(&value, "latency_p50_ms"),
        latency_p95_ms: json_f64(&value, "latency_p95_ms"),
        raw_contexts_included: json_bool(&value, "raw_contexts_included"),
    }
}

fn vault_only_benchmark_artifact_from(
    value: Option<(PathBuf, serde_json::Value)>,
) -> VaultOnlyBenchmarkArtifactJson {
    let Some((path, value)) = value else {
        return VaultOnlyBenchmarkArtifactJson {
            present: false,
            run_id: None,
            summary_json: None,
            source_bench_run_id: None,
            source_gold_fixture_count: None,
            vault_only_fixture_count: None,
            excluded_fixture_count: None,
            fts5_hits: None,
            fts5_recall: None,
            bge_hits: None,
            bge_recall: None,
            fused_hits: None,
            fused_recall: None,
            fused_mrr: None,
            fused_ndcg_at_5: None,
            latency_p50_ms: None,
            latency_p95_ms: None,
            raw_contexts_included: None,
            raw_paths_included: None,
            fixture_ids_included: None,
            secret_values_included: None,
            no_llm: None,
            no_paid_calls: None,
        };
    };
    let fts5 = value
        .get("fts5")
        .filter(|value| value.is_object())
        .unwrap_or(&serde_json::Value::Null);
    let bge = value
        .get("bge")
        .filter(|value| value.is_object())
        .unwrap_or(&serde_json::Value::Null);
    let fused = value
        .get("fused")
        .filter(|value| value.is_object())
        .unwrap_or(&serde_json::Value::Null);
    VaultOnlyBenchmarkArtifactJson {
        present: true,
        run_id: json_string(&value, "run_id"),
        summary_json: Some(path.display().to_string()),
        source_bench_run_id: json_string(&value, "source_bench_run_id"),
        source_gold_fixture_count: json_i64(&value, "source_gold_fixture_count"),
        vault_only_fixture_count: json_i64(&value, "vault_only_fixture_count"),
        excluded_fixture_count: json_i64(&value, "excluded_fixture_count"),
        fts5_hits: json_i64(fts5, "hits_at_5"),
        fts5_recall: json_f64(fts5, "recall_at_5"),
        bge_hits: json_i64(bge, "hits_at_5"),
        bge_recall: json_f64(bge, "recall_at_5"),
        fused_hits: json_i64(fused, "hits_at_5"),
        fused_recall: json_f64(fused, "recall_at_5"),
        fused_mrr: json_f64(fused, "mrr"),
        fused_ndcg_at_5: json_f64(fused, "ndcg_at_5"),
        latency_p50_ms: json_f64(&value, "latency_p50_ms"),
        latency_p95_ms: json_f64(&value, "latency_p95_ms"),
        raw_contexts_included: json_bool(&value, "raw_contexts_included"),
        raw_paths_included: json_bool(&value, "raw_paths_included"),
        fixture_ids_included: json_bool(&value, "fixture_ids_included"),
        secret_values_included: json_bool(&value, "secret_values_included"),
        no_llm: json_bool(&value, "no_llm"),
        no_paid_calls: json_bool(&value, "no_paid_calls"),
    }
}

fn release_mode_sweep_artifact_from(
    value: Option<(PathBuf, serde_json::Value)>,
) -> ReleaseModeSweepArtifactJson {
    let Some((path, value)) = value else {
        return ReleaseModeSweepArtifactJson {
            present: false,
            run_id: None,
            summary_json: None,
            status: None,
            blocker: None,
            reason: None,
            variant_count: None,
            baseline_fused_hits: None,
            best_fused_hits: None,
            high_water_fused_hits: None,
            disk_free_gb: None,
            min_required_free_gb: None,
            raw_contexts_included: None,
            no_llm: None,
            no_paid_calls: None,
            no_env_file_read: None,
            database_mutation_expected: None,
        };
    };
    let high_water = value
        .get("high_water")
        .filter(|value| value.is_object())
        .unwrap_or(&serde_json::Value::Null);
    ReleaseModeSweepArtifactJson {
        present: true,
        run_id: json_string(&value, "run_id"),
        summary_json: Some(path.display().to_string()),
        status: json_string(&value, "status"),
        blocker: json_string(&value, "blocker"),
        reason: json_string(&value, "reason"),
        variant_count: json_i64(&value, "variant_count"),
        baseline_fused_hits: json_i64(&value, "baseline_fused_hits"),
        best_fused_hits: json_i64(&value, "best_fused_hits"),
        high_water_fused_hits: json_i64(high_water, "fused_hits"),
        disk_free_gb: json_f64(&value, "disk_free_gb"),
        min_required_free_gb: json_f64(&value, "min_required_free_gb"),
        raw_contexts_included: json_bool(&value, "raw_contexts_included"),
        no_llm: json_bool(&value, "no_llm"),
        no_paid_calls: json_bool(&value, "no_paid_calls"),
        no_env_file_read: json_bool(&value, "no_env_file_read"),
        database_mutation_expected: json_bool(&value, "database_mutation_expected"),
    }
}

fn coverage_delta_refresh_artifact_from(
    value: Option<(PathBuf, serde_json::Value)>,
) -> CoverageDeltaRefreshArtifactJson {
    let Some((path, value)) = value else {
        return CoverageDeltaRefreshArtifactJson {
            present: false,
            run_id: None,
            summary_json: None,
            status: None,
            ok: None,
            baseline_coverage_run_id: None,
            full_audit_skipped: None,
            full_audit_skip_reason: None,
            disk_free_gb: None,
            min_full_audit_free_gb: None,
            router_indexer_target_coverage_pct: None,
            recall_indexer_target_coverage_pct: None,
            recall_valid_docs: None,
            recall_embeddings: None,
            recall_unembedded_valid_docs: None,
            recall_duplicate_valid_paths: None,
            remaining_recall_missing: None,
            remaining_router_missing: None,
            remaining_router_stale: None,
            remaining_router_hard_stale: None,
            remaining_router_live_write_stale: None,
            repaired_recall_missing: None,
            repaired_router_missing: None,
            repaired_router_stale: None,
            redacted_surrogate_enabled: None,
            redacted_surrogate_manifest_count: None,
            redacted_surrogate_unsafe_manifest_count: None,
            redacted_surrogate_written_count: None,
            redacted_surrogate_pattern_hits_redacted: None,
            router_missing_docs_with_redacted_surrogate: None,
            recall_missing_docs_with_redacted_surrogate: None,
            router_missing_target_docs_without_redacted_surrogate: None,
            recall_missing_target_docs_without_redacted_surrogate: None,
            raw_contexts_included: None,
            raw_paths_included: None,
            raw_secret_values_included: None,
            file_contents_read: None,
            no_llm: None,
            no_paid_calls: None,
            database_mutation_expected: None,
        };
    };
    CoverageDeltaRefreshArtifactJson {
        present: true,
        run_id: json_string(&value, "run_id"),
        summary_json: Some(path.display().to_string()),
        status: json_string(&value, "status"),
        ok: json_bool(&value, "ok"),
        baseline_coverage_run_id: json_string(&value, "baseline_coverage_run_id"),
        full_audit_skipped: json_bool(&value, "full_audit_skipped"),
        full_audit_skip_reason: json_string(&value, "full_audit_skip_reason"),
        disk_free_gb: json_f64(&value, "disk_free_gb"),
        min_full_audit_free_gb: json_f64(&value, "min_full_audit_free_gb"),
        router_indexer_target_coverage_pct: json_path_f64(
            &value,
            &[
                "coverage_estimate_on_baseline_inventory",
                "router_indexer_target_coverage_pct",
            ],
        ),
        recall_indexer_target_coverage_pct: json_path_f64(
            &value,
            &[
                "coverage_estimate_on_baseline_inventory",
                "recall_indexer_target_coverage_pct",
            ],
        ),
        recall_valid_docs: json_path_i64(
            &value,
            &[
                "coverage_estimate_on_baseline_inventory",
                "recall_valid_docs",
            ],
        ),
        recall_embeddings: json_path_i64(
            &value,
            &[
                "coverage_estimate_on_baseline_inventory",
                "recall_embedded_valid_docs",
            ],
        ),
        recall_unembedded_valid_docs: json_path_i64(
            &value,
            &[
                "coverage_estimate_on_baseline_inventory",
                "recall_unembedded_valid_docs",
            ],
        ),
        recall_duplicate_valid_paths: json_path_i64(
            &value,
            &[
                "coverage_estimate_on_baseline_inventory",
                "recall_duplicate_valid_paths",
            ],
        ),
        remaining_recall_missing: json_path_i64(
            &value,
            &["known_gap_counts", "remaining_recall_missing"],
        ),
        remaining_router_missing: json_path_i64(
            &value,
            &["known_gap_counts", "remaining_router_missing"],
        ),
        remaining_router_stale: json_path_i64(
            &value,
            &["known_gap_counts", "remaining_router_stale"],
        ),
        remaining_router_hard_stale: json_path_i64(
            &value,
            &["known_gap_counts", "remaining_router_hard_stale"],
        ),
        remaining_router_live_write_stale: json_path_i64(
            &value,
            &["known_gap_counts", "remaining_router_live_write_stale"],
        ),
        repaired_recall_missing: json_path_i64(
            &value,
            &["known_gap_counts", "repaired_recall_missing"],
        ),
        repaired_router_missing: json_path_i64(
            &value,
            &["known_gap_counts", "repaired_router_missing"],
        ),
        repaired_router_stale: json_path_i64(
            &value,
            &["known_gap_counts", "repaired_router_stale"],
        ),
        redacted_surrogate_enabled: json_path_bool(
            &value,
            &["redacted_surrogate_coverage", "enabled"],
        ),
        redacted_surrogate_manifest_count: json_path_i64(
            &value,
            &["redacted_surrogate_coverage", "manifest_count"],
        ),
        redacted_surrogate_unsafe_manifest_count: json_path_i64(
            &value,
            &["redacted_surrogate_coverage", "unsafe_manifest_count"],
        ),
        redacted_surrogate_written_count: json_path_i64(
            &value,
            &["redacted_surrogate_coverage", "surrogate_written_count"],
        ),
        redacted_surrogate_pattern_hits_redacted: json_path_i64(
            &value,
            &["redacted_surrogate_coverage", "total_pattern_hits_redacted"],
        ),
        router_missing_docs_with_redacted_surrogate: json_path_i64(
            &value,
            &[
                "redacted_surrogate_coverage",
                "router_missing_docs_with_redacted_surrogate",
            ],
        ),
        recall_missing_docs_with_redacted_surrogate: json_path_i64(
            &value,
            &[
                "redacted_surrogate_coverage",
                "recall_missing_docs_with_redacted_surrogate",
            ],
        ),
        router_missing_target_docs_without_redacted_surrogate: json_path_i64(
            &value,
            &[
                "redacted_surrogate_coverage",
                "router_missing_target_docs_without_redacted_surrogate",
            ],
        ),
        recall_missing_target_docs_without_redacted_surrogate: json_path_i64(
            &value,
            &[
                "redacted_surrogate_coverage",
                "recall_missing_target_docs_without_redacted_surrogate",
            ],
        ),
        raw_contexts_included: json_bool(&value, "raw_contexts_included"),
        raw_paths_included: json_bool(&value, "raw_paths_included"),
        raw_secret_values_included: json_path_bool(
            &value,
            &["redacted_surrogate_coverage", "raw_secret_values_included"],
        ),
        file_contents_read: json_bool(&value, "file_contents_read"),
        no_llm: json_bool(&value, "no_llm"),
        no_paid_calls: json_bool(&value, "no_paid_calls"),
        database_mutation_expected: json_bool(&value, "database_mutation_expected"),
    }
}

fn targeted_path_probe_artifact_from(
    value: Option<(PathBuf, serde_json::Value)>,
) -> TargetedPathProbeArtifactJson {
    let Some((path, value)) = value else {
        return TargetedPathProbeArtifactJson {
            present: false,
            run_id: None,
            summary_json: None,
            ok: None,
            status: None,
            mode: None,
            selected_count: None,
            router_missing_count: None,
            router_stale_count: None,
            recall_missing_count: None,
            recall_unembedded_count: None,
            raw_paths_included: None,
            raw_contexts_included: None,
            file_contents_read: None,
            require_embedded: None,
            recall_freshness_verifiable: None,
            elapsed_ms: None,
        };
    };
    TargetedPathProbeArtifactJson {
        present: true,
        run_id: json_string(&value, "run_id"),
        summary_json: Some(path.display().to_string()),
        ok: json_bool(&value, "ok"),
        status: json_string(&value, "status"),
        mode: json_string(&value, "mode"),
        selected_count: json_i64(&value, "selected_count")
            .or_else(|| json_i64(&value, "router_indexed_count"))
            .or_else(|| json_i64(&value, "recall_indexed_count"))
            .or_else(|| json_i64(&value, "file_present_count")),
        router_missing_count: json_i64(&value, "router_missing_count"),
        router_stale_count: json_i64(&value, "router_stale_count"),
        recall_missing_count: json_i64(&value, "recall_missing_count"),
        recall_unembedded_count: json_i64(&value, "recall_unembedded_count"),
        raw_paths_included: json_bool(&value, "raw_paths_included"),
        raw_contexts_included: json_bool(&value, "raw_contexts_included"),
        file_contents_read: json_bool(&value, "file_contents_read"),
        require_embedded: json_bool(&value, "require_embedded"),
        recall_freshness_verifiable: json_bool(&value, "recall_freshness_verifiable"),
        elapsed_ms: json_i64(&value, "elapsed_ms"),
    }
}

fn embedding_backlog_artifact_from(
    value: Option<(PathBuf, serde_json::Value)>,
) -> EmbeddingBacklogArtifactJson {
    let Some((path, value)) = value else {
        return EmbeddingBacklogArtifactJson {
            present: false,
            summary_json: None,
            ok: None,
            mode: None,
            total_unembedded: None,
            nonrisk_unembedded: None,
            path_risk_unembedded: None,
            router_eligible_pct: None,
            nonrisk_pct: None,
            path_risk_pct: None,
            selected_pct_of_total: None,
            selected_pct_of_router_eligible: None,
            requested_size: None,
            selected_count: None,
            max_per_prefix: None,
            raw_paths_included: None,
            raw_contexts_included: None,
            file_contents_read: None,
            path_list_contains_raw_paths: None,
            current_filter_enabled: None,
            current_unembedded_paths: None,
            stale_artifact_rows_removed: None,
        };
    };
    EmbeddingBacklogArtifactJson {
        present: true,
        summary_json: Some(path.display().to_string()),
        ok: json_bool(&value, "ok"),
        mode: json_string(&value, "mode"),
        total_unembedded: json_i64(&value, "total_unembedded"),
        nonrisk_unembedded: json_i64(&value, "nonrisk_unembedded"),
        path_risk_unembedded: json_i64(&value, "path_risk_unembedded"),
        router_eligible_pct: json_path_f64(&value, &["backlog_percentages", "router_eligible_pct"]),
        nonrisk_pct: json_path_f64(&value, &["backlog_percentages", "nonrisk_pct"]),
        path_risk_pct: json_path_f64(&value, &["backlog_percentages", "path_risk_pct"]),
        selected_pct_of_total: json_path_f64(
            &value,
            &["backlog_percentages", "selected_pct_of_total"],
        ),
        selected_pct_of_router_eligible: json_path_f64(
            &value,
            &["backlog_percentages", "selected_pct_of_router_eligible"],
        ),
        requested_size: json_path_i64(&value, &["selected_batch", "requested_size"]),
        selected_count: json_path_i64(&value, &["selected_batch", "selected_count"]),
        max_per_prefix: json_path_i64(&value, &["selected_batch", "max_per_prefix"]),
        raw_paths_included: json_bool(&value, "raw_paths_included"),
        raw_contexts_included: json_bool(&value, "raw_contexts_included"),
        file_contents_read: json_bool(&value, "file_contents_read"),
        path_list_contains_raw_paths: json_path_bool(
            &value,
            &["selected_batch", "path_list_contains_raw_paths"],
        ),
        current_filter_enabled: json_path_bool(&value, &["current_recall_filter", "enabled"]),
        current_unembedded_paths: json_path_i64(
            &value,
            &["current_recall_filter", "current_unembedded_paths"],
        ),
        stale_artifact_rows_removed: json_path_i64(
            &value,
            &["current_recall_filter", "stale_artifact_rows_removed"],
        ),
    }
}

fn embedding_coverage_freshness_from(
    recall_status: &StatusJson,
    coverage: &CoverageArtifactJson,
    coverage_delta_refresh: Option<&CoverageDeltaRefreshArtifactJson>,
) -> EmbeddingCoverageFreshnessJson {
    let current = (recall_status.docs_valid - recall_status.embeddings).max(0);
    let delta = coverage_delta_refresh
        .filter(|artifact| artifact.present && artifact.recall_unembedded_valid_docs.is_some());
    let (artifact_source, artifact_run_id, artifact) = match delta {
        Some(delta) => (
            "coverage_delta_refresh".to_string(),
            delta.run_id.clone(),
            delta.recall_unembedded_valid_docs,
        ),
        None => (
            "coverage_audit".to_string(),
            coverage.run_id.clone(),
            coverage.recall_unembedded_valid_docs,
        ),
    };
    let delta = artifact.map(|value| value - current);
    EmbeddingCoverageFreshnessJson {
        authoritative_source: "recall_status_current_db".to_string(),
        artifact_source,
        artifact_run_id,
        current_db_unembedded_valid_docs: current,
        artifact_unembedded_valid_docs: artifact,
        artifact_unembedded_delta_vs_current_db: delta,
        artifact_matches_current_db: delta.map(|value| value == 0),
    }
}

fn memory_doctor_json() -> Result<MemoryDoctorJson> {
    let store = Store::open()?;
    let recall_status = status_json_payload_from_store(&store)?;
    let vault_root = store::vault_root();
    // A developer report over recall's eval artifacts: it needs the development folder, and says so.
    let results_root = recall_dev_dir()?.join("eval").join("results");
    let coverage = coverage_artifact_from(latest_nested_summary_across_roots_by_mtime(
        coverage_audit_roots(&results_root),
        "coverage_summary.json",
    ));
    let coverage_delta_refresh =
        coverage_delta_refresh_artifact_from(latest_nested_summary_by_mtime(
            results_root.join("coverage_delta_refresh"),
            "coverage_delta_refresh_summary.json",
        ));
    let embedding_coverage_freshness =
        embedding_coverage_freshness_from(&recall_status, &coverage, Some(&coverage_delta_refresh));
    let ragas = ragas_artifact_from(latest_nested_summary(
        results_root.join("ragas_smoke"),
        "ragas_smoke_summary.json",
    ));
    let bench_root = results_root.join("bench_release");
    let latest_bench = bench_artifact_from(latest_bench_report(bench_root.clone()));
    let high_water = bench_high_water(bench_root);
    let release_mode_sweep = release_mode_sweep_artifact_from(latest_nested_summary_by_mtime(
        results_root.join("release_mode_sweep"),
        "release_mode_sweep_summary.json",
    ));
    let embedding_benchmark_gate =
        latest_release_sweep_benchmark_evidence(results_root.join("release_mode_sweep"))
            .map(|evidence| embedding_benchmark_gate_from(&evidence, &high_water))
            .unwrap_or_else(|| embedding_benchmark_gate_from(&latest_bench, &high_water));
    let vault_only_benchmark = vault_only_benchmark_artifact_from(latest_nested_summary_by_mtime(
        results_root.join("vault_only_benchmark"),
        "vault_only_benchmark_summary.json",
    ));
    let targeted_path_probe = targeted_path_probe_artifact_from(
        latest_targeted_path_probe_artifact(results_root.join("coverage_repair")),
    );
    let embedding_backlog = embedding_backlog_artifact_from(latest_nested_summary_by_mtime(
        results_root.join("embedding_backlog"),
        "embedding_backlog_summary.json",
    ));
    #[cfg(feature = "ops-health")]
    let memory_broker_db_health = memory_broker_db_health_json(20);
    #[cfg(feature = "ops-health")]
    let tailscale_local_health = tailscale_local_health_json(5);
    #[cfg(feature = "ops-health")]
    let gateway_local_health = gateway_local_health_json();
    let embed_disk_space_gate = embed_disk_space_gate(&vault_root);
    let active_retrieval_modes = ActiveRetrievalModesJson {
        denoise_mode: query::effective_denoise_mode(),
        denoise_mode_set: std::env::var_os("RECALL_DENOISE_MODE").is_some(),
        fusion_mode: query::effective_fusion_mode(),
        fusion_mode_set: std::env::var_os("RECALL_FUSION_MODE").is_some(),
        rrf_k: query::effective_rrf_k(),
        rrf_k_set: std::env::var_os("RECALL_RRF_K").is_some(),
        min_fusion_depth: query::effective_min_fusion_depth(),
        min_fusion_depth_set: std::env::var_os("RECALL_MIN_FUSION_DEPTH").is_some(),
        fusion_depth_multiplier: query::effective_fusion_depth_multiplier(),
        fusion_depth_multiplier_set: std::env::var_os("RECALL_FUSION_DEPTH_MULTIPLIER").is_some(),
    };
    let gates = MemoryDoctorGatesJson {
        recall_store_present: recall_status.docs_total > 0,
        recall_index_has_docs: recall_status.docs_valid > 0,
        recall_embeddings_complete: recall_status.docs_valid > 0
            && recall_status.embeddings >= recall_status.docs_valid,
        coverage_artifact_present: coverage.present,
        coverage_delta_refresh_present: coverage_delta_refresh.present,
        coverage_delta_refresh_redacted: coverage_delta_refresh.raw_paths_included == Some(false)
            && coverage_delta_refresh.raw_contexts_included == Some(false)
            && coverage_delta_refresh.raw_secret_values_included == Some(false)
            && coverage_delta_refresh.file_contents_read == Some(false)
            && coverage_delta_refresh.no_llm == Some(true)
            && coverage_delta_refresh.no_paid_calls == Some(true)
            && coverage_delta_refresh.database_mutation_expected == Some(false)
            && coverage_delta_refresh.redacted_surrogate_unsafe_manifest_count == Some(0)
            && coverage_delta_refresh.router_missing_target_docs_without_redacted_surrogate
                == Some(0)
            && coverage_delta_refresh.recall_missing_target_docs_without_redacted_surrogate
                == Some(0),
        ragas_smoke_green: ragas.passed == Some(true) && ragas.secret_leak_count == Some(0),
        secret_leak_rate_zero: ragas.secret_leak_count.unwrap_or(0) == 0,
        release_benchmark_present: latest_bench.present,
        release_mode_sweep_present: release_mode_sweep.present,
        release_mode_sweep_not_blocked: release_mode_sweep.status.as_deref()
            != Some("blocked_environment"),
        latest_bench_no_raw_contexts: latest_bench.raw_contexts_included == Some(false),
        vault_only_benchmark_present: vault_only_benchmark.present,
        vault_only_benchmark_redacted: vault_only_benchmark.raw_contexts_included == Some(false)
            && vault_only_benchmark.raw_paths_included == Some(false)
            && vault_only_benchmark.fixture_ids_included == Some(false)
            && vault_only_benchmark.secret_values_included == Some(false)
            && vault_only_benchmark.no_llm == Some(true)
            && vault_only_benchmark.no_paid_calls == Some(true),
        embedding_benchmark_gate_present: embedding_benchmark_gate.present,
        embedding_benchmark_gate_passed: embedding_benchmark_gate.ok,
        embed_disk_space_gate_present: embed_disk_space_gate.present,
        embed_disk_space_gate_passed: embed_disk_space_gate.ok,
        #[cfg(feature = "ops-health")]
        memory_broker_db_proven: memory_broker_db_health.ok,
        #[cfg(feature = "ops-health")]
        tailscale_local_ready: tailscale_local_health.ok,
        #[cfg(feature = "ops-health")]
        gateway_local_ready: gateway_local_health.ok,
        targeted_path_probe_present: targeted_path_probe.present,
        targeted_path_probe_redacted: targeted_path_probe.raw_paths_included == Some(false)
            && targeted_path_probe.raw_contexts_included == Some(false)
            && targeted_path_probe.file_contents_read == Some(false),
        embedding_backlog_present: embedding_backlog.present,
        embedding_backlog_redacted: embedding_backlog.raw_paths_included == Some(false)
            && embedding_backlog.raw_contexts_included == Some(false)
            && embedding_backlog.file_contents_read == Some(false)
            && embedding_backlog.path_list_contains_raw_paths == Some(false),
    };
    let ok = gates.recall_store_present
        && gates.recall_index_has_docs
        && gates.recall_embeddings_complete
        && gates.coverage_artifact_present
        && gates.coverage_delta_refresh_present
        && gates.coverage_delta_refresh_redacted
        && gates.ragas_smoke_green
        && gates.secret_leak_rate_zero
        && gates.release_benchmark_present
        && gates.release_mode_sweep_present
        && gates.release_mode_sweep_not_blocked
        && gates.latest_bench_no_raw_contexts
        && gates.vault_only_benchmark_present
        && gates.vault_only_benchmark_redacted
        && gates.embedding_benchmark_gate_present
        && gates.embedding_benchmark_gate_passed
        && gates.embed_disk_space_gate_present
        && gates.embed_disk_space_gate_passed
        && ops_health_ready(&gates)
        && gates.targeted_path_probe_present
        && gates.targeted_path_probe_redacted
        && gates.embedding_backlog_present
        && gates.embedding_backlog_redacted;
    Ok(MemoryDoctorJson {
        ok,
        status: if ok { "healthy" } else { "degraded" }.to_string(),
        recall_status,
        active_retrieval_modes,
        coverage_artifact: coverage,
        coverage_delta_refresh,
        embedding_coverage_freshness,
        ragas_smoke: ragas,
        latest_release_benchmark: latest_bench,
        release_benchmark_high_water: high_water,
        release_mode_sweep,
        vault_only_benchmark,
        embedding_benchmark_gate,
        embed_disk_space_gate,
        #[cfg(feature = "ops-health")]
        memory_broker_db_health,
        #[cfg(feature = "ops-health")]
        tailscale_local_health,
        #[cfg(feature = "ops-health")]
        gateway_local_health,
        targeted_path_probe,
        embedding_backlog,
        gates,
        raw_contexts_included: false,
        no_llm: true,
        no_paid_calls: true,
    })
}

fn cmd_refresh(scan_only: bool, no_embed: bool, force: bool, if_due: bool, status: bool, json: bool) -> Result<()> {
    let store = Store::open()?;
    store.conn.busy_timeout(std::time::Duration::from_secs(30)).ok();
    let settings = settings::Settings::load();
    let state = refresh::RefreshState::load();
    if status {
        let c = store.freshness_counts()?;
        let v = serde_json::json!({
            "refresh": settings.refresh.label(),
            "changed": c.changed, "deleted": c.deleted, "new_pending": c.new_pending,
            "last_ok": state.last_ok_unix.map(store::freshness::fmt_utc),
            "last_summary": state.last_summary,
            "watcher_pid": state.watcher_pid,
            "watcher_started": state.watcher_started_unix.map(store::freshness::fmt_utc),
        });
        if json {
            println!("{}", serde_json::to_string_pretty(&v)?);
        } else {
            println!("refresh setting : {}", settings.refresh.label());
            println!("pending         : {} changed, {} deleted, {} new", c.changed, c.deleted, c.new_pending);
            println!("last ok         : {}", v["last_ok"].as_str().unwrap_or("never"));
            println!("last summary    : {}", state.last_summary.as_deref().unwrap_or("-"));
        }
        return Ok(());
    }
    // `--status` above is a database read; everything below scans or rewrites against the disk.
    let root = store::vault_root_checked()?;
    if if_due {
        let c = store.freshness_counts()?;
        let pending = c.changed + c.deleted + c.new_pending > 0;
        if !refresh::is_due(&settings, state.last_ok_unix, store::freshness::now_secs(), pending, true) {
            println!("refresh not due (setting: {})", settings.refresh.label());
            return Ok(());
        }
    }
    if scan_only {
        let _lock = match refresh::IndexLock::try_acquire()? {
            Some(l) => l,
            None => {
                println!("SKIPPED: another index/refresh holds the index lock");
                return Ok(());
            }
        };
        let r = store.scan_freshness(&root, true, 20_000)?;
        println!(
            "scan: {} checked, {} changed, {} deleted, {} new file(s) queued, {} legacy row(s) stamped ({} left)",
            r.checked, r.changed, r.deleted, r.new_files, r.legacy_stamped, r.legacy_remaining
        );
        return Ok(());
    }
    let opts = refresh::RefreshOpts { embed: !no_embed, ignore_busy: force, ..Default::default() };
    match refresh::run(&store, &root, &opts)? {
        refresh::Outcome::Skipped(why) => {
            if json {
                println!("{}", serde_json::json!({"ran": false, "skipped": why}));
            } else {
                println!("SKIPPED (nothing was refreshed): {why}");
            }
        }
        refresh::Outcome::Ran(rep) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&rep)?);
            } else {
                println!(
                    "refresh: scanned {} docs; flagged {} changed, {} deleted, {} new",
                    rep.scan_checked, rep.scan_changed, rep.scan_deleted, rep.scan_new
                );
                println!("refresh: {}", rep.summary());
                if rep.legacy_remaining > 0 {
                    println!(
                        "refresh: {} legacy row(s) have no recorded mtime yet; the next runs stamp them",
                        rep.legacy_remaining
                    );
                }
            }
        }
    }
    Ok(())
}

fn cmd_settings(action: Option<String>, key: Option<String>, value: Option<String>) -> Result<()> {
    match (action.as_deref(), key, value) {
        (None | Some("list"), _, _) => {
            let s = settings::Settings::load();
            for k in settings::KEYS {
                println!("{k:<16} {}", s.get(k).unwrap_or_default());
            }
            println!("(file: {})", settings::settings_path().display());
        }
        (Some("get"), Some(k), _) => match settings::Settings::load().get(&k) {
            Some(v) => println!("{v}"),
            None => anyhow::bail!("unknown setting '{k}' (known: {})", settings::KEYS.join(", ")),
        },
        (Some("set"), Some(k), Some(v)) => {
            let s = settings::Settings::set(&k, &v)?;
            println!("{k} = {}", s.get(&k).unwrap_or_default());
            if k == "vault" {
                // the daemon's file watcher binds its folder at start (refresh.rs)
                println!("(restart the daemon to watch the new folder: recall daemon down, then recall daemon up)");
            } else {
                println!("(the daemon re-reads settings.json every 30 s; no restart needed)");
            }
        }
        _ => anyhow::bail!("usage: recall settings [list] | get KEY | set KEY VALUE"),
    }
    Ok(())
}


/// Refuses `index --prune-missing=apply` without rg BEFORE the index is touched. `rg_present` is
/// only asked when it matters (it spawns `rg --version`).
fn index_prune_preflight(prune_mode: Option<store::PruneMode>, rg_present: impl FnOnce() -> bool) -> Result<()> {
    if prune_mode == Some(store::PruneMode::Apply) && !rg_present() {
        store::warn_rg_missing();
        anyhow::bail!("index --prune-missing=apply refused before indexing: nothing was indexed or pruned. {}", store::RG_MISSING);
    }
    Ok(())
}

fn cmd_status(json: bool, min_coverage: Option<f64>) -> Result<()> {
    let store = Store::open()?;
    // A scoped reader gets its own view only: whole-corpus counts, sources and coverage are the
    // operator's (Codex tenancy r2). One line, plus JSON for tools.
    if let r @ scope::Reader::Agent { .. } = scope::current() {
        // A coverage gate that cannot run must fail, never pass silently (CodeRabbit, tenancy PR).
        if min_coverage.is_some() {
            anyhow::bail!("status --min-coverage is an operator check: unset RECALL_AGENT/RECALL_PROJECT/RECALL_TEAM");
        }
        let visible = store.count_visible()?;
        if json {
            println!("{}", serde_json::json!({ "reader": r.describe(), "visible_docs": visible }));
        } else {
            reply::section("WHAT I DID", &format!("{} | visible docs: {visible}", r.describe()));
        }
        return Ok(());
    }
    let s = store.stats()?;
    // Coverage of the table the query path READS (bge_chunk_embeddings). The legacy pooled
    // `embeddings` table is reported separately: counting it made `status` print 72.81% while
    // chunk coverage was 100% (E2E 2a, 2026-10-29).
    let coverage = coverage_pct(s.chunk_embedded as usize, s.valid as usize);

    if json {
        let payload = status_json_payload(&s);
        // Additive `scrub` block: the redaction settings in force, so a benchmark's output says
        // whether its index was built from redacted or raw text.
        let mut shown = serde_json::to_value(&payload)?;
        shown["scrub"] = scrub::Config::status_summary();
        shown["vault"] = serde_json::json!(store::vault_root().display().to_string());
        shown["vault_source"] = serde_json::json!(store::vault_source());
        println!("{}", serde_json::to_string_pretty(&shown)?);
        if let Some(threshold) = min_coverage {
            if !coverage_meets(coverage, threshold) {
                anyhow::bail!(
                    "status gate FAILED: coverage {coverage:.4}% < threshold {threshold:.4}%"
                );
            }
        }
        return Ok(());
    }

    println!("\n{}", reply::paint(reply::BOLD, "recall status"));
    println!("   docs (valid)  : {}", s.valid);
    println!("   docs (total)  : {}", s.total);
    println!("   superseded    : {}", s.superseded);
    // The legacy line only on a store that still holds legacy vectors.
    let lines = status_vector_lines(s.chunk_embedded, s.embeddings, s.valid);
    for line in lines.iter().take(if s.embeddings == 0 { 1 } else { 2 }) {
        println!("{line}");
    }
    // Report the SEMANTIC channel's own coverage, not just BGE's.
    //
    // Gemini vectors were write-only for weeks: the table filled up while `recall query`
    // could not read a single row, and nothing on this screen would have told you. Count
    // what is reachable THROUGH THE JOIN the query path actually uses — a row that fails
    // that join is not coverage, it is an orphan.
    #[cfg(feature = "gemini")]
    {
        let g = gemini::readable_docs(&store).unwrap_or(0);
        println!(
            "   gemini        : {} / {}  ({:.2}%)  [query-readable]",
            g,
            s.valid,
            coverage_pct(g as usize, s.valid as usize)
        );
    }
    println!(
        "   db size       : {:.1} MB",
        s.db_bytes as f64 / 1_048_576.0
    );
    println!("   db path       : {}", store::default_db_path()?.display());
    println!("   vault         : {}  (from {})", store::vault_root().display(), store::vault_source());
    if store::rg_available() {
        println!("   rg            : available");
    } else {
        println!("   rg            : MISSING. {}", store::RG_MISSING);
    }
    if let Ok(c) = store.freshness_counts() {
        let st = refresh::RefreshState::load();
        let last = st
            .last_ok_unix
            .map(store::freshness::fmt_utc)
            .unwrap_or_else(|| "never".into());
        println!(
            "   freshness     : {} changed, {} deleted, {} new on disk, not yet re-indexed (refresh: {}; last ok: {last})",
            c.changed,
            c.deleted,
            c.new_pending,
            settings::Settings::load().refresh.label()
        );
    }
    println!("   by source:");
    for (src, n) in &s.by_source {
        println!("      {:<14} {}", src, n);
    }
    println!();
    if let Some(threshold) = min_coverage {
        if !coverage_meets(coverage, threshold) {
            anyhow::bail!(
                "status gate FAILED: coverage {coverage:.4}% < threshold {threshold:.4}%"
            );
        }
        println!("   status gate PASSED: coverage {coverage:.4}% >= {threshold:.4}%\n");
    }
    Ok(())
}

// Measure recall@k of the active search path vs the exact brute-force ground
// truth. Freeze-safe: loads the model + vectors once, then runs single queries
// (no batch/all-core embedding). With `ann` off, the active path is the exact
// scan itself, so recall is 1.0 by construction — the metric only bites once
// an ANN index backs the vector channel.
fn cmd_eval_recall(top: Option<String>, min_recall: Option<f32>) -> Result<()> {
    let top: usize = top.as_deref().unwrap_or("5").parse().unwrap_or(5);
    // Clamp like the sibling count args (query/bench/embed/insights/recent). An
    // unbounded `top` flows into the ANN path's `depth * over_fetch` (ann.rs:188,
    // k*4) which wraps in release before `.min(count)` — clamp to the same
    // MAX_QUERY_K cmd_query uses so eval can't request a pathological k.
    let top = top.clamp(1, daemon::MAX_QUERY_K);
    let store = Store::open()?;
    let search = match bge::BgeSearch::load(&store)? {
        Some(s) => s,
        None => {
            reply::section("WHAT I DID", "eval: no vectors stored — nothing to measure");
            // A CI gate can't pass with nothing to measure.
            if min_recall.is_some() {
                anyhow::bail!("eval gate: no vectors stored — cannot verify recall");
            }
            return Ok(());
        }
    };

    println!("\n{}\n", reply::paint(reply::BOLD, &format!("recall eval: recall@{top} vs exact brute-force ground truth")));
    let mut sum = 0.0f32;
    let mut n = 0usize;
    let mut worst = (1.0f32, "");
    let queries = eval::fixture_queries();
    for q in queries.iter().map(String::as_str) {
        let truth = search.exact_search_ids(q, top)?;
        let approx = search.search_ids(q, top)?;
        let r = eval::recall_at_k(&approx, &truth, top);
        if r < worst.0 {
            worst = (r, q);
        }
        sum += r;
        n += 1;
        println!("   recall@{top}={r:.3}   {q}");
    }
    let mean = if n > 0 { sum / n as f32 } else { 0.0 };
    if n > 0 {
        println!("\n   mean recall@{top} = {mean:.4}   ({n} queries)");
        if worst.0 < 1.0 {
            println!("   worst: recall@{top}={:.3}  [{}]", worst.0, worst.1);
        } else {
            println!("   all queries perfect (active path == exact ground truth)");
        }
    }
    println!();
    if let Some(threshold) = min_recall {
        if mean < threshold {
            anyhow::bail!(
                "eval gate FAILED: mean recall@{top} = {mean:.4} < threshold {threshold:.4}"
            );
        }
        println!("   eval gate PASSED: mean recall@{top} = {mean:.4} ≥ {threshold:.4}\n");
    }
    Ok(())
}

/// Refuse a whole-corpus command for a scoped reader.
fn operator_only(cmd: &str) -> Result<()> {
    if !scope::current().is_operator() {
        anyhow::bail!("recall {cmd} is an operator command: unset RECALL_AGENT/RECALL_PROJECT/RECALL_TEAM");
    }
    Ok(())
}

fn cmd_insights(min: Option<String>) -> Result<()> {
    // Operator-only: it clusters and prints every observation in the corpus.
    operator_only("insights")?;
    // Upper bound mirrors dream's 10k observation load cap: a larger min can
    // never match any cluster, so it would silently return nothing. Floor of 1
    // keeps it consistent with the sibling count/top/batch clamps.
    let min: usize = min
        .as_deref()
        .unwrap_or("2")
        .parse()
        .unwrap_or(2)
        .clamp(1, 10_000);
    let store = Store::open()?;
    let found = dream::insights(&store, min)?;
    let sc = scrub::Scrubber::new(&scrub::Config::load());
    if found.is_empty() {
        reply::section(
            "WHAT I DID",
            "No recurring themes yet — save more observations.",
        );
        return Ok(());
    }
    println!("\n{}\n", reply::paint(reply::BOLD, &format!("recall insights: {} theme(s)", found.len())));
    for (i, ins) in found.iter().enumerate() {
        println!(
            "  {}. [{} memories] {}",
            i + 1,
            ins.size,
            human_text(&sc, &ins.keywords.join(", "))
        );
        println!("     e.g. {}", human_text(&sc, &ins.representative).replace('\n', " "));
    }
    println!();
    Ok(())
}

fn cmd_recent(count: Option<String>) -> Result<()> {
    let n: usize = count
        .as_deref()
        .unwrap_or("10")
        .parse()
        .unwrap_or(10)
        .clamp(1, 1000);
    let store = Store::open()?;
    let rows = store.recent(n)?;
    if rows.is_empty() {
        reply::section("WHAT I DID", "No saved observations yet.");
        return Ok(());
    }
    let sc = scrub::Scrubber::new(&scrub::Config::load());
    println!("\n{}\n", reply::paint(reply::BOLD, &format!("recall recent: last {} observation(s)", rows.len())));
    for h in &rows {
        println!(
            "  #{:<5} [{}] {}",
            h.id,
            human_text(&sc, &h.title),
            human_text(&sc, &h.snippet).replace('\n', " ")
        );
    }
    println!();
    Ok(())
}

fn cmd_forget(text: String) -> Result<()> {
    let store = Store::open()?;
    let n = store.forget(&text)?;
    reply::section(
        "WHAT I DID",
        &format!(
            "Retired {} memory(ies) matching \"{}\" (reversible via DB)",
            n, text
        ),
    );
    Ok(())
}

fn cmd_supersede(id: String) -> Result<()> {
    let id: i64 = match id.parse() {
        Ok(v) => v,
        Err(_) => {
            reply::blocking("supersede", "id must be an integer");
            return Ok(());
        }
    };
    let store = Store::open()?;
    if store.supersede_id(id)? {
        reply::section("WHAT I DID", &format!("Retired memory #{}", id));
    } else {
        reply::blocking(
            "supersede",
            &format!("#{} not found or already retired", id),
        );
    }
    Ok(())
}

fn cmd_export() -> Result<()> {
    let store = Store::open()?;
    let stdout = std::io::stdout();
    store.export_json(stdout.lock())?;
    Ok(())
}

fn cmd_gc(apply: bool, exclude_aware: bool) -> Result<()> {
    let store = Store::open()?;
    let orphaned = store.count_orphaned_embeddings()?;
    let live_before = store.count_embeddings()?;

    if exclude_aware {
        let excluded_live = store.count_exclude_policy_embeddings()?;
        if apply {
            reply::blocking(
                "gc",
                "`recall gc --exclude-aware --apply` is intentionally not implemented yet; run dry-run only and council-gate any future delete mode.",
            );
            return Ok(());
        }
        let l1 = format!(
            "Exclude-policy live embeddings (dry-run only): {}",
            excluded_live
        );
        let l2 = format!("Live embeddings checked:                 {}", live_before);
        let l3 = "No rows deleted. This mode reads paths only and reports counts for council-gated cleanup design.";
        reply::card(&[("WHAT I DID", &[l1.as_str(), l2.as_str(), l3][..])]);
        return Ok(());
    }

    // `embeddings2` (Voyage) has no ON DELETE CASCADE, so its orphans are invisible to
    // `count_orphaned_embeddings`, which only ever looked at the legacy `embeddings`
    // table. They are not harmless: the vector scan ranks them, so dead rows occupy real
    // result slots. Measured 2026-07-31 — tens of thousands of orphaned doc groups, every relevant hit
    // pushed down a rank, R@1 flat 0.0000. gc reported "store is clean" throughout.
    // `mod voyage` is gated behind the `voyage` feature, so this call has to be too -
    // without the gate a default-feature build (and therefore `cargo test`) fails to
    // compile with E0433. Feature off means no embeddings2 table to have orphans in.
    #[cfg(feature = "voyage")]
    let orphaned2 = voyage::count_orphaned_embeddings2(&store).unwrap_or(0);
    #[cfg(not(feature = "voyage"))]
    let orphaned2 = 0_i64;

    if !apply {
        // Dry-run path (default).
        let l1 = format!("Orphaned embeddings (tombstoned docs): {}", orphaned);
        let l2 = format!("Orphaned embeddings2 rows (Voyage):    {}", orphaned2);
        let l3 = format!("Live embeddings (would be kept):       {}", live_before);
        let l4 = if orphaned == 0 && orphaned2 == 0 {
            "Nothing to delete — store is clean.".to_string()
        } else {
            format!(
                "Run `recall gc --apply` to delete {} orphaned embedding(s) and {} embeddings2 row(s).",
                orphaned, orphaned2
            )
        };
        reply::card(&[(
            "WHAT I DID",
            &[l1.as_str(), l2.as_str(), l3.as_str(), l4.as_str()][..],
        )]);
        return Ok(());
    }

    // Apply path.
    let deleted = store.delete_orphaned_embeddings()?;
    #[cfg(feature = "voyage")]
    let deleted2 = voyage::delete_orphaned_embeddings2(&store).unwrap_or(0);
    #[cfg(not(feature = "voyage"))]
    let deleted2 = 0_i64;
    let live_after = store.count_embeddings()?;

    let l0 = format!("Deleted {} orphaned embeddings2 row(s).", deleted2);
    let l1 = format!("Deleted {} orphaned embedding(s).", deleted);
    let l2 = format!(
        "Live embeddings before: {}  after: {}",
        live_before, live_after
    );
    let l3 = if live_before == live_after {
        "Coverage invariant HELD — no live embeddings lost.".to_string()
    } else {
        format!(
            "WARNING: live embedding count changed ({} -> {}) — investigate before next recall embed.",
            live_before, live_after
        )
    };
    reply::card(&[(
        "WHAT I DID",
        &[l1.as_str(), l0.as_str(), l2.as_str(), l3.as_str()][..],
    )]);
    Ok(())
}

/// Build (or rebuild) the HNSW ANN index from existing BGE embeddings.
///
/// FREEZE GUARD: reads pre-stored vectors from recall.db — does NOT call
/// fastembed / ONNX / recall embed.  Safe to run under the daemon freeze
/// guard (12-core affinity + BelowNormal) on first build.
#[cfg(feature = "ann")]
fn cmd_index_ann() -> Result<()> {
    let store = Store::open()?;
    let vectors = store.load_all_embeddings()?;
    if vectors.is_empty() {
        reply::blocking(
            "index-ann",
            "No BGE vectors found — run `recall embed` first to populate embeddings.",
        );
        return Ok(());
    }
    let n = vectors.len();
    let ann_path = ann::default_ann_path()?;
    // Ensure ~/.recall/ exists.
    if let Some(parent) = ann_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    eprintln!("ann: building index over {} vectors…", n);
    let t0 = std::time::Instant::now();
    let idx = ann::AnnIndex::build(vectors)?;
    let elapsed = t0.elapsed().as_secs_f32();
    idx.dump(&ann_path)?;
    reply::card(&[(
        "WHAT I DID",
        &[
            format!("Built HNSW ANN index over {} vectors in {:.1}s", n, elapsed).as_str(),
            format!("Persisted to {}", ann_path.display()).as_str(),
        ][..],
    )]);
    Ok(())
}

#[cfg(test)]
mod rg_preflight_tests {
    use super::*;

    #[test]
    fn an_apply_prune_without_rg_is_refused_before_indexing() {
        let err = index_prune_preflight(Some(store::PruneMode::Apply), || false).unwrap_err();
        assert!(err.to_string().contains("refused before indexing"), "{err}");
        assert!(index_prune_preflight(Some(store::PruneMode::Apply), || true).is_ok());
        // no prune, or a dry prune (which returns before any write), never asks for rg here
        assert!(index_prune_preflight(None, || panic!("must not probe")).is_ok());
        assert!(index_prune_preflight(Some(store::PruneMode::Dry), || panic!("must not probe")).is_ok());
    }
}

#[cfg(test)]
mod provenance_output_tests {
    use super::{provenance_line, QueryJsonHit};
    use std::collections::BTreeMap;

    /// T10: the JSON contract is the seven legacy keys plus ranks, scores, rrf.
    #[test]
    fn query_json_hit_contract() {
        let h = QueryJsonHit {
            doc_id: 1,
            rank: 0,
            score: 2.5,
            vault_path: "a.md".into(),
            title: "A".into(),
            snippet: String::new(),
            source: "fts5+bge".into(),
            ranks: BTreeMap::from([("fts5".to_string(), 1), ("bge".to_string(), 0)]),
            scores: BTreeMap::from([("fts5".to_string(), 2.5), ("bge".to_string(), 0.78)]),
            rrf: 0.0325,
            lacks: Vec::new(),
            merged: Vec::new(),
            labels_truncated: false,
            redactions: Vec::new(),
            fresh: Default::default(),
            closest_match: false,
            absent: Vec::new(),
        };
        let v = serde_json::to_value(&h).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            ["doc_id", "rank", "ranks", "rrf", "score", "scores", "snippet", "source", "stale", "title", "vault_path"]
        );
        // `stale` is ALWAYS present; null means "could not check", never "fresh".
        assert!(v["stale"].is_null());
        assert!(v["score"].is_f64() && v["doc_id"].is_i64() && v["rank"].is_u64());
        assert_eq!(v["scores"]["bge"], 0.78);
        assert_eq!(
            provenance_line(&h.source, &h.ranks, &h.scores, h.rrf),
            "via fts5+bge · fts5 2.5000 · bge 0.7800 · rank-sum 0.03250"
        );
        // lme-improve: a closest match is stated in the JSON even when it lacks nothing.
        let c = QueryJsonHit { closest_match: true, ..h };
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["closest_match"], true);
        assert!(v.get("lacks").is_none());
        let h = QueryJsonHit { closest_match: false, ..c };
        let rg = BTreeMap::from([("rg".to_string(), 2)]);
        assert_eq!(provenance_line("rg", &rg, &BTreeMap::new(), 0.01), "via rg · rg #3 · rank-sum 0.01000");
    }
}

#[cfg(test)]
mod tests {
    use super::{
        status_vector_lines, Cli, Commands,
        coverage_meets, coverage_pct, embed_batch_len_ok, embed_disk_space_decision,
        embed_disk_space_gate_from_result, embedding_benchmark_gate_decision,
        embedding_benchmark_gate_from, embedding_benchmark_gate_from_sweep,
        embedding_coverage_freshness_from,
        latest_nested_summary, latest_nested_summary_by_mtime,
        latest_release_sweep_benchmark_evidence, latest_surrogate_manifests,
        min_embed_free_bytes_from_env, parse_timeout_seconds,
        targeted_path_probe_artifact_from, ActiveRetrievalModesJson,
        BenchArtifactJson, BenchHighWaterJson, BgeVectorStatsJson, CoverageArtifactJson,
        CoverageDeltaRefreshArtifactJson, EmbedDiskSpaceGateJson, EmbedDoctorJson,
        EmbeddingBacklogArtifactJson, EmbeddingBenchmarkGateJson, EmbeddingCoverageFreshnessJson,
        HealthDoctorJson,
        MemoryDoctorGatesJson, MemoryDoctorJson, RagasArtifactJson, ReleaseModeSweepArtifactJson,
        StatusJson, TargetedPathProbeArtifactJson,
        VaultOnlyBenchmarkArtifactJson, DEFAULT_EMBED_DOCTOR_TIMEOUT_SECS,
    };
    #[cfg(feature = "ops-health")]
    use super::{
        gateway_mcp_auth_gate_observed, gateway_mcp_candidate_ports_from_env_port, gateway_mcp_port_listening,
        memory_broker_db_health_from_value, select_gateway_mcp_probe, GatewayLocalHealthJson, GatewayMcpProbe,
        MemoryBrokerDbHealthJson, TailscaleLocalHealthJson,
    };
    use crate::store::Store;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[test]
    fn embed_batch_len_ok_accepts_equal_counts() {
        assert!(embed_batch_len_ok(64, 64).is_ok());
        assert!(embed_batch_len_ok(0, 0).is_ok());
    }

    #[cfg(feature = "ops-health")]
    #[test]
    fn memory_broker_db_health_surfaces_bus_reason_when_unhealthy() {
        let health = memory_broker_db_health_from_value(serde_json::json!({
            "ok": false,
            "source": "control-plane-http",
            "reason": "control-plane fetch failed",
            "control_plane_http": {
                "ok": false,
                "reason": "fetch failed"
            },
            "postgres_direct": {
                "ok": false
            },
            "body": null
        }));

        assert!(!health.ok);
        assert_eq!(health.source.as_deref(), Some("control-plane-http"));
        assert_eq!(
            health.error_class.as_deref(),
            Some("control_plane_fetch_failed")
        );
        assert_eq!(
            health.error.as_deref(),
            Some("redacted:control_plane_fetch_failed")
        );
        assert_eq!(health.postgres_direct_ok, Some(false));
    }

    #[test]
    fn latest_nested_summary_by_mtime_includes_root_level_summary_file() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after epoch")
            .as_nanos();
        let root: PathBuf = std::env::temp_dir().join(format!("recall-summary-root-test-{unique}"));
        let child = root.join("20260715T231600Z");
        fs::create_dir_all(&child).expect("create child run directory");
        fs::write(
            child.join("coverage_delta_refresh_summary.json"),
            r#"{"run_id":"20260715T231600Z"}"#,
        )
        .expect("write child summary");
        std::thread::sleep(Duration::from_millis(30));
        fs::write(
            root.join("coverage_delta_refresh_summary.json"),
            r#"{"run_id":"20260716T083945Z"}"#,
        )
        .expect("write root summary");

        let (_, payload) =
            latest_nested_summary_by_mtime(root.clone(), "coverage_delta_refresh_summary.json")
                .expect("summary should be selected");
        assert_eq!(payload["run_id"], "20260716T083945Z");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn latest_nested_summary_finds_newer_deeply_nested_summary_file() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after epoch")
            .as_nanos();
        let root: PathBuf = std::env::temp_dir().join(format!("recall-ragas-nested-test-{unique}"));
        let stale_child = root.join("20260718T022355Z");
        let fresh_child = root
            .join("local-safe38b-20260719T2052Z")
            .join("20260719T205250Z");
        fs::create_dir_all(&stale_child).expect("create stale child run directory");
        fs::create_dir_all(&fresh_child).expect("create fresh nested run directory");
        fs::write(
            stale_child.join("ragas_smoke_summary.json"),
            r#"{"run_id":"20260718T022355Z","passed":true}"#,
        )
        .expect("write stale summary");
        std::thread::sleep(Duration::from_millis(30));
        fs::write(
            fresh_child.join("ragas_smoke_summary.json"),
            r#"{"run_id":"20260719T205250Z","passed":true}"#,
        )
        .expect("write fresh nested summary");

        let (_, payload) = latest_nested_summary(root.clone(), "ragas_smoke_summary.json")
            .expect("summary should be selected");
        assert_eq!(payload["run_id"], "20260719T205250Z");

        let _ = fs::remove_dir_all(root);
    }

    #[cfg(feature = "ops-health")]
    #[test]
    fn memory_broker_db_health_redacts_postgres_connection_error() {
        let health = memory_broker_db_health_from_value(serde_json::json!({
            "ok": false,
            "source": "control-plane-http",
            "control_plane_http": {
                "ok": false,
                "error": "psql: error: connection to server at \"db.example.supabase.co\" (2600:1f14::1), port 6543 failed: Permission denied (0x0000271D/10013)"
            },
            "postgres_direct": {
                "ok": false
            },
            "body": null
        }));

        assert!(!health.ok);
        assert_eq!(
            health.error_class.as_deref(),
            Some("postgres_network_permission_denied")
        );
        assert_eq!(
            health.error.as_deref(),
            Some("redacted:postgres_network_permission_denied")
        );
        assert!(!health.error.as_deref().unwrap_or("").contains("db.example"));
        assert!(!health.error.as_deref().unwrap_or("").contains("2600:1f14"));
    }

    #[test]
    fn embed_batch_len_ok_rejects_short_return() {
        // 63 vectors for 64 inputs would truncate the zip → infinite loop.
        let err = embed_batch_len_ok(63, 64).unwrap_err();
        assert!(err.to_string().contains("refusing to loop"));
    }

    #[test]
    fn embed_disk_space_decision_passes_at_floor() {
        assert!(embed_disk_space_decision(1_000_000_000, 1_000_000_000).is_ok());
    }

    #[test]
    fn embed_disk_space_decision_blocks_below_floor() {
        let err = embed_disk_space_decision(999_999_999, 1_000_000_000).unwrap_err();
        assert!(err.contains("below embed apply floor"));
        assert!(err.contains("1.000 GB"));
    }

    #[test]
    fn embed_disk_space_gate_json_blocks_below_floor() {
        let gate = embed_disk_space_gate_from_result(Ok(999_999_999), 1_000_000_000);
        assert!(!gate.ok);
        assert_eq!(gate.status, "blocking");
        assert!(gate.blocks_embedding_apply);
        assert_eq!(gate.free_bytes, Some(999_999_999));
        assert_eq!(gate.min_required_free_bytes, 1_000_000_000);
        assert!(gate.no_env_file_read);
        assert!(gate.no_llm);
        assert!(gate.no_paid_calls);
    }

    #[test]
    fn embed_disk_space_gate_json_fails_closed_when_unknown() {
        let gate =
            embed_disk_space_gate_from_result(Err("disk probe unavailable".to_string()), 1_000);
        assert!(!gate.ok);
        assert_eq!(gate.status, "unknown");
        assert!(gate.blocks_embedding_apply);
        assert_eq!(gate.free_bytes, None);
        assert_eq!(gate.min_required_free_bytes, 1_000);
    }

    #[test]
    fn min_embed_free_bytes_env_override_is_bounded_to_nonnegative_number() {
        std::env::set_var("RECALL_MIN_EMBED_FREE_GB", "2.5");
        assert_eq!(min_embed_free_bytes_from_env(), 2_500_000_000);
        std::env::set_var("RECALL_MIN_EMBED_FREE_GB", "-1");
        assert_eq!(min_embed_free_bytes_from_env(), 1_000_000_000);
        std::env::remove_var("RECALL_MIN_EMBED_FREE_GB");
    }

    #[test]
    fn parse_timeout_seconds_defaults_and_clamps() {
        assert_eq!(
            parse_timeout_seconds(None),
            DEFAULT_EMBED_DOCTOR_TIMEOUT_SECS
        );
        assert_eq!(parse_timeout_seconds(Some("0".to_string())), 1);
        assert_eq!(parse_timeout_seconds(Some("9999".to_string())), 600);
        assert_eq!(
            parse_timeout_seconds(Some("not-a-number".to_string())),
            DEFAULT_EMBED_DOCTOR_TIMEOUT_SECS
        );
    }

    #[test]
    fn latest_surrogate_manifests_discovers_both_result_roots() {
        let root = std::env::temp_dir().join(format!(
            "recall-surrogate-manifest-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let singular = root
            .join("eval")
            .join("results")
            .join("secret_surrogate")
            .join("run-a");
        let plural = root
            .join("eval")
            .join("results")
            .join("secret_surrogates")
            .join("run-b")
            .join("nested");
        fs::create_dir_all(&singular).unwrap();
        fs::create_dir_all(&plural).unwrap();
        fs::write(singular.join("surrogate_manifest.json"), "{}").unwrap();
        fs::write(plural.join("surrogate_manifest.json"), "{}").unwrap();
        fs::write(plural.join("not_a_manifest.json"), "{}").unwrap();

        let manifests = latest_surrogate_manifests(&root, 16);

        assert_eq!(manifests.len(), 2);
        assert!(manifests.iter().all(|path| {
            path.file_name().and_then(|name| name.to_str()) == Some("surrogate_manifest.json")
        }));
    }

    #[test]
    fn coverage_pct_full_coverage() {
        assert_eq!(coverage_pct(100_998, 100_998), 100.0);
    }

    #[test]
    fn coverage_meets_exactly_at_threshold_passes() {
        assert!(coverage_meets(80.0, 80.0), "pct == threshold must pass");
    }

    #[test]
    fn coverage_meets_just_below_threshold_fails() {
        let pct = coverage_pct(799, 1000); // 79.9%
        assert!(!coverage_meets(pct, 80.0), "79.9% must fail an 80% gate");
    }

    #[test]
    fn coverage_meets_zero_docs_fails_nonzero_threshold() {
        let pct = coverage_pct(0, 0); // 0.0
        assert!(
            !coverage_meets(pct, 1.0),
            "empty store must fail any threshold > 0"
        );
    }

    #[test]
    fn coverage_meets_zero_threshold_always_passes() {
        assert!(coverage_meets(0.0, 0.0));
        assert!(coverage_meets(50.0, 0.0));
    }

    #[test]
    fn coverage_meets_live_floor_passes_999_gate() {
        // The live store sits at 99.999% — a 99.9% gate must pass.
        let pct = coverage_pct(100_998, 100_999);
        assert!(coverage_meets(pct, 99.9));
    }

    #[test]
    fn coverage_pct_zero_docs_no_panic() {
        assert_eq!(coverage_pct(0, 0), 0.0);
    }

    #[test]
    fn coverage_pct_partial() {
        assert_eq!(coverage_pct(50, 100), 50.0);
    }

    // ── CI smoke tests — hermetic :memory: coverage freeze guard ─────────────

    #[test]
    fn smoke_coverage_full_store() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        for i in 0..5 {
            let path = format!("doc{}.md", i);
            let id = store
                .insert(&path, &format!("Doc {}", i), "body", "cli")
                .unwrap();
            store.upsert_embedding(id, &dummy, 2).unwrap();
        }
        let emb = store.count_embeddings().unwrap() as usize;
        let docs = store.count_valid().unwrap() as usize;
        assert_eq!(
            coverage_pct(emb, docs),
            100.0,
            "all docs embedded — coverage must be 100%"
        );
    }

    #[test]
    fn smoke_coverage_superseded_doc_excluded() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        let id1 = store.insert("a.md", "A", "body a", "cli").unwrap();
        let id2 = store.insert("b.md", "B", "body b", "cli").unwrap();
        let id3 = store.insert("c.md", "C", "body c", "cli").unwrap();
        store.upsert_embedding(id1, &dummy, 2).unwrap();
        store.upsert_embedding(id2, &dummy, 2).unwrap();
        store.upsert_embedding(id3, &dummy, 2).unwrap();

        // Tombstone doc 1 (embedding row remains as orphan).
        store.supersede_id(id1).unwrap();

        // count_embeddings JOINs on valid_to IS NULL — orphan excluded from numerator.
        let emb = store.count_embeddings().unwrap() as usize; // 2, not 3
        let docs = store.count_valid().unwrap() as usize; // 2, not 3
        assert_eq!(
            coverage_pct(emb, docs),
            100.0,
            "superseded doc + orphan embedding must not distort coverage"
        );

        // Contrast: an un-embedded valid doc must drop coverage below 100%.
        let id4 = store.insert("d.md", "D", "body d", "cli").unwrap();
        // id4 intentionally has no embedding inserted.
        let docs_plus_one = store.count_valid().unwrap() as usize;
        assert!(
            coverage_pct(emb, docs_plus_one) < 100.0,
            "un-embedded valid doc (id={}) must cause sub-100% coverage",
            id4
        );
    }

    #[test]
    fn smoke_coverage_zero_docs() {
        // Empty store must return 0.0 without panic.
        assert_eq!(coverage_pct(0, 0), 0.0);
    }

    /// JSON output must contain all six required keys with correct types.
    #[test]
    fn status_json_serializes_all_fields() {
        let payload = StatusJson {
            docs_valid: 1000,
            docs_total: 1005,
            superseded: 5,
            chunk_embedded_docs: 1000,
            legacy_embeddings: 1000,
            embeddings: 1000,
            coverage_pct: 100.0,
            db_size_mb: 42.5,
            rg_available: true,
        };
        let raw = serde_json::to_string(&payload).expect("serialization must not fail");
        let v: serde_json::Value = serde_json::from_str(&raw).expect("must be valid JSON");

        assert_eq!(v["docs_valid"], 1000);
        assert_eq!(v["docs_total"], 1005);
        assert_eq!(v["superseded"], 5);
        assert_eq!(v["embeddings"], 1000);
        assert_eq!(v["coverage_pct"], 100.0);
        assert_eq!(v["db_size_mb"], 42.5);
    }

    /// When docs_valid == embeddings, coverage_pct must be exactly 100.0 in the JSON.
    #[test]
    fn status_json_coverage_100_when_fully_embedded() {
        let payload = StatusJson {
            docs_valid: 50,
            docs_total: 50,
            superseded: 0,
            chunk_embedded_docs: 50,
            legacy_embeddings: 50,
            embeddings: 50,
            coverage_pct: coverage_pct(50, 50),
            db_size_mb: 1.0,
            rg_available: true,
        };
        let raw = serde_json::to_string(&payload).expect("serialization must not fail");
        let v: serde_json::Value = serde_json::from_str(&raw).expect("must be valid JSON");
        assert_eq!(v["coverage_pct"], 100.0);
    }

    #[test]
    fn health_doctor_json_serializes_bge_vector_stats() {
        let payload = HealthDoctorJson {
            ok: true,
            recall_status: StatusJson {
                docs_valid: 100,
                docs_total: 105,
                superseded: 5,
                chunk_embedded_docs: 80,
                legacy_embeddings: 80,
                embeddings: 80,
                coverage_pct: 80.0,
                db_size_mb: 12.0,
                rg_available: true,
            },
            bge_vectors: BgeVectorStatsJson {
                stored_valid_embeddings: 80,
                queryable_embeddings: 72,
                excluded_embeddings: 8,
            },
            embed_doctor: EmbedDoctorJson {
                ok: true,
                timeout_seconds: 20,
                elapsed_ms: Some(123),
                ort_version: Some("1.24.4".to_string()),
                error: None,
            },
            no_llm: true,
            no_paid_calls: true,
        };
        let raw = serde_json::to_string(&payload).expect("serialization must not fail");
        let v: serde_json::Value = serde_json::from_str(&raw).expect("must be valid JSON");

        assert_eq!(v["bge_vectors"]["stored_valid_embeddings"], 80);
        assert_eq!(v["bge_vectors"]["queryable_embeddings"], 72);
        assert_eq!(v["bge_vectors"]["excluded_embeddings"], 8);
        assert_eq!(v["no_llm"], true);
        assert_eq!(v["no_paid_calls"], true);
    }

    // ── Strengthened regression tests (WP7 guard) ─────────────────────────────

    #[cfg(feature = "ops-health")]
    #[test]
    fn gateway_mcp_candidate_ports_prefers_env_and_dedupes() {
        assert_eq!(
            gateway_mcp_candidate_ports_from_env_port(Some(9986)),
            vec![9986, 9876]
        );
        assert_eq!(
            gateway_mcp_candidate_ports_from_env_port(Some(9999)),
            vec![9999, 9876, 9986]
        );
        assert_eq!(
            gateway_mcp_candidate_ports_from_env_port(None),
            vec![9876, 9986]
        );
    }

    #[cfg(feature = "ops-health")]
    #[test]
    fn select_gateway_mcp_probe_falls_back_to_first_non_listening_probe() {
        let probes = vec![
            GatewayMcpProbe {
                port: 9876,
                mcp_status: None,
                mcp_health_status: None,
                events_status: None,
            },
            GatewayMcpProbe {
                port: 9986,
                mcp_status: None,
                mcp_health_status: None,
                events_status: None,
            },
        ];

        let selected = select_gateway_mcp_probe(&probes).expect("fallback probe expected");

        assert_eq!(selected.port, 9876);
        assert!(!gateway_mcp_port_listening(selected));
    }

    #[cfg(feature = "ops-health")]
    #[test]
    fn select_gateway_mcp_probe_prefers_auth_gate_over_listening_probe() {
        let probes = vec![
            GatewayMcpProbe {
                port: 9876,
                mcp_status: Some(200),
                mcp_health_status: Some(200),
                events_status: Some(200),
            },
            GatewayMcpProbe {
                port: 9986,
                mcp_status: Some(404),
                mcp_health_status: Some(404),
                events_status: Some(401),
            },
        ];

        let selected = select_gateway_mcp_probe(&probes).expect("auth probe expected");

        assert_eq!(selected.port, 9986);
        assert!(gateway_mcp_auth_gate_observed(selected));
    }

    #[test]
    fn memory_doctor_json_serializes_without_raw_contexts() {
        let payload = MemoryDoctorJson {
            ok: false,
            status: "degraded".to_string(),
            recall_status: StatusJson {
                docs_valid: 100,
                docs_total: 105,
                superseded: 5,
                chunk_embedded_docs: 80,
                legacy_embeddings: 80,
                embeddings: 80,
                coverage_pct: 80.0,
                db_size_mb: 12.0,
                rg_available: true,
            },
            active_retrieval_modes: ActiveRetrievalModesJson {
                denoise_mode: "off".to_string(),
                denoise_mode_set: false,
                fusion_mode: "rrf".to_string(),
                fusion_mode_set: false,
                rrf_k: 60.0,
                rrf_k_set: false,
                min_fusion_depth: 20,
                min_fusion_depth_set: false,
                fusion_depth_multiplier: 4,
                fusion_depth_multiplier_set: false,
            },
            coverage_artifact: CoverageArtifactJson {
                present: true,
                run_id: Some("coverage".to_string()),
                summary_json: Some("coverage_summary.json".to_string()),
                recall_embedding_coverage_pct: Some(80.0),
                recall_unembedded_valid_docs: Some(20),
                recall_missing_target_docs: Some(0),
                router_missing_target_docs: Some(0),
                router_stale_target_docs: Some(0),
                router_stale_hard_target_docs: Some(0),
                router_stale_live_write_target_docs: Some(0),
                secret_risk_target_docs: Some(0),
            },
            coverage_delta_refresh: CoverageDeltaRefreshArtifactJson {
                present: true,
                run_id: Some("delta".to_string()),
                summary_json: Some("coverage_delta_refresh_summary.json".to_string()),
                status: Some("pass".to_string()),
                ok: Some(true),
                baseline_coverage_run_id: Some("coverage".to_string()),
                full_audit_skipped: Some(true),
                full_audit_skip_reason: Some("low_disk_space".to_string()),
                disk_free_gb: Some(0.5),
                min_full_audit_free_gb: Some(1.0),
                router_indexer_target_coverage_pct: Some(99.9963),
                recall_indexer_target_coverage_pct: Some(99.9963),
                recall_valid_docs: Some(100),
                recall_embeddings: Some(80),
                recall_unembedded_valid_docs: Some(20),
                recall_duplicate_valid_paths: Some(1),
                remaining_recall_missing: Some(2),
                remaining_router_missing: Some(2),
                remaining_router_stale: Some(3),
                remaining_router_hard_stale: Some(0),
                remaining_router_live_write_stale: Some(3),
                repaired_recall_missing: Some(3),
                repaired_router_missing: Some(3),
                repaired_router_stale: Some(5),
                redacted_surrogate_enabled: Some(true),
                redacted_surrogate_manifest_count: Some(1),
                redacted_surrogate_unsafe_manifest_count: Some(0),
                redacted_surrogate_written_count: Some(2),
                redacted_surrogate_pattern_hits_redacted: Some(7),
                router_missing_docs_with_redacted_surrogate: Some(2),
                recall_missing_docs_with_redacted_surrogate: Some(2),
                router_missing_target_docs_without_redacted_surrogate: Some(0),
                recall_missing_target_docs_without_redacted_surrogate: Some(0),
                raw_contexts_included: Some(false),
                raw_paths_included: Some(false),
                raw_secret_values_included: Some(false),
                file_contents_read: Some(false),
                no_llm: Some(true),
                no_paid_calls: Some(true),
                database_mutation_expected: Some(false),
            },
            embedding_coverage_freshness: EmbeddingCoverageFreshnessJson {
                authoritative_source: "recall_status_current_db".to_string(),
                artifact_source: "coverage_delta_refresh".to_string(),
                artifact_run_id: Some("delta".to_string()),
                current_db_unembedded_valid_docs: 20,
                artifact_unembedded_valid_docs: Some(20),
                artifact_unembedded_delta_vs_current_db: Some(0),
                artifact_matches_current_db: Some(true),
            },
            ragas_smoke: RagasArtifactJson {
                present: true,
                run_id: Some("ragas".to_string()),
                summary_json: Some("ragas_smoke_summary.json".to_string()),
                passed: Some(true),
                pass_count: Some(10),
                failed_count: Some(0),
                secret_leak_count: Some(0),
                recall_at_5: Some(1.0),
                precision_at_5: Some(0.2),
                mrr: Some(1.0),
                ndcg_at_5: Some(1.0),
                latency_p50_ms: Some(100.0),
                latency_p95_ms: Some(200.0),
            },
            latest_release_benchmark: BenchArtifactJson {
                present: true,
                run_id: Some("bench".to_string()),
                summary_json: Some("bench.json".to_string()),
                denoise_mode: Some("off".to_string()),
                fusion_mode: Some("rrf".to_string()),
                rrf_k: Some(60.0),
                top_k: Some(5),
                positive_fixtures: Some(45),
                fts5_hits: Some(18),
                bge_hits: Some(23),
                fused_hits: Some(26),
                fused_recall: Some(0.5777),
                latency_p50_ms: Some(1000.0),
                latency_p95_ms: Some(2000.0),
                raw_contexts_included: Some(false),
            },
            release_benchmark_high_water: BenchHighWaterJson {
                present: true,
                run_id: Some("bench".to_string()),
                summary_json: Some("bench.json".to_string()),
                fused_hits: Some(26),
                fused_recall: Some(0.5777),
                denoise_mode: Some("off".to_string()),
            },
            release_mode_sweep: ReleaseModeSweepArtifactJson {
                present: true,
                run_id: Some("sweep".to_string()),
                summary_json: Some("release_mode_sweep_summary.json".to_string()),
                status: Some("no_improvement".to_string()),
                blocker: None,
                reason: None,
                variant_count: Some(2),
                baseline_fused_hits: Some(26),
                best_fused_hits: Some(26),
                high_water_fused_hits: Some(26),
                disk_free_gb: Some(10.0),
                min_required_free_gb: Some(1.0),
                raw_contexts_included: Some(false),
                no_llm: Some(true),
                no_paid_calls: Some(true),
                no_env_file_read: Some(true),
                database_mutation_expected: Some(false),
            },
            vault_only_benchmark: VaultOnlyBenchmarkArtifactJson {
                present: true,
                run_id: Some("vault_only".to_string()),
                summary_json: Some("vault_only_benchmark_summary.json".to_string()),
                source_bench_run_id: Some("bench".to_string()),
                source_gold_fixture_count: Some(45),
                vault_only_fixture_count: Some(26),
                excluded_fixture_count: Some(19),
                fts5_hits: Some(18),
                fts5_recall: Some(0.6923),
                bge_hits: Some(21),
                bge_recall: Some(0.8077),
                fused_hits: Some(24),
                fused_recall: Some(0.9231),
                fused_mrr: Some(0.7179),
                fused_ndcg_at_5: Some(0.7686),
                latency_p50_ms: Some(755.126),
                latency_p95_ms: Some(1079.3125),
                raw_contexts_included: Some(false),
                raw_paths_included: Some(false),
                fixture_ids_included: Some(false),
                secret_values_included: Some(false),
                no_llm: Some(true),
                no_paid_calls: Some(true),
            },
            embedding_benchmark_gate: EmbeddingBenchmarkGateJson {
                present: true,
                ok: true,
                status: "pass".to_string(),
                blocks_embedding_apply: false,
                reason: None,
                latest_run_id: Some("bench".to_string()),
                high_water_run_id: Some("bench".to_string()),
                latest_fused_hits: Some(26),
                high_water_fused_hits: Some(26),
                hit_gap: Some(0),
                latest_fused_recall: Some(0.5777),
                high_water_fused_recall: Some(0.5777),
                latest_raw_contexts_included: Some(false),
                no_llm: true,
                no_paid_calls: true,
            },
            embed_disk_space_gate: EmbedDiskSpaceGateJson {
                present: true,
                ok: false,
                status: "blocking".to_string(),
                blocks_embedding_apply: true,
                reason: Some("free disk 0.500 GB is below embed apply floor 1.000 GB".to_string()),
                free_bytes: Some(500_000_000),
                free_gb: Some(0.5),
                min_required_free_bytes: 1_000_000_000,
                min_required_free_gb: 1.0,
                no_env_file_read: true,
                no_llm: true,
                no_paid_calls: true,
            },
            #[cfg(feature = "ops-health")]
            memory_broker_db_health: MemoryBrokerDbHealthJson {
                ok: true,
                source: Some("postgres-direct".to_string()),
                events: Some(110),
                chunks: Some(110),
                cards: Some(110),
                embeddings: Some(110),
                pending_embeddings: Some(0),
                latest_event_at: Some("2026-06-12T04:29:08.475882+00:00".to_string()),
                latest_chunk_at: Some("2026-06-12T04:29:08.475882+00:00".to_string()),
                latest_card_at: Some("2026-06-12T04:29:08.475882+00:00".to_string()),
                control_plane_http_status: Some(404),
                postgres_direct_ok: Some(true),
                error_class: None,
                error: None,
            },
            #[cfg(feature = "ops-health")]
            tailscale_local_health: TailscaleLocalHealthJson {
                ok: true,
                installed: true,
                backend_state: Some("Running".to_string()),
                self_online: Some(true),
                peer_count: Some(28),
                online_peer_count: Some(26),
                has_local_tailscale_ip: Some(true),
                error: None,
            },
            #[cfg(feature = "ops-health")]
            gateway_local_health: GatewayLocalHealthJson {
                ok: true,
                mcp_port: 9876,
                rest_port: 8810,
                mcp_code_present: true,
                rest_code_present: true,
                mcp_port_listening: true,
                mcp_status: Some(401),
                mcp_auth_required: Some(true),
                mcp_health_status: Some(404),
                events_status: Some(401),
                events_auth_required: Some(true),
                sse_bearer_configured: true,
                rest_port_listening: false,
                rest_health_status: None,
                rest_health_ok: None,
                error: None,
            },
            targeted_path_probe: TargetedPathProbeArtifactJson {
                present: true,
                run_id: Some("probe".to_string()),
                summary_json: Some("coverage_path_probe.json".to_string()),
                ok: Some(false),
                status: Some("degraded".to_string()),
                mode: Some("targeted_path_coverage_probe".to_string()),
                selected_count: Some(12),
                router_missing_count: Some(3),
                router_stale_count: Some(3),
                recall_missing_count: Some(3),
                recall_unembedded_count: Some(12),
                raw_paths_included: Some(false),
                raw_contexts_included: Some(false),
                file_contents_read: Some(false),
                require_embedded: Some(false),
                recall_freshness_verifiable: Some(false),
                elapsed_ms: Some(50),
            },
            embedding_backlog: EmbeddingBacklogArtifactJson {
                present: true,
                summary_json: Some("embedding_backlog_summary.json".to_string()),
                ok: Some(true),
                mode: Some("embedding_backlog_plan".to_string()),
                total_unembedded: Some(12),
                nonrisk_unembedded: Some(10),
                path_risk_unembedded: Some(2),
                router_eligible_pct: Some(100.0),
                nonrisk_pct: Some(83.3333),
                path_risk_pct: Some(16.6667),
                selected_pct_of_total: Some(83.3333),
                selected_pct_of_router_eligible: Some(83.3333),
                requested_size: Some(64),
                selected_count: Some(10),
                max_per_prefix: Some(8),
                raw_paths_included: Some(false),
                raw_contexts_included: Some(false),
                file_contents_read: Some(false),
                path_list_contains_raw_paths: Some(false),
                current_filter_enabled: Some(true),
                current_unembedded_paths: Some(12),
                stale_artifact_rows_removed: Some(3),
            },
            gates: MemoryDoctorGatesJson {
                recall_store_present: true,
                recall_index_has_docs: true,
                recall_embeddings_complete: false,
                coverage_artifact_present: true,
                coverage_delta_refresh_present: true,
                coverage_delta_refresh_redacted: true,
                ragas_smoke_green: true,
                secret_leak_rate_zero: true,
                release_benchmark_present: true,
                release_mode_sweep_present: true,
                release_mode_sweep_not_blocked: true,
                latest_bench_no_raw_contexts: true,
                vault_only_benchmark_present: true,
                vault_only_benchmark_redacted: true,
                embedding_benchmark_gate_present: true,
                embedding_benchmark_gate_passed: true,
                embed_disk_space_gate_present: true,
                embed_disk_space_gate_passed: false,
                #[cfg(feature = "ops-health")]
                memory_broker_db_proven: true,
                #[cfg(feature = "ops-health")]
                tailscale_local_ready: true,
                #[cfg(feature = "ops-health")]
                gateway_local_ready: true,
                targeted_path_probe_present: true,
                targeted_path_probe_redacted: true,
                embedding_backlog_present: true,
                embedding_backlog_redacted: true,
            },
            raw_contexts_included: false,
            no_llm: true,
            no_paid_calls: true,
        };
        let raw = serde_json::to_string(&payload).expect("serialization must not fail");
        let v: serde_json::Value = serde_json::from_str(&raw).expect("must be valid JSON");

        assert_eq!(v["status"], "degraded");
        assert_eq!(v["active_retrieval_modes"]["denoise_mode"], "off");
        assert_eq!(v["active_retrieval_modes"]["min_fusion_depth"], 20);
        assert_eq!(v["active_retrieval_modes"]["fusion_depth_multiplier"], 4);
        assert_eq!(
            v["embedding_coverage_freshness"]["authoritative_source"],
            "recall_status_current_db"
        );
        assert_eq!(
            v["embedding_coverage_freshness"]["artifact_source"],
            "coverage_delta_refresh"
        );
        assert_eq!(
            v["embedding_coverage_freshness"]["artifact_run_id"],
            "delta"
        );
        assert_eq!(
            v["embedding_coverage_freshness"]["current_db_unembedded_valid_docs"],
            20
        );
        assert_eq!(
            v["embedding_coverage_freshness"]["artifact_unembedded_delta_vs_current_db"],
            0
        );
        assert_eq!(
            v["embedding_coverage_freshness"]["artifact_matches_current_db"],
            true
        );
        assert_eq!(v["coverage_delta_refresh"]["status"], "pass");
        assert_eq!(
            v["coverage_delta_refresh"]["router_indexer_target_coverage_pct"],
            99.9963
        );
        assert_eq!(
            v["coverage_delta_refresh"]["recall_indexer_target_coverage_pct"],
            99.9963
        );
        assert_eq!(v["coverage_delta_refresh"]["remaining_recall_missing"], 2);
        assert_eq!(
            v["coverage_delta_refresh"]["remaining_router_hard_stale"],
            0
        );
        assert_eq!(
            v["coverage_delta_refresh"]["router_missing_docs_with_redacted_surrogate"],
            2
        );
        assert_eq!(
            v["coverage_delta_refresh"]["router_missing_target_docs_without_redacted_surrogate"],
            0
        );
        assert_eq!(v["coverage_delta_refresh"]["raw_paths_included"], false);
        assert_eq!(v["coverage_delta_refresh"]["raw_contexts_included"], false);
        assert_eq!(
            v["coverage_delta_refresh"]["raw_secret_values_included"],
            false
        );
        assert_eq!(v["coverage_delta_refresh"]["file_contents_read"], false);
        assert_eq!(v["gates"]["coverage_delta_refresh_present"], true);
        assert_eq!(v["gates"]["coverage_delta_refresh_redacted"], true);
        assert_eq!(v["latest_release_benchmark"]["fused_hits"], 26);
        assert_eq!(v["release_benchmark_high_water"]["fused_hits"], 26);
        assert_eq!(v["release_mode_sweep"]["status"], "no_improvement");
        assert_eq!(v["release_mode_sweep"]["high_water_fused_hits"], 26);
        assert_eq!(v["gates"]["release_mode_sweep_present"], true);
        assert_eq!(v["gates"]["release_mode_sweep_not_blocked"], true);
        assert_eq!(v["vault_only_benchmark"]["fused_hits"], 24);
        assert_eq!(v["vault_only_benchmark"]["vault_only_fixture_count"], 26);
        assert_eq!(v["vault_only_benchmark"]["excluded_fixture_count"], 19);
        assert_eq!(v["vault_only_benchmark"]["raw_paths_included"], false);
        assert_eq!(v["vault_only_benchmark"]["raw_contexts_included"], false);
        assert_eq!(v["vault_only_benchmark"]["fixture_ids_included"], false);
        assert_eq!(v["vault_only_benchmark"]["secret_values_included"], false);
        assert_eq!(v["gates"]["vault_only_benchmark_present"], true);
        assert_eq!(v["gates"]["vault_only_benchmark_redacted"], true);
        assert_eq!(v["embedding_benchmark_gate"]["status"], "pass");
        assert_eq!(
            v["embedding_benchmark_gate"]["blocks_embedding_apply"],
            false
        );
        assert_eq!(v["embedding_benchmark_gate"]["hit_gap"], 0);
        assert_eq!(
            v["embedding_benchmark_gate"]["latest_raw_contexts_included"],
            false
        );
        assert_eq!(v["gates"]["embedding_benchmark_gate_present"], true);
        assert_eq!(v["gates"]["embedding_benchmark_gate_passed"], true);
        assert_eq!(v["embed_disk_space_gate"]["present"], true);
        assert_eq!(v["embed_disk_space_gate"]["status"], "blocking");
        assert_eq!(v["embed_disk_space_gate"]["blocks_embedding_apply"], true);
        assert_eq!(v["embed_disk_space_gate"]["free_bytes"], 500_000_000);
        assert_eq!(
            v["embed_disk_space_gate"]["min_required_free_bytes"],
            1_000_000_000
        );
        assert_eq!(v["embed_disk_space_gate"]["no_env_file_read"], true);
        assert_eq!(v["embed_disk_space_gate"]["no_llm"], true);
        assert_eq!(v["embed_disk_space_gate"]["no_paid_calls"], true);
        assert_eq!(v["gates"]["embed_disk_space_gate_present"], true);
        assert_eq!(v["gates"]["embed_disk_space_gate_passed"], false);
        #[cfg(feature = "ops-health")]
        assert_eq!(v["memory_broker_db_health"]["ok"], true);
        #[cfg(feature = "ops-health")]
        assert_eq!(v["memory_broker_db_health"]["source"], "postgres-direct");
        #[cfg(feature = "ops-health")]
        assert_eq!(v["gates"]["memory_broker_db_proven"], true);
        #[cfg(feature = "ops-health")]
        assert_eq!(v["tailscale_local_health"]["ok"], true);
        #[cfg(feature = "ops-health")]
        assert_eq!(v["tailscale_local_health"]["backend_state"], "Running");
        #[cfg(feature = "ops-health")]
        assert_eq!(v["gates"]["tailscale_local_ready"], true);
        #[cfg(feature = "ops-health")]
        assert_eq!(v["gateway_local_health"]["ok"], true);
        #[cfg(feature = "ops-health")]
        assert_eq!(v["gateway_local_health"]["mcp_port"], 9876);
        #[cfg(feature = "ops-health")]
        assert_eq!(v["gateway_local_health"]["mcp_status"], 401);
        #[cfg(feature = "ops-health")]
        assert_eq!(v["gateway_local_health"]["events_status"], 401);
        #[cfg(feature = "ops-health")]
        assert_eq!(v["gateway_local_health"]["events_auth_required"], true);
        #[cfg(feature = "ops-health")]
        assert_eq!(v["gateway_local_health"]["sse_bearer_configured"], true);
        #[cfg(feature = "ops-health")]
        assert_eq!(v["gates"]["gateway_local_ready"], true);
        assert_eq!(v["targeted_path_probe"]["raw_paths_included"], false);
        assert_eq!(v["targeted_path_probe"]["raw_contexts_included"], false);
        assert_eq!(v["targeted_path_probe"]["file_contents_read"], false);
        assert_eq!(v["gates"]["targeted_path_probe_present"], true);
        assert_eq!(v["gates"]["targeted_path_probe_redacted"], true);
        assert_eq!(v["embedding_backlog"]["total_unembedded"], 12);
        assert_eq!(v["embedding_backlog"]["selected_count"], 10);
        assert_eq!(v["embedding_backlog"]["nonrisk_pct"], 83.3333);
        assert_eq!(v["embedding_backlog"]["path_risk_pct"], 16.6667);
        assert_eq!(
            v["embedding_backlog"]["selected_pct_of_router_eligible"],
            83.3333
        );
        assert_eq!(v["embedding_backlog"]["raw_paths_included"], false);
        assert_eq!(v["embedding_backlog"]["raw_contexts_included"], false);
        assert_eq!(v["embedding_backlog"]["file_contents_read"], false);
        assert_eq!(
            v["embedding_backlog"]["path_list_contains_raw_paths"],
            false
        );
        assert_eq!(v["gates"]["embedding_backlog_present"], true);
        assert_eq!(v["gates"]["embedding_backlog_redacted"], true);
        assert_eq!(v["raw_contexts_included"], false);
        assert_eq!(v["no_llm"], true);
        assert_eq!(v["no_paid_calls"], true);
        assert!(!raw.contains("snippet"));
        assert!(!raw.contains("body"));
    }

    #[test]
    fn embedding_coverage_freshness_uses_current_db_as_authority() {
        let recall_status = StatusJson {
            docs_valid: 100,
            docs_total: 105,
            superseded: 5,
            chunk_embedded_docs: 75,
            legacy_embeddings: 75,
            embeddings: 75,
            coverage_pct: 75.0,
            db_size_mb: 12.0,
            rg_available: true,
        };
        let coverage = CoverageArtifactJson {
            present: true,
            run_id: Some("coverage".to_string()),
            summary_json: Some("coverage_summary.json".to_string()),
            recall_embedding_coverage_pct: Some(75.0),
            recall_unembedded_valid_docs: Some(20),
            recall_missing_target_docs: Some(0),
            router_missing_target_docs: Some(0),
            router_stale_target_docs: Some(0),
            router_stale_hard_target_docs: Some(0),
            router_stale_live_write_target_docs: Some(0),
            secret_risk_target_docs: Some(0),
        };

        let freshness = embedding_coverage_freshness_from(&recall_status, &coverage, None);

        assert_eq!(freshness.current_db_unembedded_valid_docs, 25);
        assert_eq!(freshness.artifact_source, "coverage_audit");
        assert_eq!(freshness.artifact_run_id.as_deref(), Some("coverage"));
        assert_eq!(freshness.artifact_unembedded_valid_docs, Some(20));
        assert_eq!(freshness.artifact_unembedded_delta_vs_current_db, Some(-5));
        assert_eq!(freshness.artifact_matches_current_db, Some(false));
    }

    #[test]
    fn embedding_coverage_freshness_prefers_delta_refresh_when_present() {
        let recall_status = StatusJson {
            docs_valid: 100,
            docs_total: 105,
            superseded: 5,
            chunk_embedded_docs: 75,
            legacy_embeddings: 75,
            embeddings: 75,
            coverage_pct: 75.0,
            db_size_mb: 12.0,
            rg_available: true,
        };
        let coverage = CoverageArtifactJson {
            present: true,
            run_id: Some("coverage".to_string()),
            summary_json: Some("coverage_summary.json".to_string()),
            recall_embedding_coverage_pct: Some(75.0),
            recall_unembedded_valid_docs: Some(20),
            recall_missing_target_docs: Some(0),
            router_missing_target_docs: Some(0),
            router_stale_target_docs: Some(0),
            router_stale_hard_target_docs: Some(0),
            router_stale_live_write_target_docs: Some(0),
            secret_risk_target_docs: Some(0),
        };
        let delta = CoverageDeltaRefreshArtifactJson {
            present: true,
            run_id: Some("delta".to_string()),
            summary_json: Some("coverage_delta_refresh_summary.json".to_string()),
            status: Some("pass".to_string()),
            ok: Some(true),
            baseline_coverage_run_id: Some("coverage".to_string()),
            full_audit_skipped: Some(true),
            full_audit_skip_reason: Some("low_disk_space".to_string()),
            disk_free_gb: Some(0.8),
            min_full_audit_free_gb: Some(1.0),
            router_indexer_target_coverage_pct: Some(99.0),
            recall_indexer_target_coverage_pct: Some(99.0),
            recall_valid_docs: Some(100),
            recall_embeddings: Some(75),
            recall_unembedded_valid_docs: Some(25),
            recall_duplicate_valid_paths: Some(0),
            remaining_recall_missing: Some(0),
            remaining_router_missing: Some(0),
            remaining_router_stale: Some(0),
            remaining_router_hard_stale: Some(0),
            remaining_router_live_write_stale: Some(0),
            repaired_recall_missing: Some(0),
            repaired_router_missing: Some(0),
            repaired_router_stale: Some(0),
            redacted_surrogate_enabled: Some(true),
            redacted_surrogate_manifest_count: Some(1),
            redacted_surrogate_unsafe_manifest_count: Some(0),
            redacted_surrogate_written_count: Some(0),
            redacted_surrogate_pattern_hits_redacted: Some(0),
            router_missing_docs_with_redacted_surrogate: Some(0),
            recall_missing_docs_with_redacted_surrogate: Some(0),
            router_missing_target_docs_without_redacted_surrogate: Some(0),
            recall_missing_target_docs_without_redacted_surrogate: Some(0),
            raw_contexts_included: Some(false),
            raw_paths_included: Some(false),
            raw_secret_values_included: Some(false),
            file_contents_read: Some(false),
            no_llm: Some(true),
            no_paid_calls: Some(true),
            database_mutation_expected: Some(false),
        };

        let freshness = embedding_coverage_freshness_from(&recall_status, &coverage, Some(&delta));

        assert_eq!(freshness.current_db_unembedded_valid_docs, 25);
        assert_eq!(freshness.artifact_source, "coverage_delta_refresh");
        assert_eq!(freshness.artifact_run_id.as_deref(), Some("delta"));
        assert_eq!(freshness.artifact_unembedded_valid_docs, Some(25));
        assert_eq!(freshness.artifact_unembedded_delta_vs_current_db, Some(0));
        assert_eq!(freshness.artifact_matches_current_db, Some(true));
    }

    #[test]
    fn targeted_path_probe_artifact_accepts_router_stale_probe_summary() {
        let value = serde_json::json!({
            "run_id": "20260626T173127Z",
            "ok": true,
            "mode": "targeted_path_coverage_probe",
            "router_indexed_count": 10,
            "router_missing_count": 0,
            "router_stale_count": 0,
            "recall_missing_count": 0,
            "recall_unembedded_count": 0,
            "raw_paths_included": false,
            "raw_contexts_included": false,
            "file_contents_read": false,
            "require_embedded": true,
            "recall_freshness_verifiable": false,
            "elapsed_ms": 3771
        });
        let artifact =
            targeted_path_probe_artifact_from(Some(("router_stale_probe.json".into(), value)));

        assert!(artifact.present);
        assert_eq!(artifact.run_id.as_deref(), Some("20260626T173127Z"));
        assert_eq!(artifact.selected_count, Some(10));
        assert_eq!(artifact.raw_paths_included, Some(false));
        assert_eq!(artifact.raw_contexts_included, Some(false));
        assert_eq!(artifact.file_contents_read, Some(false));
        assert_eq!(artifact.require_embedded, Some(true));
    }

    fn bench_artifact(fused_hits: Option<i64>, redacted: Option<bool>) -> BenchArtifactJson {
        BenchArtifactJson {
            present: true,
            run_id: Some("bench_latest".to_string()),
            summary_json: Some("bench_latest.json".to_string()),
            denoise_mode: Some("ops_noise_path_terms".to_string()),
            fusion_mode: Some("rrf4_bge1".to_string()),
            rrf_k: Some(60.0),
            top_k: Some(5),
            positive_fixtures: Some(45),
            fts5_hits: Some(18),
            bge_hits: Some(23),
            fused_hits,
            fused_recall: Some(0.53),
            latency_p50_ms: Some(1000.0),
            latency_p95_ms: Some(2000.0),
            raw_contexts_included: redacted,
        }
    }

    fn high_water(fused_hits: Option<i64>) -> BenchHighWaterJson {
        BenchHighWaterJson {
            present: true,
            run_id: Some("bench_high".to_string()),
            summary_json: Some("bench_high.json".to_string()),
            fused_hits,
            fused_recall: Some(0.57),
            denoise_mode: Some("ops_noise_path_terms".to_string()),
        }
    }

    fn release_sweep(
        best_hits: Option<i64>,
        high_hits: Option<i64>,
    ) -> ReleaseModeSweepArtifactJson {
        ReleaseModeSweepArtifactJson {
            present: true,
            run_id: Some("20260629T020901Z".to_string()),
            summary_json: Some("release_mode_sweep_summary.json".to_string()),
            status: Some("no_improvement".to_string()),
            blocker: None,
            reason: None,
            variant_count: Some(2),
            baseline_fused_hits: high_hits,
            best_fused_hits: best_hits,
            high_water_fused_hits: high_hits,
            disk_free_gb: None,
            min_required_free_gb: None,
            raw_contexts_included: Some(false),
            no_llm: Some(true),
            no_paid_calls: Some(true),
            no_env_file_read: Some(true),
            database_mutation_expected: Some(false),
        }
    }

    #[test]
    fn embedding_benchmark_gate_passes_at_high_water() {
        let latest = bench_artifact(Some(26), Some(false));
        let best = high_water(Some(26));

        assert!(embedding_benchmark_gate_decision(&latest, &best).is_ok());
    }

    #[test]
    fn embedding_benchmark_gate_rejects_regression() {
        let latest = bench_artifact(Some(24), Some(false));
        let best = high_water(Some(26));

        let err = embedding_benchmark_gate_decision(&latest, &best).unwrap_err();
        assert!(err.contains("below high-water"));
        assert!(err.contains("24 < 26"));
    }

    #[test]
    fn embedding_benchmark_gate_rejects_missing_evidence() {
        let latest = bench_artifact(None, Some(false));
        let best = high_water(Some(26));

        let err = embedding_benchmark_gate_decision(&latest, &best).unwrap_err();
        assert!(err.contains("missing fused_hits"));
    }

    #[test]
    fn embedding_benchmark_gate_rejects_unredacted_latest_bench() {
        let latest = bench_artifact(Some(26), Some(true));
        let best = high_water(Some(26));

        let err = embedding_benchmark_gate_decision(&latest, &best).unwrap_err();
        assert!(err.contains("not redacted"));
    }

    #[test]
    fn embedding_benchmark_gate_status_exposes_blocking_gap() {
        let latest = bench_artifact(Some(24), Some(false));
        let best = high_water(Some(26));

        let gate = embedding_benchmark_gate_from(&latest, &best);

        assert!(gate.present);
        assert!(!gate.ok);
        assert_eq!(gate.status, "blocking");
        assert!(gate.blocks_embedding_apply);
        assert_eq!(gate.hit_gap, Some(-2));
        assert_eq!(gate.latest_fused_hits, Some(24));
        assert_eq!(gate.high_water_fused_hits, Some(26));
        assert_eq!(gate.latest_raw_contexts_included, Some(false));
        assert!(gate
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("below high-water"));
        assert!(gate.no_llm);
        assert!(gate.no_paid_calls);
    }

    #[test]
    fn embedding_benchmark_gate_uses_release_sweep_best_when_latest_variant_regressed() {
        let latest = bench_artifact(Some(39), Some(false));
        let best = high_water(Some(40));
        let sweep = release_sweep(Some(40), Some(40));

        let gate = embedding_benchmark_gate_from_sweep(&latest, &best, &sweep);

        assert!(gate.present);
        assert!(gate.ok);
        assert_eq!(gate.status, "pass");
        assert!(!gate.blocks_embedding_apply);
        assert_eq!(
            gate.latest_run_id.as_deref(),
            Some("release_sweep:20260629T020901Z")
        );
        assert_eq!(gate.latest_fused_hits, Some(40));
        assert_eq!(gate.high_water_fused_hits, Some(40));
        assert_eq!(gate.hit_gap, Some(0));
        assert_eq!(gate.latest_raw_contexts_included, Some(false));
    }

    #[test]
    fn embedding_benchmark_gate_ignores_blocked_release_sweep() {
        let latest = bench_artifact(Some(39), Some(false));
        let best = high_water(Some(40));
        let mut sweep = release_sweep(Some(40), Some(40));
        sweep.status = Some("blocked_environment".to_string());

        let gate = embedding_benchmark_gate_from_sweep(&latest, &best, &sweep);

        assert!(!gate.ok);
        assert_eq!(gate.latest_fused_hits, Some(39));
        assert_eq!(gate.hit_gap, Some(-1));
    }

    #[test]
    fn latest_release_sweep_benchmark_evidence_skips_newer_blocked_sweep() {
        let root =
            std::env::temp_dir().join(format!("recall-sweep-evidence-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let usable = root.join("20260629T020901Z");
        let blocked = root.join("20260629T021500Z");
        std::fs::create_dir_all(&usable).unwrap();
        std::fs::create_dir_all(&blocked).unwrap();
        std::fs::write(
            usable.join("release_mode_sweep_summary.json"),
            serde_json::json!({
                "run_id": "20260629T020901Z",
                "status": "no_improvement",
                "best_fused_hits": 40,
                "raw_contexts_included": false,
                "no_llm": true,
                "no_paid_calls": true,
                "database_mutation_expected": false,
                "high_water": {"fused_hits": 40}
            })
            .to_string(),
        )
        .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(
            blocked.join("release_mode_sweep_summary.json"),
            serde_json::json!({
                "run_id": "20260629T021500Z",
                "status": "blocked_environment",
                "blocker": "low_disk_space",
                "best_fused_hits": null,
                "raw_contexts_included": false,
                "no_llm": true,
                "no_paid_calls": true,
                "database_mutation_expected": false,
                "high_water": {"fused_hits": 40}
            })
            .to_string(),
        )
        .unwrap();

        let evidence = latest_release_sweep_benchmark_evidence(root.clone()).unwrap();

        assert_eq!(
            evidence.run_id.as_deref(),
            Some("release_sweep:20260629T020901Z")
        );
        assert_eq!(evidence.fused_hits, Some(40));
        let _ = std::fs::remove_dir_all(root);
    }

    /// 100% exact: 1-doc corpus with 1 embedding must return exactly 100.0,
    /// not 99.999… or 100.00001 due to float arithmetic.
    #[test]
    fn coverage_pct_one_of_one_is_exactly_100() {
        assert_eq!(coverage_pct(1, 1), 100.0);
    }

    /// Rounding to 2 dp: 99_990 embeddings / 100_000 docs = exactly 99.99%.
    /// Confirms the helper does not produce floating-point drift that would
    /// change the formatted output (e.g. 99.98999… → "99.99" or "99.990").
    #[test]
    fn coverage_pct_rounds_cleanly_to_2dp() {
        let pct = coverage_pct(99_990, 100_000);
        assert!(pct < 100.0, "partial coverage must be < 100.0, got {pct}");
        // Format as the status line does: {:.2}%
        let formatted = format!("{pct:.2}");
        assert_eq!(
            formatted, "99.99",
            "expected \"99.99\" at 2 dp, got \"{formatted}\""
        );
    }

    /// Zero embeddings, non-zero docs: coverage is 0.0 (no divide-by-zero,
    /// no negative result).
    #[test]
    fn coverage_pct_zero_embeddings_nonzero_docs() {
        let pct = coverage_pct(0, 42);
        assert_eq!(pct, 0.0);
    }

    /// emb > docs: the helper does NOT clamp — it returns a value > 100.
    /// This test documents the current (unclamped) contract so that any future
    /// change that accidentally clamps or panics is caught immediately.
    /// The status command casts `s.embeddings as usize` before calling this
    /// helper, so in practice emb > docs only occurs from data anomalies.
    #[test]
    fn coverage_pct_emb_exceeds_docs_returns_over_100() {
        let pct = coverage_pct(101, 100);
        assert!(
            pct > 100.0,
            "emb > docs should return > 100.0 (unclamped), got {pct}"
        );
        assert_eq!(pct, 101.0);
    }

    // ── Status coverage-line string-shape guard ────────────────────────────────
    // The status command emits:  "   embeddings    : N / M  ({:.2}%)"
    // These tests assert the parenthesised percentage sub-string that the
    // WP7 freeze-guard parser reads.  If the format template ever changes
    // (e.g. dropping parens, changing precision) these will fail loudly.

    /// Full coverage formats as "(100.00%)" — the parenthesised exact string
    /// that the freeze-guard parser must match.
    #[test]
    fn status_coverage_line_100_percent_exact_shape() {
        let pct = coverage_pct(100, 100);
        let fragment = format!("({pct:.2}%)");
        assert_eq!(fragment, "(100.00%)");
    }

    /// Partial coverage formats with two decimal places inside parens.
    #[test]
    fn status_coverage_line_partial_shape() {
        let pct = coverage_pct(1, 3);
        let fragment = format!("({pct:.2}%)");
        assert_eq!(fragment, "(33.33%)");
    }

    /// Zero-doc corpus formats as "(0.00%)" — guard the zero path too.
    #[test]
    fn status_coverage_line_zero_shape() {
        let pct = coverage_pct(0, 0);
        let fragment = format!("({pct:.2}%)");
        assert_eq!(fragment, "(0.00%)");
    }

    /// Full status embeddings line fragment: assert the complete sub-string
    /// that cmd_status emits for the embeddings row, including the parens and
    /// percent sign, so a whitespace or precision change is immediately visible.
    #[test]
    fn status_vector_lines_label_chunk_coverage_and_legacy_separately() {
        let [chunk, legacy] = status_vector_lines(100, 50, 100);
        assert_eq!(chunk, "   chunk vectors : 100 / 100  (100.00%)  [query path]");
        assert_eq!(
            legacy,
            "   legacy pooled : 50 / 100  [legacy embeddings table, not the query path]"
        );
    }

    #[test]
    fn query_accepts_allow_fallback_flag() {
        use clap::Parser;
        let cli = Cli::try_parse_from(["recall", "query", "--allow-fallback", "--", "x"]).unwrap();
        match cli.command {
            Commands::Query { allow_fallback, text, .. } => {
                assert!(allow_fallback);
                assert_eq!(text, "x");
            }
            _ => panic!("expected Query"),
        }
    }
}
