// store.rs — SQLite FTS5 store with bi-temporal validity.
//
// One SQLite file holds every observation + ingested vault doc. FTS5 gives the
// keyword channel (<10ms). Bi-temporal columns (valid_from / valid_to /
// superseded_by) let RecallDreamer supersede stale memories instead of
// overwriting them — the Zep/Graphiti technique FTS5-only rivals cannot do.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

pub mod freshness;
#[cfg(test)]
mod tenancy_tests;
use freshness::{now_secs, stat_sig, StatSig};

/// A single retrieval hit. `path` is vault-relative with forward slashes
/// (empty for ad-hoc `save` observations that have no source file).
#[derive(Debug, Clone, serde::Serialize)]
pub struct Hit {
    pub id: i64,
    pub path: String,
    pub title: String,
    pub snippet: String,
    #[allow(dead_code)] // Written by every channel; consumed by RRF once ANN index lands (WP2).
    pub score: f64,
}

/// Where a chunk came from — the provenance columns of `bge_chunk_embeddings`.
///
/// Grouped into a struct rather than three more positional arguments so that
/// `upsert_chunk_embedding` cannot be called with `char_start` and `char_end` transposed, which
/// is silent, permanent, and invisible until someone renders a hit in context.
///
/// Build one straight from a `bge::ChunkSpan`.
pub struct ChunkPlacement<'a> {
    pub heading_path: &'a str,
    /// 🔴 BYTE offsets into `documents.body`, not code-point offsets. See `bge::ChunkSpan`.
    pub char_start: usize,
    pub char_end: usize,
}

impl<'a> From<&'a crate::bge::ChunkSpan> for ChunkPlacement<'a> {
    fn from(s: &'a crate::bge::ChunkSpan) -> Self {
        Self {
            heading_path: &s.heading_path,
            char_start: s.char_start,
            char_end: s.char_end,
        }
    }
}

pub struct Store {
    pub conn: Connection,
    /// [`Store::searchable_snapshot`] per view, with the database version it was taken at.
    snapshots: std::cell::RefCell<HashMap<String, ((i64, i64), Snapshot)>>,
}

/// [`Store::searchable_snapshot`]: (fingerprint, searchable ids ascending, MinHash sketch).
pub type Snapshot = (String, Vec<i64>, Vec<u64>);

/// Views [`Store::searchable_snapshot`] caches per connection before it starts over.
const SNAPSHOT_CACHE_VIEWS: usize = 16;

/// The counts a guarded mutation checks before and after it runs.
///
/// Shared by every guarded operation — the stamp backfill today, the Qwen3 re-embed later — so
/// that "what a guard looks at" is one definition rather than one per command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuardCounts {
    /// `valid_to IS NULL`. Must never change across a mutation that does not intend to.
    pub live_docs: i64,
    /// Every row of `bge_chunk_embeddings`, tombstoned documents included.
    pub chunk_vectors: i64,
    /// Rows with no content hash. A stamp backfill must leave this at ZERO.
    pub content_sha_null: i64,
    /// Documents `NEEDS_CHUNK_EMBEDDING` would return — the real work queue.
    pub queue: i64,
    /// Live documents that have vectors and a content hash but no stamp: the stamp backfill's
    /// exact work set. Must be ZERO afterwards, or the run was partial.
    pub stampable: i64,
    /// Vectors whose document no longer exists. Must always be zero.
    pub orphan_vectors: i64,
}

/// Aggregate store health for `recall status`.
pub struct Stats {
    pub total: i64,
    pub valid: i64,
    pub superseded: i64,
    /// Legacy pooled `embeddings` table — NOT what the query path reads.
    pub embeddings: i64,
    /// Valid docs with chunk vectors (`bge_chunk_embeddings`) — what the bge channel reads.
    pub chunk_embedded: i64,
    pub by_source: Vec<(String, i64)>,
    pub db_bytes: u64,
}

/// `~/.recall/recall.db` — USERPROFILE is a path, not a secret, so env is fine.
pub fn default_db_path() -> Result<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .context("USERPROFILE/HOME not set — cannot locate ~/.recall")?;
    let dir = PathBuf::from(home).join(".recall");
    std::fs::create_dir_all(&dir).context("create ~/.recall")?;
    Ok(dir.join("recall.db"))
}

/// `~/.recall/exclude.txt` — the durable source for query-time path exclusions.
///
/// Deliberately does NOT create `~/.recall`: this is read on every query, and a read path
/// that makes directories is a surprise. If the directory is absent the file simply isn't
/// there, and the builtin patterns still apply.
pub fn exclude_file_path() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    PathBuf::from(home).join(".recall").join("exclude.txt")
}

/// `~/.recall/tiers.txt` — durable tier policy, same rationale as [`exclude_file_path`].
pub fn tiers_file_path() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default();
    PathBuf::from(home).join(".recall").join("tiers.txt")
}

/// [`vault_root`] for anything that WRITES or judges files against the disk: refuses when the
/// root is not an existing directory. Without this, a caller that forgot RECALL_VAULT resolved
/// every path under the default `~/recall-notes`, found nothing, skipped it, and exited 0
/// ("Indexed 0 docs"), which is how the hourly capture stopped indexing on 2026-11-06.
/// Notes larger than this are skipped by index (generated dumps); the `recall ui` preview counts them.
pub const MAX_DOC_BYTES: usize = 512 * 1024;

pub fn vault_root_checked() -> Result<PathBuf> {
    check_vault_root(vault_root())
}

pub(crate) fn check_vault_root(root: PathBuf) -> Result<PathBuf> {
    if root.is_dir() {
        return Ok(root);
    }
    let how = match vault_source() {
        "env" => "RECALL_VAULT points at",
        "settings" => "the `vault` setting points at",
        _ => "neither RECALL_VAULT nor the `vault` setting is set, and the default",
    };
    anyhow::bail!(
        "vault root not found: {how} {} is not an existing folder. Run `recall settings set vault <your notes folder>`.",
        root.display()
    )
}

/// Vault root: RECALL_VAULT (a path, not a secret), else the `vault` setting, else ~/recall-notes.
/// ONE root; several roots are a known v1 limit. Read per call, so `index`, `refresh` and every
/// other caller agree; a running daemon resolves it when it starts (restart it after a change).
pub fn vault_root() -> PathBuf {
    vault_resolved().0
}

/// "env" | "settings" | "default": where [`vault_root`] came from (`recall status`).
pub fn vault_source() -> &'static str {
    vault_resolved().1
}

fn vault_resolved() -> (PathBuf, &'static str) {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
    resolve_vault(std::env::var("RECALL_VAULT").ok(), crate::settings::Settings::load().vault, &home)
}

/// Pure precedence rule, so it is testable without touching the process environment.
pub(crate) fn resolve_vault(env: Option<String>, setting: Option<PathBuf>, home: &str) -> (PathBuf, &'static str) {
    match (env.filter(|e| !e.is_empty()), setting) {
        (Some(e), _) => (PathBuf::from(e), "env"),
        (None, Some(s)) => (s, "settings"),
        // No personal default: a stranger's notes live wherever they say. ~/recall-notes is only
        // a landing spot so a first `recall save` works.
        (None, None) => (PathBuf::from(home).join("recall-notes"), "default"),
    }
}

#[cfg(test)]
mod vault_tests {
    use super::resolve_vault;
    use std::path::PathBuf;

    #[test]
    fn env_wins_then_the_setting_then_the_default() {
        let set = Some(PathBuf::from("/notes"));
        assert_eq!(resolve_vault(Some("/env".into()), set.clone(), "/h"), (PathBuf::from("/env"), "env"));
        assert_eq!(resolve_vault(None, set.clone(), "/h"), (PathBuf::from("/notes"), "settings"));
        assert_eq!(resolve_vault(Some(String::new()), set, "/h").1, "settings", "an empty RECALL_VAULT is unset");
        assert_eq!(resolve_vault(None, None, "/h"), (PathBuf::from("/h").join("recall-notes"), "default"));
    }
}

/// What makes a live document need (re-)embedding. Hoisted so the work QUEUE and the progress
/// COUNTER are the same predicate — a counter that disagrees with the queue is how a run reports
/// "105 / 100 done" and nobody can tell whether it is finished.
///
/// Three cases, and the third is why the NULL branch is not optional: `content_sha256 <>
/// embedded_sha256` is NULL when either side is NULL, and `NULL IS NOT NULL` is false, so
/// without the explicit `IS NULL` branch every document embedded before these columns existed
/// is silently declared fresh.
/// [`NEEDS_CHUNK_EMBEDDING`] plus the SAME tier gate the QUERY path applies.
///
/// 🔴 Embedding a document the query path will never return is pure waste, and it was not a small
/// amount: `TierFilter`'s default is `["BRAIN","VENDOR"]`, so **SKILLS is dropped in SQL before
/// the loader's Rust filter even runs** — measured on the live corpus, 7,377 live documents and
/// 69,204 chunk rows that no search can reach. They were still being queued for embedding.
///
/// Derived from `TierFilter::load()` rather than a second list, deliberately. If the tier policy
/// changes — `RECALL_TIERS`, or `~/.recall/tiers.txt` — what gets EMBEDDED moves with what gets
/// SERVED, automatically. A hard-coded "skip SKILLS" here would silently stop embedding a tier
/// the moment the owner re-enabled it for search, which is the same class of drift this whole change
/// exists to remove.
fn needs_chunk_embedding_clause() -> String {
    format!(
        "{NEEDS_CHUNK_EMBEDDING}{}",
        TierFilter::load().sql_clause("d")
    )
}

const NEEDS_CHUNK_EMBEDDING: &str = "d.valid_to IS NULL
   AND (
        NOT EXISTS (SELECT 1 FROM bge_chunk_embeddings c WHERE c.doc_id = d.id)
     OR d.embedded_sha256 IS NULL
     OR d.content_sha256 IS NOT d.embedded_sha256
   )";

/// The UPDATE `backfill_content_hashes` issues, hoisted so its regression test asserts the
/// STATEMENT THAT SHIPS rather than a copy of it. `AND content_sha256 IS NULL` is defence in
/// depth against writing a stale digest over a hash another writer stored; duplicating this SQL
/// in the test would have let that predicate be deleted with every test still green.
const BACKFILL_UPDATE_SQL: &str = "UPDATE documents SET content_sha256 = ?1
      WHERE id = ?2 AND content_sha256 IS NULL";

/// Tenancy v1 schema, applied once by [`Store::migrate_tenancy`] inside one IMMEDIATE transaction.
///
/// `scope` gets `NOT NULL DEFAULT 'global'` and no REFERENCES clause: with foreign keys on, SQLite
/// only allows ADD COLUMN ... REFERENCES with a NULL default, and a NULL scope would be an unscoped
/// row. The tag grammar mirrors `scope::valid_tag` exactly — kind, ONE colon, a 1-64 char slug of
/// `[a-z0-9._-]` starting alphanumeric — spelled out with instr/substr because GLOB `*` alone
/// accepted `agent:a:b` and over-long slugs (Codex tenancy r2). Scope is immutable in v1 (design
/// decided 2026-11-06): moving a memory is supersede + re-save, so no cache ever goes stale.
const TENANCY_DDL: &str = "
    ALTER TABLE documents ADD COLUMN scope TEXT NOT NULL DEFAULT 'global';
    ALTER TABLE documents ADD COLUMN agent_id TEXT;
    CREATE TRIGGER IF NOT EXISTS documents_scope_ok_i BEFORE INSERT ON documents
    WHEN NOT (
        NEW.scope = 'global'
        OR (instr(NEW.scope, ':') > 0
            AND substr(NEW.scope, 1, instr(NEW.scope, ':') - 1) IN ('team', 'project', 'agent')
            AND length(substr(NEW.scope, instr(NEW.scope, ':') + 1)) BETWEEN 1 AND 64
            AND instr(substr(NEW.scope, instr(NEW.scope, ':') + 1), ':') = 0
            AND substr(NEW.scope, instr(NEW.scope, ':') + 1) GLOB '[a-z0-9]*'
            AND substr(NEW.scope, instr(NEW.scope, ':') + 1) NOT GLOB '*[^a-z0-9._-]*'))
    BEGIN SELECT RAISE(ABORT, 'documents.scope: malformed scope tag'); END;
    CREATE TRIGGER IF NOT EXISTS documents_scope_immutable BEFORE UPDATE OF scope ON documents
    WHEN NEW.scope IS NOT OLD.scope
    BEGIN SELECT RAISE(ABORT, 'documents.scope is immutable'); END;
";

/// Lowercase hex SHA-256 of `bytes`.
///
/// The identity/freshness primitive for the whole store: a document's `content_sha256` is
/// this over `body.as_bytes()`. Hex rather than a BLOB so the value is greppable in a
/// `sqlite3` shell and comparable with `<>` in plain SQL.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut s = String::with_capacity(out.len() * 2);
    for b in out {
        use std::fmt::Write as _;
        // Infallible: writing to a String never fails.
        let _ = write!(s, "{b:02x}");
    }
    s
}

impl Store {
    pub fn open() -> Result<Self> {
        let path = default_db_path()?;
        let conn = Connection::open(&path).with_context(|| format!("open {}", path.display()))?;
        let s = Store { conn, snapshots: Default::default() };
        s.init_schema()?;
        // A profile with nothing indexed yet gets the redacting default here, at the one place every
        // writer passes, whichever command created the profile (`settings set vault`, `recall ui`,
        // `status`, the daemon). How the index starts (redacted, or raw by an explicit choice) is
        // recorded in the database, so later runs do not depend on settings.json surviving.
        if matches!(s.has_documents(), Ok(false)) {
            crate::scrub::init_new_install();
            if let Err(e) = s.set_started_redacted(crate::scrub::Config::load().store_redacted) {
                eprintln!("recall: WARNING: could not record the redaction choice in the database ({e:#}); later runs will follow settings.json only.");
            }
        }
        s.apply_redaction_marker();
        Ok(s)
    }

    /// True when this database records that its index was started redacted. A database without the
    /// `profile_meta` table (made by an older version, or opened read-only before any writable
    /// open) has no marker: that is an existing index, and it keeps what it has.
    pub fn started_redacted(&self) -> bool {
        self.conn
            .query_row("SELECT value = 'on' FROM profile_meta WHERE key = 'scrub.store_redacted'", [], |r| r.get(0))
            .unwrap_or(false)
    }

    pub(crate) fn set_started_redacted(&self, on: bool) -> Result<()> {
        if on {
            self.conn.execute("INSERT OR REPLACE INTO profile_meta(key, value) VALUES ('scrub.store_redacted', 'on')", [])?;
        } else {
            self.conn.execute("DELETE FROM profile_meta WHERE key = 'scrub.store_redacted'", [])?;
        }
        Ok(())
    }

    /// A marked database makes this process redact unless the env pin or settings.json says otherwise.
    fn apply_redaction_marker(&self) {
        if self.started_redacted() {
            crate::scrub::redact_this_run();
        }
    }

    /// Any row at all, live or superseded: has this profile ever stored text?
    pub fn has_documents(&self) -> Result<bool> {
        Ok(self.conn.query_row("SELECT EXISTS(SELECT 1 FROM documents)", [], |r| r.get(0))?)
    }

    /// The existing db, read-only: creates no directory or file and runs no schema change
    /// (`recall ui` GET routes). A missing db is an error, not an empty store.
    pub fn open_readonly() -> Result<Self> {
        let home = std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOME"))
            .context("USERPROFILE/HOME not set — cannot locate ~/.recall")?;
        let path = PathBuf::from(home).join(".recall").join("recall.db");
        anyhow::ensure!(path.is_file(), "no database at {} yet (run `recall index`)", path.display());
        let conn = Connection::open_with_flags(
            &path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("open {} read-only", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let s = Self::from_conn(conn);
        s.apply_redaction_marker(); // a read; nothing is created here
        Ok(s)
    }

    /// Wrap a connection as-is: no schema init, no migration (read-only views and tests).
    pub(crate) fn from_conn(conn: Connection) -> Self {
        Store { conn, snapshots: Default::default() }
    }

    #[allow(dead_code)] // Used in tests and future bench harness (WP6).
    pub fn open_at(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        let s = Store { conn, snapshots: Default::default() };
        s.init_schema()?;
        Ok(s)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            -- SQLite defaults FK enforcement OFF, which silently no-ops the
            -- `ON DELETE CASCADE` on embeddings.doc_id below. Without this the
            -- hard DELETEs in ingest_dir (re-index by path) and reset() leave
            -- ghost embedding rows with no parent document, bloating the db on
            -- every re-index. Enabling here makes the documented cascade real.
            -- (SQLite does NOT enforce FKs on pre-existing data, so this never
            -- mass-deletes already-orphaned rows on open — it only governs
            -- future writes.)
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS documents (
                id            INTEGER PRIMARY KEY,
                path          TEXT NOT NULL DEFAULT '',
                title         TEXT NOT NULL DEFAULT '',
                body          TEXT NOT NULL,
                source        TEXT NOT NULL DEFAULT 'cli',
                created_at    TEXT NOT NULL,
                valid_from    TEXT NOT NULL,
                valid_to      TEXT,            -- NULL = currently valid
                superseded_by INTEGER,         -- id that replaced this row
                tier          TEXT,             -- BRAIN | SKILLS | VENDOR, NULL = untagged
                -- When the document was AUTHORED, as `YYYY-MM-DD`. NULL = unknown.
                -- Distinct from `created_at`, which is the INGEST timestamp: 95.5% of
                -- created_at values are one of two batch stamps (the 2026-06-11 and
                -- 2026-07-31 index runs), so it orders documents by when recall last
                -- scanned them, not by when the owner wrote them. Backfilled out-of-band
                -- from filename > frontmatter > mtime; see the commit that added it.
                -- Data only — nothing in the ranking path reads this column yet.
                authored_on   TEXT,
                -- SHA-256 of `body` AS IT IS NOW.
                --
                -- Written by every path that writes a body: `insert`, `ingest_dir`'s update /
                -- revive / insert arms, and `ingest_paths`' insert. `ingest_dir` also backfills
                -- any row still missing one before it walks, because a NULL here makes move
                -- detection silently inert.
                --
                -- ⚠️ It can still be NULL: `add_column_if_missing` adds it NULL to a
                -- pre-existing database, and it stays NULL until something touches that row.
                -- Readers must handle NULL - see `NEEDS_CHUNK_EMBEDDING`, which has an explicit
                -- IS NULL branch for exactly this.
                content_sha256  TEXT,
                -- SHA-256 of `body` AS IT WAS when this row's chunk vectors were built.
                --
                -- 🔴 Stale is NOT `content_sha256 <> embedded_sha256`, and it is NOT
                -- `content_sha256 IS NOT embedded_sha256` either. `<>` against a NULL yields
                -- NULL, which WHERE treats as false; and `IS NOT` between two NULLs is
                -- outright false. Straight after this migration BOTH columns are NULL on all
                -- live rows, so either predicate on its own silently excludes the
                -- entire corpus — exactly the rows that most need re-embedding.
                --
                -- The only correct staleness predicate spells the missing-hash case out:
                --
                --     embedded_sha256 IS NULL OR content_sha256 IS NOT embedded_sha256
                --
                -- The first branch catches "never embedded, or embedded before this column
                -- existed"; the second is null-safe for a populated pair and also fires when
                -- `content_sha256` is NULL against a populated `embedded_sha256`.
                embedded_sha256 TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_documents_path  ON documents(path);
            CREATE INDEX IF NOT EXISTS idx_documents_valid ON documents(valid_to);
            CREATE INDEX IF NOT EXISTS idx_documents_valid_source ON documents(valid_to, source);

            -- Dense BGE-small vectors, one per document. Stored as little-endian
            -- f32 BLOBs (see bge::to_bytes). ON DELETE CASCADE keeps vectors in
            -- step with the documents they describe.
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                vec    BLOB NOT NULL,
                dim    INTEGER NOT NULL
            );

            -- Per-CHUNK BGE vectors — the table that lets a long document be found
            -- by a passage in its middle.
            --
            -- `embeddings` above is keyed `doc_id INTEGER PRIMARY KEY`, which FORCES
            -- exactly one vector per document. The embed path therefore chunked a
            -- document, embedded every chunk, then MEAN-POOLED the chunk vectors back
            -- into that single row. Mean-pooling a 20-chunk note produces a vector that
            -- resembles none of its chunks: measured over the vault, 58.1% of documents
            -- exceed BGE's ~512-token window and 76.5% of all characters were never
            -- separately retrievable. Dropping the single-row constraint is the fix;
            -- the query path then scores a document by its BEST chunk
            -- (`bge::best_per_doc`), exactly as the gemini and voyage channels do.
            --
            -- NAMED, not numbered, on purpose: `embeddings2/3/4` are Voyage/Gemini and
            -- ~90 files under eval/ assume that numbering. A fifth number here would
            -- silently mean "Gemini rebuild 3" to every one of them.
            --
            -- `embeddings` is deliberately left in place, so an index built before this
            -- change keeps serving queries until the per-chunk table has been populated
            -- (see `BgeSearch::load`).
            -- `chunk_index` IS the chunk order — it is the second half of the primary key and
            -- what `load_all_chunk_embeddings` orders by. A separate `ord` column would be a
            -- duplicate of it that can drift, so there is deliberately only one.
            --
            -- `heading_path` / `char_start` / `char_end` are provenance, nullable so a row
            -- written by any other tool is still valid. They let a hit be shown in its
            -- surrounding context and re-verified against the note, instead of trusted because
            -- a vector said so. 🔴 char_start/char_end are BYTE offsets into `documents.body`,
            -- not code-point offsets — see `bge::ChunkSpan`.
            -- Qwen3-Embedding-0.6B vectors, 1024-dim. A SEPARATE table from
            -- `bge_chunk_embeddings` on purpose: the two models produce different widths (1024
            -- vs 384) and, more importantly, different CHUNKS. Qwen3 sees ~900 tokens where BGE
            -- saw ~350, so a doc_id has a different number of rows in each table and the
            -- chunk_index values do not correspond. Sharing a table would make every join a
            -- silent mixture of two geometries.
            --
            -- Keeping BGE means the swap is reversible: if Qwen3 loses on the evaluation set, nothing
            -- has been destroyed, and `--channels bge` still answers.
            --
            -- `embedded_sha256` is per-ROW here rather than per-document. The lesson from the
            -- stamp incident is that a vector must carry the hash of the text it was BUILT from;
            -- putting it on the document forces a whole-doc re-embed to correct one chunk.
            CREATE TABLE IF NOT EXISTS qwen3_chunk_embeddings (
                doc_id          INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index     INTEGER NOT NULL,
                vec             BLOB NOT NULL,
                dim             INTEGER NOT NULL,
                heading_path    TEXT,
                char_start      INTEGER,
                char_end        INTEGER,
                embedded_sha256 TEXT,
                PRIMARY KEY (doc_id, chunk_index)
            );

            CREATE TABLE IF NOT EXISTS bge_chunk_embeddings (
                doc_id       INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index  INTEGER NOT NULL,
                vec          BLOB NOT NULL,
                dim          INTEGER NOT NULL,
                heading_path TEXT,
                char_start   INTEGER,
                char_end     INTEGER,
                PRIMARY KEY (doc_id, chunk_index)
            );

            -- One row per vault enumeration. Exists for exactly one reason: the
            -- `--prune-missing` sanity gate needs to know what a HEALTHY enumeration of this
            -- vault looks like before it is allowed to tombstone anything. `enumerate_docs`
            -- shells out to `rg --files` behind a 5-second deadline and silently falls back
            -- to a slower, narrower recursive walk; both paths can return a SHORT list, and a
            -- short list is row-for-row indistinguishable from "the user deleted those files".
            -- Without a recorded history there is nothing to compare against and the gate
            -- would have to trust whatever this run happened to see.
            -- Per-corpus abstain calibration (query::phantom_null): key = corpus fingerprint,
            -- value = JSON list of phantom-probe scores. A cache; deleting it is always safe.
            CREATE TABLE IF NOT EXISTS abstain_calibration (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Facts about this profile that must not depend on settings.json surviving. One row so
            -- far: 'scrub.store_redacted' = 'on' when the index was started redacted. Not a cache.
            CREATE TABLE IF NOT EXISTS profile_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS index_runs (
                id            INTEGER PRIMARY KEY,
                ran_at        TEXT NOT NULL,
                enumerated    INTEGER NOT NULL,
                used_fallback INTEGER NOT NULL,  -- 1 = the recursive walk, not rg
                tombstoned    INTEGER NOT NULL DEFAULT 0,
                enum_ms       INTEGER           -- wall time of the enumeration; NULL = not measured
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title, body,
                content='documents',
                content_rowid='id',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, title, body)
                VALUES (new.id, new.title, new.body);
            END;
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, body)
                VALUES ('delete', old.id, old.title, old.body);
            END;
            -- 🔴 `UPDATE OF title, body`, not a bare `AFTER UPDATE`. `documents_fts` is an
            -- external-content FTS5 index over title+body ONLY, so an update to any other
            -- column needs no FTS work at all — but a bare AFTER UPDATE fires for all of them
            -- and rewrites the row's whole FTS entry every time.
            --
            -- Measured with a PRISTINE byte-identical copy of the real 7.1 GB database restored
            -- before each arm, the same fixed 10,000 rows, and the identical metadata write
            -- issued through sqlite3. (The binary cannot be used for the wide arm: opening the
            -- store runs the migration below and re-narrows the trigger, so the treatment would
            -- never reach the measurement. That mistake was made once and the run discarded.)
            --
            --     AFTER UPDATE              18.93 s   +6,548 rows in documents_fts_data   +6,155 pages
            --     AFTER UPDATE OF title,body  1.85 s           0 rows                        +36 pages
            --
            -- 4 KB pages: ~25.2 MB of net page allocation against ~147 KB, and 10.2x the wall
            -- time, to write 10,000 hashes of 64 bytes each. Over all 166,053 rows that is
            -- ~419 MB against ~2.4 MB. These are NET deltas, not a count of every page or row
            -- written — FTS5 merges as it goes — so treat them as direction and order of
            -- magnitude, not an exact write count. Every other metadata UPDATE in this file —
            -- valid_to tombstoning, superseded_by, tier, authored_on — was paying the same toll.
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE OF title, body ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, title, body)
                VALUES ('delete', old.id, old.title, old.body);
                INSERT INTO documents_fts(rowid, title, body)
                VALUES (new.id, new.title, new.body);
            END;
            "#,
        )?;

        // Both columns were first added by an out-of-band ALTER against the live database,
        // so they are missing from every other one. See `add_column_if_missing`.
        self.add_column_if_missing("tier", "TEXT")?;

        // `authored_on` has exactly the same history as `tier`: populated by an out-of-band
        // backfill against the live database, so every OTHER database would lack it.
        self.add_column_if_missing("authored_on", "TEXT")?;

        // The provenance columns were added to `bge_chunk_embeddings` one commit after the table
        // itself, so any database created from that first commit has the 4-column shape and
        // `CREATE TABLE IF NOT EXISTS` above will not touch it. Exactly the `tier` story again.
        self.add_column_to_if_missing("bge_chunk_embeddings", "heading_path", "TEXT")?;
        self.add_column_to_if_missing("bge_chunk_embeddings", "char_start", "INTEGER")?;
        self.add_column_to_if_missing("bge_chunk_embeddings", "char_end", "INTEGER")?;
        // Enumeration wall time, so the Activity panel can show how close a run came to ENUM_TIMEOUT.
        self.add_column_to_if_missing("index_runs", "enum_ms", "INTEGER")?;

        // Content-freshness columns. Same `tier` story: added to the CREATE TABLE above, so
        // only a database created after this commit has them; every existing file needs the
        // ALTER. The index is created AFTER them on purpose — on a pre-existing database
        // `content_sha256` does not exist until the line above has run, and CREATE INDEX on a
        // missing column is a hard error that would make the store unopenable.
        //
        // Both columns arrive NULL on every existing row and STAY NULL until the backfill
        // subcommand runs: adding a column writes no values. That is deliberate — an ALTER
        // that also populated 166k rows would fire the `documents_au` FTS trigger 166k times
        // inside schema init, on every open.
        // Replace the pre-existing WIDE `documents_au` on databases created before the
        // narrowing above. `CREATE TRIGGER IF NOT EXISTS` cannot do this — the trigger already
        // exists, so it is a silent no-op and the old wide definition survives forever.
        // Guarded on the stored SQL so this is a one-time migration, not a schema write on
        // every open.
        // 🔴 DROP and CREATE go in ONE transaction. SQLite DDL is transactional, but
        // `execute_batch` does NOT implicitly wrap its statements in one: run bare, the DROP
        // commits on its own, and a crash before the CREATE leaves the database with NO
        // `documents_au` at all — every later title/body edit would then silently stop updating
        // the FTS index, and nothing would report it.
        //
        // Two processes opening at once is safe without an explicit write lock: the transaction
        // is DEFERRED, so both may read the wide trigger, but the first `DROP TRIGGER` escalates
        // to the write lock. The loser either takes SQLITE_BUSY and rolls back whole, or runs
        // afterwards and replaces the narrow trigger with an identical narrow trigger. Every
        // outcome ends with exactly one narrow `documents_au`.
        //
        // The guard matches the full `AFTER UPDATE OF title, body`, not a bare "UPDATE OF": a
        // trigger narrowed to `UPDATE OF title` alone would otherwise be mistaken for this one.
        {
            let tx = self.conn.unchecked_transaction()?;
            let au_sql: Option<String> = tx
                .query_row(
                    "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='documents_au'",
                    [],
                    |r| r.get::<_, String>(0),
                )
                .optional()?;
            if au_sql
                .map(|sql| !sql.contains("AFTER UPDATE OF title, body"))
                .unwrap_or(false)
            {
                tx.execute_batch(
                    "DROP TRIGGER documents_au;
                     CREATE TRIGGER documents_au AFTER UPDATE OF title, body ON documents BEGIN
                         INSERT INTO documents_fts(documents_fts, rowid, title, body)
                         VALUES ('delete', old.id, old.title, old.body);
                         INSERT INTO documents_fts(rowid, title, body)
                         VALUES (new.id, new.title, new.body);
                     END;",
                )?;
            }
            tx.commit()?;
        }

        self.add_column_if_missing("content_sha256", "TEXT")?;
        self.add_column_if_missing("embedded_sha256", "TEXT")?;
        self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(content_sha256);",
        )?;

        // File-change freshness (see store/freshness.rs). All NULL/0 on existing rows: an ALTER
        // writes no values, and the first scan or re-index fills them in.
        self.add_column_if_missing("modified_at", "INTEGER")?;
        self.add_column_if_missing("file_size", "INTEGER")?;
        self.add_column_if_missing("indexed_at", "INTEGER")?;
        self.add_column_if_missing("changed_at", "INTEGER")?;
        self.add_column_if_missing("stale", "INTEGER NOT NULL DEFAULT 0")?;
        self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_documents_stale ON documents(stale) WHERE stale <> 0;
             CREATE TABLE IF NOT EXISTS fs_events (
                 path    TEXT PRIMARY KEY,
                 kind    TEXT NOT NULL,
                 seen_at INTEGER NOT NULL
             );",
        )?;
        self.migrate_tenancy()?;
        // Partial: every pre-tenancy row is 'global', so on an existing database this index is
        // EMPTY and its one-time build is a single scan; it makes `denied_ids` cost O(private rows).
        self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_documents_nonglobal ON documents(scope) WHERE scope <> 'global';",
        )?;
        Ok(())
    }

    /// Ids the current reader may NOT see, or `None` for the operator. Only non-`global` rows can be
    /// denied, and `idx_documents_nonglobal` holds exactly those. Tombstoned rows are included on
    /// purpose: resident matrices keep their vectors, and a denied tombstone that ranked above the
    /// cut would still use up a top-k slot (Codex tenancy b2).
    pub fn denied_ids(&self) -> Result<Option<HashSet<i64>>> {
        let reader = crate::scope::current();
        if reader.is_operator() {
            return Ok(None);
        }
        let mut stmt = self
            .conn
            .prepare("SELECT id, scope FROM documents WHERE scope <> 'global'")?;
        let mut out = HashSet::new();
        for row in stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
            let (id, tag) = row?;
            if !reader.allows(&tag) {
                out.insert(id);
            }
        }
        Ok(Some(out))
    }

    /// Tenancy v1 (src/scope.rs): `scope` + `agent_id` on `documents`, plus the tag-grammar and
    /// immutability triggers.
    ///
    /// 🔴 The presence check runs BEFORE any lock. `init_schema` runs on every `Store::open`, and a
    /// `BEGIN IMMEDIATE` there would make every reader contend for the writer lock (Codex tenancy
    /// r2). Only a database that lacks the column pays for the lock — once, ever.
    ///
    /// 🔴 Columns and triggers go in ONE transaction: `execute_batch` adds none of its own (see the
    /// trigger swap above), and a worker writing between the ALTER and the trigger would insert a
    /// row no trigger validated. Re-checked inside the lock because another process may have won.
    ///
    /// Constant time on a large table: a plain ADD COLUMN rewrites no rows (sqlite.org
    /// lang_altertable.html). No CHECK on the column (that scans every row) and no index (nothing
    /// looks rows up by scope); existing rows read `scope = 'global'`, `agent_id = NULL`.
    fn migrate_tenancy(&self) -> Result<()> {
        if self.has_column("documents", "scope")? {
            return Ok(());
        }
        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        let r = (|| -> Result<()> {
            if !self.has_column("documents", "scope")? {
                self.conn.execute_batch(TENANCY_DDL)?;
            }
            Ok(())
        })();
        match r {
            Ok(()) => self.conn.execute_batch("COMMIT")?,
            Err(e) => {
                let _ = self.conn.execute_batch("ROLLBACK");
                return Err(e);
            }
        }
        Ok(())
    }

    fn has_column(&self, table: &str, name: &str) -> Result<bool> {
        Ok(self
            .conn
            .prepare("SELECT 1 FROM pragma_table_info(?1) WHERE name = ?2")?
            .exists(params![table, name])?)
    }

    /// Idempotently add a column to `documents`.
    ///
    /// `CREATE TABLE IF NOT EXISTS` is a no-op on a database that already exists, so a column
    /// added to the definition above never reaches an older file. Without this, any database
    /// other than the live one — a restored B2 snapshot, a fresh test fixture, a second brain
    /// on another machine — would still lack the column and every query would die on
    /// `no such column: d.tier`. SQLite has no `ADD COLUMN IF NOT EXISTS`, so check the table
    /// info and tolerate the duplicate-column error if we lose a race.
    fn add_column_if_missing(&self, name: &str, decl: &str) -> Result<()> {
        self.add_column_to_if_missing("documents", name, decl)
    }

    /// [`Store::add_column_if_missing`] for any table.
    ///
    /// `table`, `name` and `decl` are compile-time literals from this file only — never caller
    /// input — which is why they can be formatted into the DDL (SQLite cannot bind identifiers).
    fn add_column_to_if_missing(&self, table: &str, name: &str, decl: &str) -> Result<()> {
        let present = self
            .conn
            .prepare("SELECT 1 FROM pragma_table_info(?1) WHERE name = ?2")?
            .exists(params![table, name])?;
        if present {
            return Ok(());
        }
        match self
            .conn
            .execute(&format!("ALTER TABLE {table} ADD COLUMN {name} {decl}"), [])
        {
            Ok(_) => Ok(()),
            // Another process won the race and added it first — that is success.
            Err(e) if e.to_string().contains("duplicate column name") => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    /// Fill `content_sha256` from the stored body for rows that lack it.
    ///
    /// Reads `body`, writes ONE column. Never touches `body`, `path`, `valid_to` or any vector.
    /// Resumable and idempotent: the `IS NULL` filter means a re-run picks up where an
    /// interrupted one stopped, and a finished corpus is a no-op.
    ///
    /// 🔴 The SELECT is INSIDE the transaction, and the UPDATE re-asserts `content_sha256 IS
    /// NULL`. Both are needed to stop a stale hash being written: if the read happened outside
    /// the transaction, another writer could store a new body and its correct hash in the gap,
    /// and this would then overwrite that correct hash with the digest of the body it read
    /// earlier — a WRONG value that is non-NULL and therefore never revisited by any later run.
    /// Inside the transaction, an intervening commit makes the UPDATE fail the WAL snapshot;
    /// that error propagates and the whole batch rolls back, so nothing wrong is ever committed.
    /// The `IS NULL` in the UPDATE is defence in depth, not the mechanism: under snapshot
    /// isolation it should never be the thing that saves us, and it is there for the case where
    /// this function is ever called on a connection whose transaction semantics differ.
    ///
    /// Returns how many rows were ACTUALLY updated, not how many were selected. Call in a
    /// loop until it returns 0. Zero means one thing only: the SELECT found no unhashed rows.
    /// It is NOT how a concurrent writer shows up — that case does not return zero, it
    /// returns an error, because the failed WAL snapshot propagates through `?`.
    ///
    /// Batched rather than one big statement because a single UPDATE over all ~166k rows holds
    /// the database's write lock for the whole run.
    ///
    /// It does NOT fire `documents_au`: that trigger is `AFTER UPDATE OF title, body`, and this
    /// writes neither. Before that narrowing it did fire on every row — see the trigger's own
    /// comment for what that cost, measured.
    pub fn backfill_content_hashes(&self, limit: usize) -> Result<usize> {
        let tx = self.conn.unchecked_transaction()?;
        let rows: Vec<(i64, String)> = {
            let mut stmt = tx.prepare(
                "SELECT id, body FROM documents
                  WHERE content_sha256 IS NULL
                  ORDER BY id LIMIT ?1",
            )?;
            let out = stmt
                .query_map(params![limit as i64], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            out
        };
        if rows.is_empty() {
            return Ok(0);
        }
        let mut changed = 0usize;
        {
            let mut up = tx.prepare(BACKFILL_UPDATE_SQL)?;
            for (id, body) in &rows {
                changed += up.execute(params![sha256_hex(body.as_bytes()), id])?;
            }
        }
        tx.commit()?;
        Ok(changed)
    }

    /// DESTRUCTIVE — wipe all rows. Caller must gate behind explicit confirmation.
    pub fn reset(&self) -> Result<()> {
        self.conn.execute_batch(
            "DELETE FROM documents; INSERT INTO documents_fts(documents_fts) VALUES('rebuild');",
        )?;
        Ok(())
    }

    fn now() -> String {
        // RFC3339-ish UTC without pulling chrono. Good enough for ordering.
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        format!("{}", secs)
    }

    /// Insert a currently-valid document, returning its id.
    ///
    /// Tags `tier` at insert time via [`tier_for_path`] — this is a brand-new row, so there
    /// is no existing value to protect; every later touch of this document goes through an
    /// UPDATE path that never overwrites tier.
    pub fn insert(&self, path: &str, title: &str, body: &str, source: &str) -> Result<i64> {
        self.insert_scoped(path, title, body, source, crate::scope::GLOBAL, None)
    }

    /// [`Store::insert`] into a scope, recording who wrote it (`agent_id`; None = operator). The
    /// caller picks the tag through `scope::save_target`; the trigger re-checks its grammar.
    pub fn insert_scoped(
        &self,
        path: &str,
        title: &str,
        body: &str,
        source: &str,
        scope: &str,
        agent_id: Option<&str>,
    ) -> Result<i64> {
        let now = Self::now();
        let tier = tier_for_path(path);
        self.conn.execute(
            "INSERT INTO documents (path, title, body, source, created_at, valid_from, valid_to, superseded_by, tier, content_sha256, scope, agent_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5, NULL, NULL, ?6, ?7, ?8, ?9)",
            params![path, title, body, source, now, tier, sha256_hex(body.as_bytes()), scope, agent_id],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// FTS5 keyword search over currently-valid docs. Lower bm25 = better, so we
    /// return bm25 negated as `score` (higher = better) for uniform RRF ranking.
    ///
    /// The bm25 column weights are [`BM25_TITLE_WEIGHT`] / [`BM25_BODY_WEIGHT`].
    ///
    /// RANKING search: its MATCH holds up to [`MAX_MATCH_TOKENS`] query tokens.
    pub fn fts_search(&self, query: &str, k: usize) -> Result<Vec<Hit>> {
        crate::query::prof::time("fts_search", || self.fts_search_with_tiers(query, k, &TierFilter::load()))
    }

    /// [`Store::fts_search`] over the abstain judge's tokens ([`MAX_FTS_TOKENS`]), with snippets:
    /// exactly what `fts_search` returned before the cap split. Used where hits are RETURNED but
    /// must not differ from what the judge saw (PLAN.md R1, R2).
    pub fn fts_search_judged(&self, query: &str, k: usize) -> Result<Vec<Hit>> {
        crate::query::prof::time("fts_search_judged", || {
            self.fts_search_capped(query, k, &TierFilter::load(), MAX_FTS_TOKENS, true)
        })
    }

    /// Ids of [`Store::fts_search_judged`], without the snippet query: the abstain judge's
    /// EVIDENCE search, never returned to a caller.
    pub fn fts_judge_ids(&self, query: &str, k: usize) -> Result<Vec<i64>> {
        Ok(self
            .fts_search_capped(query, k, &TierFilter::load(), MAX_FTS_TOKENS, false)?
            .into_iter()
            .map(|h| h.id)
            .collect())
    }

    /// `fts_search` with the tier gate supplied explicitly. Tests use this rather than
    /// setting `RECALL_TIERS`: mutating process environment is not thread-safe under the
    /// parallel test harness, which is the same reason `PathExcludeList::with_extra` exists.
    pub fn fts_search_with_tiers(
        &self,
        query: &str,
        k: usize,
        tiers: &TierFilter,
    ) -> Result<Vec<Hit>> {
        self.fts_search_capped(query, k, tiers, MAX_MATCH_TOKENS, true)
    }

    /// The one body behind every fts search: MATCH of the first `cap` query tokens, snippets
    /// fetched for the survivors only when `snippets`.
    fn fts_search_capped(
        &self,
        query: &str,
        k: usize,
        tiers: &TierFilter,
        cap: usize,
        snippets: bool,
    ) -> Result<Vec<Hit>> {
        let profile = std::time::Instant::now();
        // Stores at or under the floor take the query path that shipped: no prune, original SQL.
        let raw = fts_tokens_capped(query, cap);
        if raw.is_empty() {
            return Ok(vec![]); // no token, no database read (as before the gate)
        }
        let big = self.fts_over_floor()?;
        let tokens = if big { self.prune_common_tokens(raw)? } else { raw };
        let match_expr = tokens.iter().map(|t| format!("\"{t}\"")).collect::<Vec<_>>().join(" OR ");
        if match_expr.is_empty() {
            return Ok(vec![]);
        }
        // The vector channels have always filtered excluded paths; this one did not, so the
        // exclusion silently depended on a vector channel being live. When Gemini was
        // switched off, an excluded scratch copy of a profile for a DIFFERENT, closed company
        // with the same name came straight back to rank 1 for a question about the real one.
        // A filter that only works while an optional channel is up is not a filter.
        let exclude = PathExcludeList::load();
        // Over-fetch before filtering: applying the exclusion after a `LIMIT k` would return
        // fewer than k hits whenever any excluded path scored into the top k. And when the first
        // window is still short (more excluded rows ranked above the eligible ones than the
        // over-fetch allows), widen it and rank again until k eligible hits or the matches run out
        // (loop2 r3, Codex r2 #6: a depth-5 fetch read only 37 rows, so 38 strongly-ranked
        // excluded notes hid the one eligible runbook and the bge-only judge abstained on it).
        let mut fetch = k.saturating_mul(4).max(k + 32);

        // 🔴 TWO QUERIES, DELIBERATELY. `snippet()` used to sit in the ranking SELECT, so it was
        // evaluated while scanning every matching row — and `fts_match_expr` ORs every query term,
        // so a typical question matches ~94% of the corpus. Measured 2026-08-21 on a real long
        // question (16 terms, ~94% of the corpus matching):
        //
        //   rank + bm25 + join, LIMIT 72          729 ms
        //   the same query WITH snippet()       1,544 ms   <- snippet DOUBLED it
        //   snippet fetched separately for 72     245 ms
        //
        // So ranking first and snippetting only the survivors costs ~974 ms instead of ~1,544 ms.
        // The second query MUST repeat `MATCH ?1`: `snippet()` is only defined in the context of a
        // full-text match and errors out without it. That re-scan is what the 245 ms buys.
        // Scope rides the tier rail, inside SQL, so `LIMIT` counts only rows this reader may see.
        let rank_sql = Self::fts_rank_sql(&format!("{}{}", tiers.sql_clause("d"), crate::scope::clause("d")), big);
        let mut stmt = self.conn.prepare(&rank_sql)?;
        let mut out = Vec::new();
        loop {
            out.clear();
            let mut read = 0usize;
            // bm25 of the LAST row of this window (every row, eligible or not): the pushdown cuts by
            // (score, rowid), so rows tied with it that sit past the cut may belong ahead of ours by
            // (score, path, id).
            let last_score = std::cell::Cell::new(f64::NAN);
            let rows = stmt.query_map(params![match_expr, fetch as i64], |r| {
                last_score.set(r.get::<_, f64>(3)?);
                let eligible = !big || r.get::<_, Option<bool>>(4)?.unwrap_or(false);
                Ok(eligible.then(|| -> rusqlite::Result<Hit> {
                    Ok(Hit {
                        id: r.get(0)?,
                        path: r.get(1)?,
                        title: r.get(2)?,
                        snippet: String::new(),
                        score: -r.get::<_, f64>(3)?,
                    })
                }).transpose()?)
            })?;
            for r in rows {
                read += 1;
                // Not live, another tier or scope, or an fts row with no document: ranked, never returned.
                let Some(hit) = r? else { continue };
                if is_query_excluded_path(&exclude, &hit.path) {
                    continue;
                }
                if out.len() < k {
                    out.push(hit);
                }
                // The original single-SELECT form is fully ordered, so it can stop at k. The
                // pushdown form reads the whole window to learn its cut score.
                if out.len() >= k && !big {
                    break;
                }
            }
            // A full window whose k-th hit ties the cut score may have lost tied rows to the inner
            // LIMIT: widen until the tie group fits (strictly better k-th score, or window exhausted).
            let tie_at_cut = big && out.len() >= k && read >= fetch && k > 0 && -out[k - 1].score == last_score.get();
            if (out.len() >= k && !tie_at_cut) || read < fetch {
                break;
            }
            fetch = fetch.saturating_mul(4);
        }

        // Snippets for the SURVIVORS only — after the exclusion filter and after the truncation to
        // `k`, so nothing is rendered for a row that was never going to be returned.
        if snippets && !out.is_empty() {
            let ids: Vec<i64> = out.iter().map(|h| h.id).collect();
            let snip_sql = Self::fts_snippet_sql(ids.len());
            let mut stmt = self.conn.prepare(&snip_sql)?;
            let mut params: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(ids.len() + 1);
            params.push(&match_expr);
            for id in &ids {
                params.push(id);
            }
            let mut snips: HashMap<i64, String> = HashMap::with_capacity(ids.len());
            let rows = stmt.query_map(params.as_slice(), |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })?;
            for row in rows {
                let (id, snip) = row?;
                snips.insert(id, snip);
            }
            for hit in out.iter_mut() {
                if let Some(s) = snips.remove(&hit.id) {
                    hit.snippet = s;
                }
            }
        }
        if std::env::var_os("RECALL_PROFILE").is_some_and(|v| v == "1") {
            eprintln!("profile fts_ms={:.1} tokens={} hits={}", profile.elapsed().as_secs_f64() * 1000.0, tokens.len(), out.len());
        }
        Ok(out)
    }

    /// More than [`FTS_PRUNE_MIN_DF`] rows in the fts index (tombstoned documents included)? Counts at
    /// most floor + 1 rows, so it costs the same on a 150k-row store as on a 50-row one. This is the
    /// one switch for the prune and the pushdown SQL: both were chosen on DEV stores above the floor.
    fn fts_over_floor(&self) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM (SELECT 1 FROM documents_fts_docsize LIMIT ?1)",
            params![FTS_PRUNE_MIN_DF + 1],
            |r| r.get(0),
        )?;
        Ok(n > FTS_PRUNE_MIN_DF)
    }

    /// Drop query tokens held by more than `RECALL_FTS_COMMON_FRAC` of the fts rows from the ranking
    /// MATCH, keeping the [`FTS_KEEP_RAREST`] rarest tokens whatever their df. OR-ing a near-universal
    /// word ("what", "the", "did") makes the MATCH rank most of the corpus (0.8-6.8 s on a 155k-row
    /// store) while bm25 gives such a word almost no weight. On by default at 0.25 (stores above the
    /// floor only); `RECALL_FTS_COMMON_FRAC=0` turns it off, a fraction in (0, 1] changes the cut. df counts fts rows (tombstoned documents included), capped just above the cut.
    fn prune_common_tokens(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let Some(frac) = fts_common_frac() else { return Ok(tokens) };
        if tokens.len() <= FTS_KEEP_RAREST {
            return Ok(tokens);
        }
        let n: i64 = self.conn.query_row("SELECT count(*) FROM documents_fts_docsize", [], |r| r.get(0))?;
        // Count each token's matching rows only up to the cut + 1: a common token reads at most that
        // many doclist entries, never its whole posting list (a temp fts5vocab lookup cost p95 2.3 s).
        // Never below FTS_PRUNE_MIN_DF rows: in a small store a speaker name can sit in a quarter of
        // the documents and is the word that matters, while ranking the whole store is cheap anyway.
        let cut = ((frac * n as f64).floor() as i64).max(FTS_PRUNE_MIN_DF);
        let mut stmt = self.conn.prepare_cached(
            "SELECT count(*) FROM (SELECT 1 FROM documents_fts WHERE documents_fts MATCH ?1 LIMIT ?2)",
        )?;
        let mut df = Vec::with_capacity(tokens.len());
        for t in &tokens {
            df.push(stmt.query_row(params![format!("\"{t}\""), cut + 1], |r| r.get::<_, i64>(0))?);
        }
        Ok(keep_rare(tokens, &df, cut as f64))
    }

    /// `pushdown` is the big-store form (bm25 ORDER BY/LIMIT inside a subquery, eligibility checked
    /// after); a store at or under [`FTS_PRUNE_MIN_DF`] fts rows takes the original single-SELECT form
    /// so its query path is the one that shipped, byte for byte (4 columns; the pushdown form adds
    /// `eligible` as column 4 because it filters after the LIMIT).
    /// The RANKING half of `fts_search`. Must NOT contain `snippet(` — see the two-query comment
    /// in `fts_search_with_tiers`. Split out so a test can assert that structurally rather than
    /// trusting a comment to stay true.
    fn fts_rank_sql(tier_clause: &str, pushdown: bool) -> String {
        if !pushdown {
            return format!(
                "SELECT d.id, d.path, d.title,
                    bm25(documents_fts, {BM25_TITLE_WEIGHT:.1}, {BM25_BODY_WEIGHT:.1}) AS score
             FROM documents_fts
             JOIN documents d ON d.id = documents_fts.rowid
             WHERE documents_fts MATCH ?1 AND d.valid_to IS NULL{tier_clause}
             ORDER BY score, d.path, d.id
             LIMIT ?2"
            );
        }
        format!(
            "SELECT f.rowid, d.path, d.title, f.score,
                    d.id IS NOT NULL AND d.valid_to IS NULL{tier_clause} AS eligible
             FROM (SELECT rowid, bm25(documents_fts, {BM25_TITLE_WEIGHT:.1}, {BM25_BODY_WEIGHT:.1}) AS score
                   FROM documents_fts WHERE documents_fts MATCH ?1 ORDER BY score, rowid LIMIT ?2) f
             LEFT JOIN documents d ON d.id = f.rowid
             ORDER BY f.score, d.path, d.id"
        )
    }

    /// The SNIPPET half: one targeted lookup for the rows that survived ranking and filtering.
    ///
    /// Repeats `MATCH ?1` because `snippet()` is undefined outside a full-text match context —
    /// dropping it does not merely change the text, it raises a SQL error.
    fn fts_snippet_sql(n: usize) -> String {
        let placeholders = std::iter::repeat_n("?", n).collect::<Vec<_>>().join(", ");
        format!(
            "SELECT documents_fts.rowid,
                    snippet(documents_fts, 1, '[', ']', '…', 12)
             FROM documents_fts
             WHERE documents_fts MATCH ?1
               AND documents_fts.rowid IN ({placeholders})"
        )
    }

    /// Live docs the current reader may see (= `count_valid` for the operator).
    pub fn count_visible(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            &format!(
                "SELECT COUNT(*) FROM documents WHERE valid_to IS NULL{}",
                crate::scope::clause("documents")
            ),
            [],
            |r| r.get(0),
        )?)
    }

    pub fn count_valid(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE valid_to IS NULL",
            [],
            |r| r.get(0),
        )?)
    }

    /// Aggregate health stats for `recall status`.
    pub fn stats(&self) -> Result<Stats> {
        let total: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))?;
        let valid = self.count_valid()?;
        let superseded: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE valid_to IS NOT NULL",
            [],
            |r| r.get(0),
        )?;
        let mut stmt = self.conn.prepare(
            "SELECT source, COUNT(*) FROM documents WHERE valid_to IS NULL GROUP BY source ORDER BY 2 DESC",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
        let mut by_source = Vec::new();
        for r in rows {
            by_source.push(r?);
        }
        let db_bytes = default_db_path()
            .ok()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .unwrap_or(0);
        let embeddings = self.count_embeddings()?;
        let chunk_embedded = self.count_chunk_embedded_docs()?;
        Ok(Stats {
            total,
            valid,
            superseded,
            embeddings,
            chunk_embedded,
            by_source,
            db_bytes,
        })
    }

    /// Most recent valid observations (non-vault), newest-first.
    pub fn recent(&self, n: usize) -> Result<Vec<Hit>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT id, path, source, substr(body,1,120)
             FROM documents
             WHERE valid_to IS NULL AND source != 'vault'{}
             ORDER BY created_at DESC, id DESC LIMIT ?1",
            crate::scope::clause("documents")
        ))?;
        let rows = stmt.query_map(params![n as i64], |r| {
            Ok(Hit {
                id: r.get(0)?,
                path: r.get(1)?,
                title: r.get(2)?, // reuse title slot for source label
                snippet: r.get(3)?,
                score: 0.0,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Manually retire valid observations whose body contains `text`
    /// (case-insensitive). Bi-temporal: sets valid_to, superseded_by stays NULL
    /// (user-initiated forget, no replacement). Returns rows retired.
    pub fn forget(&self, text: &str) -> Result<usize> {
        let now = Self::now();
        let pat = format!("%{}%", text);
        let n = self.conn.execute(
            &format!(
                "UPDATE documents SET valid_to = ?1
                 WHERE valid_to IS NULL AND source != 'vault' AND body LIKE ?2 COLLATE NOCASE{}",
                crate::scope::current().write_clause("documents")
            ),
            params![now, pat],
        )?;
        Ok(n)
    }

    /// Retire a single document by id. Returns true if it was valid.
    pub fn supersede_id(&self, id: i64) -> Result<bool> {
        let now = Self::now();
        let n = self.conn.execute(
            &format!(
                "UPDATE documents SET valid_to = ?1 WHERE id = ?2 AND valid_to IS NULL{}",
                crate::scope::current().write_clause("documents")
            ),
            params![now, id],
        )?;
        Ok(n > 0)
    }

    /// Export all valid docs as a JSON array (backup / piping).
    ///
    /// Streams one object at a time to `w` instead of building a `Vec` in RAM,
    /// keeping RSS flat even with 100k+ docs (~400 MB avoided at 101k docs).
    pub fn export_json<W: Write>(&self, mut w: W) -> Result<()> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT id, path, title, body, source, created_at FROM documents WHERE valid_to IS NULL{} ORDER BY id",
            crate::scope::clause("documents")
        ))?;
        let mut rows = stmt.query([])?;
        write!(w, "[")?;
        let mut first = true;
        while let Some(r) = rows.next()? {
            let obj = serde_json::json!({
                "id": r.get::<_, i64>(0)?,
                "path": r.get::<_, String>(1)?,
                "title": r.get::<_, String>(2)?,
                "body": r.get::<_, String>(3)?,
                "source": r.get::<_, String>(4)?,
                "created_at": r.get::<_, String>(5)?,
            });
            if first {
                first = false;
            } else {
                write!(w, ",")?;
            }
            write!(w, "\n{}", serde_json::to_string_pretty(&obj)?)?;
        }
        writeln!(w, "\n]")?;
        Ok(())
    }

    /// Ingest every .md / .qmd under `root` into FTS5. Returns docs processed.
    ///
    /// Re-ingest is idempotent on path AND **stable on `id`**: an existing vault
    /// row is updated in place, never deleted and re-inserted. Doc ids are the
    /// join key for every vector table (`embeddings`, `embeddings2`, …), so
    /// renumbering them orphans the entire semantic index — see the long comment
    /// at the update site below for the incident this rule comes from.
    ///
    /// Commits in batches and checkpoints the WAL so memory stays bounded even
    /// on a vault with tens of thousands of docs (a single mega-transaction
    /// pins the entire WAL in RAM — learned the hard way).
    ///
    /// Snapshot mirror directories (e.g. `tools/state/skill-radar/disabled-loader-roots-*`)
    /// are excluded at enumeration time via `PathExcludeList`. Extend the list
    /// at runtime without a recompile: `RECALL_EXCLUDE_EXTRA="archived,backup"`.
    pub fn ingest_dir(&self, root: &Path) -> Result<usize> {
        Ok(self.ingest_dir_counted(root)?.0)
    }

    /// [`Store::ingest_dir`], plus what the pass did to each file (for the `recall index` summary).
    pub fn ingest_dir_counted(&self, root: &Path) -> Result<(usize, IngestCounts)> {
        const BATCH: usize = 1000;
        const MAX_BYTES: usize = MAX_DOC_BYTES; // skip pathological files (generated dumps)

        let exclude = PathExcludeList::load();
        let Enumeration {
            files,
            used_fallback,
            elapsed_ms,
        } = enumerate_docs(root, &exclude)?;
        #[cfg(test)]
        let files = tests::arrange_enumeration(files);
        eprintln!(
            "recall index: found {} docs in {:.1} s{}",
            files.len(),
            elapsed_ms as f64 / 1000.0,
            if used_fallback { " (slow walk: rg did not run)" } else { "" }
        );

        // 🔴 Fill missing content hashes BEFORE the walk, or move detection is INERT.
        //
        // `add_column_if_missing` leaves `content_sha256` NULL on every row of a pre-existing
        // database, and the `unchanged` arm below writes nothing — so a document nobody has
        // edited since the migration stays NULL forever. Both move predicates compare
        // `content_sha256 = ?1`, and NULL equals nothing, so on the real 7.1 GB corpus (NULL on
        // all 166,053 rows) rename detection would find no candidates, print `0 moved (renamed)`,
        // and be indistinguishable from "no renames happened". That is the fail-open shape this
        // project keeps getting caught by, so it is closed here rather than documented.
        //
        // Here and not only in `cmd_embed`, because this is the function that NEEDS the hashes;
        // putting it in one caller left `recall index`, the path-list route and the MCP server
        // all running with the feature switched off. Idempotent, and on an already-hashed corpus
        // it is a single SELECT that returns nothing.
        {
            let mut filled = 0usize;
            loop {
                let n = self.backfill_content_hashes(5_000)?;
                if n == 0 {
                    break;
                }
                filled += n;
            }
            if filled > 0 {
                eprintln!(
                    "recall index: backfilled {filled} missing content hashes (move detection needs them)"
                );
            }
        }
        // Feed the prune gate's history even on a plain `recall index`, so the baseline is
        // built by ordinary use rather than only by prune runs. Safe to record before the
        // gate reads it in the same command, because the baseline is the MAX over a window
        // of recent runs (see PRUNE_BASELINE_RUNS) — this row cannot lower the bar.
        self.record_enumeration(files.len(), used_fallback, 0, Some(elapsed_ms))?;

        let now = Self::now();
        let mut n = 0usize;
        let mut skipped = 0usize;
        let (mut inserted, mut updated, mut unchanged) = (0usize, 0usize, 0usize);
        let mut revived = 0usize;
        let mut moved = 0usize;
        // How many INCOMING files carry each body hash.
        //
        // 🔴 Load-bearing, and proved so the hard way. It was written, then deleted as redundant
        // when a different structure made the database count cover the same case, then the
        // structure changed again and `a_rename_and_a_copy_in_one_run_never_steal_the_identity`
        // went red without it. With `a.md` renamed to `z.md` AND copied to `b.md`, resolution
        // sees exactly one absent candidate for whichever file it reaches first, so THAT file
        // takes the original's id — and which file that is depends on the order the walk
        // happened to return them in. Two files carry the same body; which one is "the rename"
        // is unknowable; so neither may claim it.
        let mut incoming_digests: HashMap<String, usize> = HashMap::new();
        // Files that MIGHT be a move: same body as some existing live document. Deferred
        // UNWRITTEN until the enumeration is complete, and resolved in one pass at the end.
        //
        // `(absolute path, vault-relative path, digest AS ENUMERATED)`. The body is re-read at
        // resolution rather than buffered — buffering would put the whole corpus in RAM on a
        // vault where every file duplicates another — and the enumerated digest is kept so the
        // re-read can be CHECKED against it. If the file changed in between, every count and
        // eligibility decision taken for it is about content it no longer has, so it is skipped
        // and left to the next run.
        //
        // ⚠️ NOT empty on a first index, despite the eligibility test asking for an existing live
        // document: the test sees this transaction's own earlier inserts, so the second of two
        // identical files is deferred even against an empty database. That is correct — they are
        // genuinely ambiguous — but it is why no claim is made here about the list being small.
        let mut deferred: Vec<(PathBuf, String, String, Option<StatSig>)> = Vec::new();
        let mut tx = self.conn.unchecked_transaction()?;

        for f in &files {
            let rel = f
                .strip_prefix(root)
                .unwrap_or(f)
                .to_string_lossy()
                .replace('\\', "/");

            // Belt-and-suspenders: re-check the vault-relative path even when
            // enumerate_docs already filtered (rg may return paths enumerate
            // couldn't pre-filter due to root-prefix stripping edge cases).
            if exclude.is_excluded(&rel) {
                skipped += 1;
                continue;
            }

            let meta = std::fs::metadata(f);
            if let Ok(m) = &meta {
                if m.len() as usize > MAX_BYTES {
                    continue;
                }
            }
            // Captured BEFORE the read: a write that lands mid-read must leave a stale-looking
            // stat, never a fresh-looking one.
            let sig = meta.as_ref().ok().map(stat_sig);
            let body = match read_markdown_text(f) {
                Ok(b) => b,
                Err(_) => continue, // skip unreadable / binary
            };
            let title = derive_title(&rel, &body);
            let digest = sha256_hex(body.as_bytes());
            if !body.trim().is_empty() {
                *incoming_digests.entry(digest.clone()).or_insert(0) += 1;
            }
            let mut wrote = false;

            // 🔴 DO NOT go back to DELETE-then-INSERT here.
            //
            // `id` is a plain INTEGER PRIMARY KEY, so a delete+insert hands the row a
            // BRAND-NEW id. The old code did that unconditionally for every file, which
            // means one `recall index` renumbered all ~136k documents. Every vector table
            // keyed on doc_id was orphaned in one shot: `embeddings` (FK ON DELETE CASCADE)
            // was silently wiped and had to be recomputed, and `embeddings2` / `embeddings3`
            // (no FK) were left as dead rows pointing at ids that no longer exist — tens of thousands of
            // orphaned doc groups, recall@5 collapsing by roughly 80%, measured 2026-07-31. Dead rows
            // still occupy rank slots, so R@1 went to 0.0000.
            //
            // Match on path and keep the id. Unchanged bodies touch nothing at all, which
            // also makes a no-op re-index cheap instead of a full rewrite.
            let existing: Option<(i64, String)> = tx
                .query_row(
                    "SELECT id, body FROM documents
                     WHERE path = ?1 AND source = 'vault' AND valid_to IS NULL",
                    params![rel],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .ok();

            match existing {
                // Content identical — leave the row (and its embeddings) completely alone.
                Some((_, old_body)) if freshness::same_text_modulo_line_endings(&old_body, &body) => {
                    unchanged += 1;
                }
                // Content changed — update in place so the id, and every vector keyed to
                // it, survives. Vectors are now STALE rather than ORPHANED, which is a
                // recoverable state: `recall embed` re-embeds, and doc_id still resolves.
                Some((id, _)) => {
                    wrote = true;
                    // 🔴 `content_sha256` moves with the body; `embedded_sha256` deliberately
                    // does NOT. That difference is the whole staleness signal: the row keeps its
                    // id and its vectors, and the two hashes now disagree, so
                    // `docs_missing_chunk_embeddings` queues it for re-embedding. Updating both
                    // here would re-create the original bug in one line.
                    tx.execute(
                        "UPDATE documents SET title = ?2, body = ?3, valid_from = ?4,
                                              content_sha256 = ?5
                         WHERE id = ?1",
                        params![id, title, body, now, &digest],
                    )?;
                    updated += 1;
                }
                None => {
                    // No live row at this path. It is either a brand-new file, a tombstone
                    // coming back, or the DESTINATION of a move — and which one cannot be
                    // decided here.
                    //
                    // 🔴 Deciding it in this loop is what produced three separate defects:
                    //   * enumeration order changed the answer, because a rival candidate that
                    //     this run is about to edit still looks like a rival when it is visited
                    //     later;
                    //   * a rename plus a copy of the same content in one run let whichever file
                    //     was visited first claim the original's id and vectors;
                    //   * writing a provisional row and deleting it later meant an interrupted
                    //     run left the provisional row committed and unreconcilable, and a
                    //     concurrent writer's changes could be deleted with it.
                    //
                    // So NOTHING is written for these files here. The cheap indexed test below
                    // asks only whether a move is even conceivable — is there any other live
                    // vault document with this exact body? If not, which is the overwhelming
                    // majority, the decision is final and taken now. If so, the file is deferred
                    // and resolved after the whole enumeration, when the picture is complete.
                    //
                    // ⚠️ Note this test sees THIS transaction's own earlier inserts, so it is not
                    // true that a first index defers nothing: two identical files in an empty
                    // vault will defer the second. That is correct — they are genuinely
                    // ambiguous — but it means no assumption may be made about how small the
                    // deferred list is.
                    let could_be_a_move = !body.trim().is_empty()
                        && tx.query_row(
                            "SELECT EXISTS(
                                 SELECT 1 FROM documents
                                  WHERE content_sha256 = ?1 AND valid_to IS NULL
                                    AND source = 'vault' AND path <> '' AND path <> ?2)",
                            params![&digest, rel],
                            |r| r.get::<_, i64>(0),
                        )? == 1;

                    if could_be_a_move {
                        // Only the path is kept — the body is re-read at resolution time rather
                        // than buffered, so a vault full of duplicates cannot make this hold the
                        // corpus in RAM.
                        deferred.push((f.clone(), rel.clone(), digest.clone(), sig));
                    } else {
                        wrote = true;
                        Self::revive_or_insert(
                            &tx,
                            &rel,
                            &title,
                            &body,
                            &digest,
                            &now,
                            &mut inserted,
                            &mut revived,
                        )?;
                    }
                }
            }

            // Record the file's stat on its row (a no-op for a deferred file, which has no row yet
            // — the resolution pass stamps it). Unchanged rows are only written when the stat
            // differs, so a no-op re-index stays a read.
            if let Some(sig) = sig {
                Self::stamp_stat(&tx, &rel, sig, now_secs(), wrote)?;
            }

            n += 1;

            if n.is_multiple_of(BATCH) {
                tx.commit()?;
                self.conn
                    .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
                    .ok();
                eprintln!("recall index: {} / {} committed", n, files.len());
                tx = self.conn.unchecked_transaction()?;
            }
        }

        // ── MOVE RESOLUTION ───────────────────────────────────────────────────────────────
        //
        // Every file that MIGHT be a move was deferred, unwritten, until here. Nothing above has
        // touched the database for them, so this pass is the first and only write — there is no
        // provisional row to reconcile, an interrupted run simply leaves those files unindexed
        // and a retry re-derives everything, and the answer does not depend on the order the
        // walk happened to visit files in.
        //
        // Three outcomes per file, in this order:
        //   MOVE    — exactly one live document holds this exact body and its own path is gone
        //             from disk. Keep that document's id, and therefore its vectors.
        //   REVIVE  — no move, but this path has a tombstone: bring the old row back.
        //   INSERT  — a genuinely new document.
        //
        // Move BEATS revive deliberately, and the reason is the STRANDING, not the vectors.
        // Renaming a live note onto a path some other note used to occupy must move the live
        // note: revive it instead and the real document is left live at a path that no longer
        // exists, findable by search and unopenable by the user, until a prune notices.
        //
        // (An earlier version of this comment said "the tombstone has no vectors". That is not
        // an invariant — `revive_or_insert` keeps whatever vectors a tombstoned row still has,
        // and a tombstone can perfectly well carry them. It happens to be true of the test
        // fixture, which is exactly how a comment like that survives.)
        // 🔴 Start the resolution phase on a FRESH transaction. Inheriting the main loop's open
        // one means that if it already holds BATCH-1 writes, this pass adds another BATCH before
        // committing — 2*BATCH-1 in one transaction, which is precisely what BATCH exists to
        // bound. Committing at the boundary also means the loop's work is durable before a pass
        // that may be long.
        if !deferred.is_empty() {
            tx.commit()?;
            self.conn
                .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
                .ok();
            tx = self.conn.unchecked_transaction()?;
        }
        let mut resolved = 0usize;
        for (fpath, rel, enumerated_digest, sig) in &deferred {
            // Counted BEFORE any `continue`. A skipped entry still has to advance the commit
            // check, or a long run of unreadable / changed / occupied destinations keeps one
            // write transaction open across the rest of the pass.
            resolved += 1;
            if resolved.is_multiple_of(BATCH) {
                tx.commit()?;
                self.conn
                    .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
                    .ok();
                eprintln!(
                    "recall index: resolved {} / {} deferred",
                    resolved,
                    deferred.len()
                );
                tx = self.conn.unchecked_transaction()?;
            }
            // Re-read rather than buffer. A file that vanished between the walk and now is
            // skipped, exactly as it would be in the loop above.
            let Ok(body) = read_markdown_text(fpath) else {
                continue;
            };
            let title = derive_title(rel, &body);
            let digest = sha256_hex(body.as_bytes());
            let blank = body.trim().is_empty();

            // 🔴 The file must still hold what it held when it was counted. Otherwise the
            // incoming-duplicate count below is about content this file no longer has: two
            // deferred files whose bodies swapped between the walk and now would each be counted
            // as unique, and each would claim the OTHER's original — an identity assignment that
            // no later run can detect or repair. Skip; the next run re-derives it correctly.
            if digest != *enumerated_digest {
                continue;
            }

            // 🔴 And the destination must still be free. "No live row at this path" was observed
            // in an earlier transaction, possibly several batch commits ago. If another writer
            // has since indexed this path, moving a document onto it would create two live
            // documents claiming one file — the identity collision invariant 3 exists to catch.
            let destination_taken: i64 = tx.query_row(
                "SELECT count(*) FROM documents
                  WHERE path = ?1 AND source = 'vault' AND valid_to IS NULL",
                params![rel],
                |r| r.get(0),
            )?;
            if destination_taken > 0 {
                continue;
            }

            // Ambiguous among the INCOMING files: two of them carry this body, so which one
            // moved is unknowable and answering would depend on walk order. Both become new
            // documents and the original is left where it is.
            let unique_incoming = incoming_digests.get(&digest).copied().unwrap_or(0) == 1;
            if let Some((old_id, old_title)) = if unique_incoming {
                Self::find_moved_document(&tx, root, rel, &digest, blank, -1)?
            } else {
                None
            } {
                // The body is unchanged by definition — it is what the hash matched on — so
                // `content_sha256` does not move and the document stays fresh.
                //
                // 🔴 Except when the TITLE changes. `derive_title` falls back to the filename,
                // so renaming a note with no H1 changes the title, and the title is part of the
                // text that was embedded. `content_sha256` covers the body only and would not
                // notice, so clear `embedded_sha256` and let it be re-embedded under its new
                // name.
                //
                // ⚠️ KNOWN HOLE, stated rather than hidden. Clearing the stamp does not stop work
                // ALREADY IN FLIGHT. An embed worker that read this document before the rename
                // finishes afterwards and certifies with `sha256(body)` — which still matches,
                // because the body did not change — so vectors built under the OLD title are
                // stamped current and the document leaves the queue. It needs a `recall embed`
                // running concurrently with an `index` that renames the same note.
                //
                // The durable fix is for certification to cover the TITLE as well as the body,
                // which means redefining what `content_sha256` hashes and re-running the
                // backfill. That is a change to a contract three other commits already depend
                // on, so it is written up rather than done here.
                if title != old_title {
                    tx.execute(
                        "UPDATE documents
                            SET path = ?2, title = ?3, valid_from = ?4, embedded_sha256 = NULL
                          WHERE id = ?1",
                        params![old_id, rel, title, now],
                    )?;
                } else {
                    tx.execute(
                        "UPDATE documents SET path = ?2, valid_from = ?3 WHERE id = ?1",
                        params![old_id, rel, now],
                    )?;
                }
                moved += 1;
            } else {
                Self::revive_or_insert(
                    &tx, rel, &title, &body, &digest, &now, &mut inserted, &mut revived,
                )?;
            }
            if let Some(sig) = sig {
                Self::stamp_stat(&tx, rel, *sig, now_secs(), true)?;
            }
        }


        tx.commit()?;
        self.conn
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
            .ok();
        if skipped > 0 {
            eprintln!(
                "recall index: skipped {} mirror-copy docs (snapshot exclude filter)",
                skipped
            );
        }
        Ok((n, IngestCounts { new: inserted, updated, moved, restored: revived, unchanged }))
    }

    /// Record one enumeration so later runs have something to compare against.
    fn record_enumeration(
        &self,
        enumerated: usize,
        used_fallback: bool,
        tombstoned: usize,
        enum_ms: Option<u64>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO index_runs (ran_at, enumerated, used_fallback, tombstoned, enum_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                Self::now(),
                enumerated as i64,
                i64::from(used_fallback),
                tombstoned as i64,
                enum_ms.map(|m| m as i64)
            ],
        )?;
        Ok(())
    }

    /// The healthy-enumeration baseline: the MAX count over the last
    /// [`PRUNE_BASELINE_RUNS`] runs **that used the same enumeration method**.
    ///
    /// Like-for-like on purpose. rg and the fallback walk return genuinely different sets over
    /// the same tree (rg reads `.gitignore` and skips hidden dirs; the walk skips
    /// `node_modules`/`target`/`vendor`/dotdirs and reads no ignore file), so a count from one
    /// is not a measurement of the other and comparing across them produces noise in both
    /// directions. Keying the baseline on the method also stops the gate going inert on a
    /// machine where rg is never available: such a machine still accumulates walk baselines
    /// and is still protected, instead of forever having "no baseline" and always allowing.
    ///
    /// `None` means no run with this method has been recorded — there is no baseline, and the
    /// gate says so out loud rather than inventing one.
    fn prune_enumeration_baseline(&self, used_fallback: bool) -> Result<Option<i64>> {
        let v: Option<i64> = self.conn.query_row(
            "SELECT MAX(enumerated) FROM (
                 SELECT enumerated FROM index_runs
                 WHERE used_fallback = ?1
                 ORDER BY id DESC LIMIT ?2
             )",
            params![i64::from(used_fallback), PRUNE_BASELINE_RUNS as i64],
            |r| r.get(0),
        )?;
        Ok(v)
    }

    /// Tombstone documents whose file has been deleted from the vault.
    ///
    /// `enumerate_docs`/`ingest_dir` only ever insert and update: nothing has ever marked a
    /// document dead when its file disappeared. Measured 2026-08-18 against the live store:
    /// 69,596 live docs, 58,959 whose file still existed, **10,637 (15.3%) whose file was
    /// gone** — a full `recall index` immediately after the deletion left that number
    /// unchanged. Those rows keep their embeddings and keep competing for top-5 slots.
    ///
    /// The mechanism is the bi-temporal `valid_to` column the schema already has, so this is
    /// reversible: `UPDATE documents SET valid_to = NULL WHERE id IN (…)` puts a row back,
    /// and the id — the join key every vector table depends on — is never touched.
    ///
    /// 🔴 **Two independent things must agree before any row is tombstoned.**
    ///
    /// 1. The path is not in the enumerated set. This alone is nowhere near sufficient. The
    ///    enumerated set is what `ingest_dir` would INGEST: `enumerate_docs` has already
    ///    dropped every `is_excluded` path, `rg --files` additionally honours `.gitignore`
    ///    and skips hidden directories, files over 512 KB are never ingested, and the
    ///    fallback walk skips `node_modules`/`target`/`vendor` outright. Diffing against it
    ///    would tombstone every excluded-but-present document in the vault — ~29% of the
    ///    corpus — for the crime of being excluded.
    /// 2. The filesystem positively reports the file absent ([`path_is_absent`]). This is the
    ///    only authority. "Not enumerated" is a candidate; "NotFound on disk" is a verdict.
    ///
    /// On top of that sits [`prune_gate_verdict`], which refuses the whole pass when the
    /// enumeration itself looks untrustworthy. The gate catches the failure the per-file
    /// check cannot: an unmounted drive or a wrong `RECALL_VAULT` makes *every* file report
    /// NotFound, and per-file logic would then dutifully tombstone the entire corpus.
    ///
    /// The full list is always written to `list_dir` — in both modes — so the set is
    /// inspectable before it is approved and recoverable after it is applied.
    pub fn prune_missing(
        &self,
        root: &Path,
        mode: PruneMode,
        list_dir: &Path,
    ) -> Result<PruneReport> {
        self.prune_missing_inner(root, mode, list_dir, rg_available())
    }

    fn prune_missing_inner(
        &self,
        root: &Path,
        mode: PruneMode,
        list_dir: &Path,
        rg_present: bool,
    ) -> Result<PruneReport> {
        // Preflight: without rg the enumeration is the fallback walk (no .gitignore, skips
        // dotdirs), which is not a trustworthy basis for tombstoning. Refuse instead of degrading.
        if !rg_present {
            warn_rg_missing();
            anyhow::bail!("prune refused, no document was tombstoned: {RG_MISSING}");
        }
        // Disbelieve the vault root first. It is the single input that could turn every
        // document in the store into a tombstone in one pass, and it is exactly what a
        // detached drive, a bad RECALL_VAULT, or a lost network share looks like.
        let root_meta = std::fs::metadata(root).with_context(|| {
            format!(
                "vault root {} is not readable — refusing to prune (every file would look deleted)",
                root.display()
            )
        })?;
        if !root_meta.is_dir() {
            anyhow::bail!(
                "vault root {} is not a directory — refusing to prune",
                root.display()
            );
        }

        let exclude = PathExcludeList::load();
        let Enumeration {
            files,
            used_fallback,
            elapsed_ms,
        } = enumerate_docs(root, &exclude)?;
        if used_fallback {
            // rg is installed but this run failed or timed out: the fallback walk ignores
            // .gitignore and skips dotdirs, so its file set is no basis for tombstoning.
            anyhow::bail!(
                "prune refused, no document was tombstoned: the rg enumeration failed this run (see the line above); re-run when `rg --files` works on the vault"
            );
        }
        let enumerated = files.len();
        let baseline = self.prune_enumeration_baseline(used_fallback)?;

        if let Err(reason) = prune_gate_verdict(enumerated, baseline, used_fallback) {
            // Loud, and non-zero via the Err below. A silent skip is the failure mode this
            // codebase keeps paying for; if the gate trips, the operator has to see it.
            eprintln!();
            eprintln!("{}", crate::reply::paint_err(crate::reply::BAD, "PRUNE REFUSED — enumeration is not trustworthy, nothing was changed"));
            eprintln!("     {reason}");
            eprintln!(
                "     enumerated now : {enumerated}{}",
                if used_fallback {
                    "  (via the FALLBACK walk, not rg)"
                } else {
                    ""
                }
            );
            match baseline {
                Some(b) => eprintln!(
                    "     recent baseline: {b}  (max of last {PRUNE_BASELINE_RUNS} {} enumerations)",
                    if used_fallback { "fallback-walk" } else { "rg" }
                ),
                None => eprintln!("     recent baseline: none recorded"),
            }
            eprintln!("     Re-run `recall index` and check the candidate count first. If the vault");
            eprintln!("     really did shrink this much, override with:");
            eprintln!("       RECALL_CONFIRM_PRUNE_SHRINK=yes recall index --prune-missing=dry");
            eprintln!();
            anyhow::bail!("prune gate refused: {reason}");
        }

        // Vault-relative, forward-slashed, lowercased — the same shape `ingest_dir` writes
        // into `documents.path`. Lowercased because NTFS is case-insensitive: a case
        // difference between the DB and the disk is not a deletion.
        let mut on_disk: HashSet<String> = HashSet::with_capacity(enumerated);
        for f in &files {
            let rel = f
                .strip_prefix(root)
                .unwrap_or(f)
                .to_string_lossy()
                .replace('\\', "/");
            on_disk.insert(rel.to_lowercase());
        }

        // `source = 'vault'` is load-bearing: `save` observations and other non-file rows have
        // no file to be missing and must never be swept up here.
        let rows: Vec<(i64, String)> = {
            let mut stmt = self.conn.prepare(
                "SELECT id, path FROM documents
                 WHERE source = 'vault' AND valid_to IS NULL AND path != ''
                 ORDER BY path",
            )?;
            let it = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            it.collect::<rusqlite::Result<Vec<_>>>()?
        };
        let live_vault_docs = rows.len();

        let mut not_enumerated = 0usize;
        let mut missing: Vec<(i64, String)> = Vec::new();
        for (id, path) in rows {
            let rel = path.replace('\\', "/");
            if on_disk.contains(&rel.to_lowercase()) {
                continue;
            }
            not_enumerated += 1;
            // Step 2 — the filesystem, not the enumeration, decides.
            if path_is_absent(root, &rel) {
                missing.push((id, path));
            }
        }

        let list_path = write_prune_list(
            list_dir,
            mode,
            &missing,
            enumerated,
            baseline,
            used_fallback,
            not_enumerated,
            live_vault_docs,
        )?;

        let tombstoned = match mode {
            PruneMode::Dry => 0,
            PruneMode::Apply => {
                let now = Self::now();
                let tx = self.conn.unchecked_transaction()?;
                let mut n = 0usize;
                for (id, _) in &missing {
                    // Sets valid_to only. `superseded_by` stays NULL — nothing replaced this
                    // document, its file is gone — matching `supersede_matching`'s shape. The
                    // id and every vector keyed to it are untouched, so this is reversible.
                    n += tx.execute(
                        "UPDATE documents SET valid_to = ?2 WHERE id = ?1 AND valid_to IS NULL",
                        params![id, now],
                    )?;
                }
                tx.commit()?;
                self.conn
                    .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
                    .ok();
                n
            }
        };

        // Dry mode records NOTHING. "Changing nothing" has to include the run history, or
        // `--prune-missing=dry` is a write against the live store dressed up as an inspection.
        // The baseline is fed by `ingest_dir` on every ordinary `recall index`, so it is not
        // starved by this.
        if mode == PruneMode::Apply {
            self.record_enumeration(enumerated, used_fallback, tombstoned, Some(elapsed_ms))?;
        }

        Ok(PruneReport {
            enumerated,
            used_fallback,
            baseline,
            live_vault_docs,
            not_enumerated,
            missing: missing.into_iter().map(|(_, p)| p).collect(),
            tombstoned,
            list_path,
        })
    }

    /// Ingest an explicit list of vault-relative docs. This is the safe repair
    /// path for coverage gaps: callers can dry-run the exact batch before any
    /// broad vault scan or production mutation.
    pub fn ingest_paths(&self, root: &Path, paths: &[PathBuf], dry_run: bool) -> Result<usize> {
        const MAX_BYTES: usize = MAX_DOC_BYTES;

        let exclude = PathExcludeList::load();
        let mut candidates = Vec::new();

        for raw in paths {
            let rel = normalize_requested_path(root, raw)?;
            if exclude.is_excluded(&rel) {
                continue;
            }
            let ext = Path::new(&rel)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            if ext != "md" && ext != "qmd" {
                continue;
            }
            let abs = resolve_ingest_abs_path(root, &rel)?;
            let meta = match std::fs::metadata(&abs) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if !meta.is_file() || meta.len() as usize > MAX_BYTES {
                continue;
            }
            candidates.push((abs, rel));
        }

        if dry_run {
            let mut n = 0usize;
            for (abs, rel) in &candidates {
                let body = match read_markdown_text(abs) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let title = derive_title(rel, &body);
                let existing = {
                    let mut stmt = self.conn.prepare(
                        "SELECT title, body FROM documents
                         WHERE lower(replace(path, '\\', '/')) = lower(replace(?1, '\\', '/'))
                           AND valid_to IS NULL
                         ORDER BY id",
                    )?;
                    let rows = stmt.query_map(params![rel], |r| {
                        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
                    })?;
                    rows.collect::<rusqlite::Result<Vec<_>>>()?
                };
                if existing.len() > 1
                    && existing.iter().any(|(existing_title, existing_body)| {
                        existing_title != &title || existing_body != &body
                    })
                {
                    anyhow::bail!(
                        "path-list refresh blocked by divergent case-equivalent valid paths"
                    );
                }
                if existing.is_empty() || existing[0] != (title, body) {
                    n += 1;
                }
            }
            return Ok(n);
        }

        let now = Self::now();
        let tx = self.conn.unchecked_transaction()?;
        let mut n = 0usize;
        for (abs, rel) in candidates {
            let sig = std::fs::metadata(&abs).ok().map(|m| stat_sig(&m));
            let body = match read_markdown_text(&abs) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let title = derive_title(&rel, &body);
            let existing = {
                let mut stmt = tx.prepare(
                    "SELECT id, title, body FROM documents
                     WHERE lower(replace(path, '\\', '/')) = lower(replace(?1, '\\', '/'))
                       AND valid_to IS NULL
                     ORDER BY id",
                )?;
                let rows = stmt.query_map(params![rel], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()?
            };
            if existing.len() > 1 {
                if existing.iter().all(|(_, existing_title, existing_body)| {
                    existing_title == &title && existing_body == &body
                }) {
                    continue;
                }
                anyhow::bail!("path-list refresh blocked by divergent case-equivalent valid paths");
            }
            if existing
                .first()
                .map(|(_, existing_title, existing_body)| {
                    existing_title == &title && existing_body == &body
                })
                .unwrap_or(false)
            {
                continue;
            }
            let old_id = existing.first().map(|(existing_id, _, _)| *existing_id);
            if let Some(existing_id) = old_id {
                tx.execute(
                    "UPDATE documents SET valid_to = ?2 WHERE id = ?1 AND valid_to IS NULL",
                    params![existing_id, now],
                )?;
            }
            // A third insert site (beyond the two `apply_tiers.py`'s docstring names) — this
            // is the path-list repair route (`recall index --paths`). Same rule: brand-new
            // row, tagged at insert time, never overwriting anything.
            let tier = tier_for_path(&rel);
            tx.execute(
                "INSERT INTO documents (path, title, body, source, created_at, valid_from, valid_to, superseded_by, tier, content_sha256)
                 VALUES (?1, ?2, ?3, 'vault', ?4, ?4, NULL, NULL, ?5, ?6)",
                params![rel, title, body, now, tier, sha256_hex(body.as_bytes())],
            )?;
            if let Some(old_id) = old_id {
                let new_id = tx.last_insert_rowid();
                tx.execute(
                    "UPDATE documents SET superseded_by = ?2 WHERE id = ?1",
                    params![old_id, new_id],
                )?;
            }
            if let Some(sig) = sig {
                Self::stamp_stat(&tx, &rel, sig, now_secs(), true)?;
            }
            n += 1;
        }
        tx.commit()?;
        self.conn
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
            .ok();
        Ok(n)
    }

    // ── BGE vector layer ────────────────────────────────────────────────────

    /// Store (or replace) one document's dense vector.
    /// Replace every Qwen3 chunk row for one document, in ONE transaction.
    ///
    /// All-or-nothing on purpose. A partial replacement leaves a document whose vectors are half
    /// from the old text and half from the new, which no later check can detect: both halves are
    /// valid vectors and the row count looks right.
    pub fn replace_qwen3_chunks(
        &mut self,
        doc_id: i64,
        dim: usize,
        chunks: &[(String, i64, i64, Vec<f32>)],
        sha: &str,
    ) -> Result<usize> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "DELETE FROM qwen3_chunk_embeddings WHERE doc_id = ?1",
            [doc_id],
        )?;
        {
            let mut ins = tx.prepare(
                "INSERT INTO qwen3_chunk_embeddings
                     (doc_id, chunk_index, vec, dim, heading_path, char_start, char_end, embedded_sha256)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            )?;
            for (i, (heading, a, b, v)) in chunks.iter().enumerate() {
                anyhow::ensure!(
                    v.len() == dim,
                    "doc {doc_id} chunk {i}: vector width {} != declared dim {dim}",
                    v.len()
                );
                let mut blob = Vec::with_capacity(v.len() * 4);
                for f in v {
                    blob.extend_from_slice(&f.to_le_bytes());
                }
                ins.execute(rusqlite::params![
                    doc_id,
                    i as i64,
                    blob,
                    dim as i64,
                    heading,
                    a,
                    b,
                    sha
                ])?;
            }
        }
        tx.commit()?;
        Ok(chunks.len())
    }

    /// The staleness predicate for the Qwen3 channel, shared by the fetch and the count.
    ///
    /// One string on purpose: if the queue a run PRINTS and the work it DOES come from two
    /// different predicates, they drift, and the number stops meaning anything. That is exactly
    /// how a queue once printed a negative value on this codebase.
    fn qwen3_queue_where(&self) -> String {
        format!(
            "d.valid_to IS NULL
                AND d.body IS NOT NULL
                AND d.path <> ''{}
                AND ( NOT EXISTS (SELECT 1 FROM qwen3_chunk_embeddings q WHERE q.doc_id = d.id)
                   OR d.content_sha256 IS NULL
                   OR d.content_sha256 IS NOT (SELECT q.embedded_sha256
                                                 FROM qwen3_chunk_embeddings q
                                                WHERE q.doc_id = d.id LIMIT 1) )",
            TierFilter::load().sql_clause("d")
        )
    }

    /// How many documents still need Qwen3 vectors.
    ///
    /// 🔴 A COUNT, never `docs_needing_qwen3(huge).len()`. That is what the first version did,
    /// and it materialised ~52,000 full document BODIES - about 340 MB of text - to produce one
    /// integer. The command printed nothing and wrote nothing for ten minutes and looked hung,
    /// because it was: counting is not fetching.
    pub fn count_docs_needing_qwen3(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            &format!(
                "SELECT count(*) FROM documents d WHERE {}",
                self.qwen3_queue_where()
            ),
            [],
            |r| r.get(0),
        )?)
    }

    /// Documents that still need Qwen3 vectors: no rows, or rows built from different text.
    ///
    /// Tier-gated through the same `TierFilter` the query path uses, so this never queues a
    /// document search is configured never to return - the SKILLS lesson, applied up front.
    pub fn docs_needing_qwen3(&self, limit: usize) -> Result<Vec<(i64, String, String)>> {
        let sql = format!(
            "SELECT d.id, COALESCE(d.title, ''), d.body
               FROM documents d
              WHERE {}
              ORDER BY d.id
              LIMIT ?1",
            self.qwen3_queue_where()
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([limit as i64], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// (documents with Qwen3 vectors, total Qwen3 chunk rows, documents still queued).
    pub fn qwen3_progress(&self) -> Result<(i64, i64, i64)> {
        let docs: i64 = self.conn.query_row(
            "SELECT COUNT(DISTINCT q.doc_id) FROM qwen3_chunk_embeddings q
               JOIN documents d ON d.id = q.doc_id WHERE d.valid_to IS NULL",
            [],
            |r| r.get(0),
        )?;
        let rows: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM qwen3_chunk_embeddings", [], |r| {
                    r.get(0)
                })?;
        let queued = self.count_docs_needing_qwen3()?;
        Ok((docs, rows, queued))
    }

    /// Every live Qwen3 chunk vector as `(doc_id, vector)`, doc_id repeated per chunk.
    ///
    /// Same tier gate and same path exclusion as the BGE twin, so a swap between channels does
    /// not silently change WHICH documents are reachable - only how they are ranked.
    pub fn load_all_qwen3_chunks(&self) -> Result<Vec<(i64, Vec<f32>)>> {
        let exclude = PathExcludeList::load();
        let sql = format!(
            "SELECT q.doc_id, d.path, q.vec
             FROM qwen3_chunk_embeddings q
             JOIN documents d ON d.id = q.doc_id
             WHERE d.valid_to IS NULL{}
             ORDER BY q.doc_id, q.chunk_index
             LIMIT 1000000",
            TierFilter::load().sql_clause("d")
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Vec<u8>>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (doc_id, path, blob) = row?;
            if exclude.is_excluded(&path) {
                continue;
            }
            if blob.len() % 4 != 0 {
                continue;
            }
            let v: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            out.push((doc_id, v));
        }
        Ok(out)
    }

    /// Sample whole document bodies (title, body) for chunk-size experiments. Read-only.
    ///
    /// The chunk-size question cannot be answered from the existing chunk rows: those were cut
    /// for a 512-token model. Re-splitting real BODIES at a different target is the only way to
    /// see what a larger window actually costs.
    pub fn sample_doc_bodies(&self, limit: usize) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT COALESCE(d.title, ''), d.body
               FROM documents d
              WHERE d.valid_to IS NULL
                AND d.body IS NOT NULL
                AND length(d.body) > 200
              LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?;
        let mut out = Vec::with_capacity(limit);
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Reconstruct real chunk text for benchmarking: heading path + the body slice the chunk
    /// covers. Read-only.
    ///
    /// 🔴 `char_start`/`char_end` are BYTE offsets despite the name - see the schema comment on
    /// `bge_chunk_embeddings`. Slicing by chars here silently returned SHIFTED text on every
    /// document containing a multi-byte character, which this vault is full of. Byte-slice, and
    /// walk to a char boundary rather than panicking on one.
    pub fn sample_chunk_texts(&self, limit: usize) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.heading_path, d.title, d.body, c.char_start, c.char_end
               FROM bge_chunk_embeddings c
               JOIN documents d ON d.id = c.doc_id
              WHERE d.valid_to IS NULL
                AND d.body IS NOT NULL
                AND c.char_start IS NOT NULL
                AND c.char_end > c.char_start
              LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], |r| {
            Ok((
                r.get::<_, Option<String>>(0)?.unwrap_or_default(),
                r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, i64>(4)?,
            ))
        })?;

        let mut out = Vec::with_capacity(limit);
        for row in rows {
            let (heading, title, body, start, end) = row?;
            let mut a = (start.max(0) as usize).min(body.len());
            let mut b = (end.max(0) as usize).min(body.len());
            while a < body.len() && !body.is_char_boundary(a) {
                a += 1;
            }
            while b < body.len() && !body.is_char_boundary(b) {
                b += 1;
            }
            if b <= a {
                continue;
            }
            let slice: String = body[a..b].to_string();
            let prefix = if heading.is_empty() { title } else { heading };
            out.push(if prefix.is_empty() {
                slice
            } else {
                format!("{prefix}
{slice}")
            });
        }
        Ok(out)
    }

    pub fn upsert_embedding(&self, doc_id: i64, vec: &[u8], dim: usize) -> Result<()> {
        self.conn.execute(
            "INSERT INTO embeddings (doc_id, vec, dim) VALUES (?1, ?2, ?3)
             ON CONFLICT(doc_id) DO UPDATE SET vec = excluded.vec, dim = excluded.dim",
            params![doc_id, vec, dim as i64],
        )?;
        Ok(())
    }

    /// Store (or replace) one CHUNK vector of one document.
    ///
    /// `chunk_index` is the position the chunk had in `bge::chunk_for_embedding_spans`'s output,
    /// so `(doc_id, chunk_index)` identifies a passage and re-embedding the same document
    /// overwrites in place rather than accumulating a second copy of every chunk.
    ///
    /// The provenance columns are overwritten alongside the vector on conflict: a stale
    /// `char_start` pointing into a body that has since been edited is worse than none, because
    /// it would render a hit with text the chunk never contained.
    pub fn upsert_chunk_embedding(
        &self,
        doc_id: i64,
        chunk_index: usize,
        vec: &[u8],
        dim: usize,
        place: &ChunkPlacement<'_>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO bge_chunk_embeddings
                 (doc_id, chunk_index, vec, dim, heading_path, char_start, char_end)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(doc_id, chunk_index) DO UPDATE SET
                 vec          = excluded.vec,
                 dim          = excluded.dim,
                 heading_path = excluded.heading_path,
                 char_start   = excluded.char_start,
                 char_end     = excluded.char_end",
            params![
                doc_id,
                chunk_index as i64,
                vec,
                dim as i64,
                place.heading_path,
                place.char_start as i64,
                place.char_end as i64,
            ],
        )?;
        Ok(())
    }

    /// Bring a file that has no LIVE row at its path into the store: revive its tombstone if it
    /// has one, else insert it.
    ///
    /// Extracted because it is now reached from two places — the main loop, for files that
    /// cannot possibly be a move, and the resolution pass, for files that could have been but
    /// turned out not to be. One implementation, so the two cannot drift.
    fn revive_or_insert(
        tx: &Connection,
        rel: &str,
        title: &str,
        body: &str,
        digest: &str,
        now: &str,
        inserted: &mut usize,
        revived: &mut usize,
    ) -> Result<()> {
        // 🔴 The file may be coming BACK from a `--prune-missing` tombstone.
        //
        // Before tombstoning existed this branch could only ever mean "brand-new file", so a
        // plain INSERT was safe. It is not any more: a deleted-then-restored note would get a
        // FRESH id here and orphan every vector keyed to the old one — the 2026-07-31
        // renumbering incident in miniature. Revive the tombstoned row instead, which keeps the
        // id and everything joined to it.
        //
        // `superseded_by IS NULL` distinguishes a prune tombstone (nothing replaced it, the file
        // simply went away) from an `ingest_paths` supersede chain, whose old rows point at their
        // replacement and must stay dead.
        let revivable: Option<i64> = tx
            .query_row(
                "SELECT id FROM documents
                 WHERE path = ?1 AND source = 'vault' AND valid_to IS NOT NULL
                   AND superseded_by IS NULL
                 ORDER BY id DESC LIMIT 1",
                params![rel],
                |r| r.get(0),
            )
            .ok();
        match revivable {
            Some(id) => {
                // Deliberately does not touch `tier`, exactly like the UPDATE arm: a tier the owner
                // corrected by hand must survive a delete/restore too. Refresh the content hash,
                // leave the embedded hash alone — a note that changed while it was tombstoned
                // comes back with its old vectors and must be re-embedded.
                tx.execute(
                    "UPDATE documents
                     SET title = ?2, body = ?3, valid_from = ?4, valid_to = NULL,
                         content_sha256 = ?5
                     WHERE id = ?1",
                    params![id, title, body, now, digest],
                )?;
                *revived += 1;
            }
            None => {
                // Brand-new row: tag tier at insert time so a document is never even momentarily
                // unclassified longer than it has to be. Never touches an existing row's tier.
                let tier = tier_for_path(rel);
                tx.execute(
                    "INSERT INTO documents (path, title, body, source, created_at, valid_from, valid_to, superseded_by, tier, content_sha256)
                     VALUES (?1, ?2, ?3, 'vault', ?4, ?4, NULL, NULL, ?5, ?6)",
                    params![rel, title, body, now, tier, digest],
                )?;
                *inserted += 1;
            }
        }
        Ok(())
    }

    /// Find the live document this file was MOVED from, or `None`.
    ///
    /// 🔴 Identity is not the bytes. This is the WEAKEST of the three signals in the design —
    /// a durable identifier (frontmatter `uid:`, or the NTFS file id) beats it and survives an
    /// edit, which this cannot. It is safe only because of the guards, and every guard is here
    /// because removing it breaks a real case in this vault:
    ///
    ///  * **non-empty body** — measured on the live corpus, of the live documents there is
    ///    exactly ONE content hash shared by more than one row, and it is the SHA-256 of the
    ///    EMPTY STRING, shared by several rows. Matching on it would let any new empty file inherit
    ///    an arbitrary one of them.
    ///  * **`path <> ''`** — some live rows are CLI/orchestrator notes with no file at all, and
    ///    `Path::new("").exists()` is FALSE, so without this they would all look like "a file
    ///    that has gone missing" and be eligible to be stolen.
    ///  * **`source = 'vault'`** — only file-backed documents have a path identity to move.
    ///  * **EXACTLY ONE candidate, counted** — never `LIMIT 1`. With two identical bodies there
    ///    is no way to know which one moved, and guessing silently deletes a document from
    ///    search. Ambiguity must produce a NEW document.
    ///  * **the candidate's old path is GONE from disk** — a file that still exists was COPIED,
    ///    not moved, and a copy must never take the original's identity. Uses
    ///    [`path_is_absent`], which answers "yes" only to a real `NotFound` — a permission error
    ///    reads as present, which is the safe direction.
    ///
    /// 🔴 Uniqueness in the DATABASE is not enough, which is why this is called from a pass that
    /// runs AFTER the whole enumeration rather than inline. Rename `A.md` to `Z.md` and then copy
    /// `Z.md` to `B.md` before indexing: when `B.md` is visited there is exactly one candidate in
    /// the database and `A.md` really is gone, so every guard above passes and the COPY takes
    /// A's id and vectors. The caller must therefore also require that exactly ONE incoming file
    /// carries this content — a fact that is only known once every file has been seen.
    fn find_moved_document(
        tx: &Connection,
        root: &Path,
        rel: &str,
        digest: &str,
        body_is_blank: bool,
        exclude_id: i64,
    ) -> Result<Option<(i64, String)>> {
        if body_is_blank {
            return Ok(None);
        }
        // `exclude_id` skips the row this run just inserted for the incoming file itself.
        // Without it the count is always at least 2 - the old document and the new row - and
        // no move is ever detected. -1 excludes nothing (ids are positive).
        let candidates: i64 = tx.query_row(
            "SELECT count(*) FROM documents
              WHERE content_sha256 = ?1 AND valid_to IS NULL AND source = 'vault'
                AND path <> '' AND path <> ?2 AND id <> ?3",
            params![digest, rel, exclude_id],
            |r| r.get(0),
        )?;
        if candidates != 1 {
            return Ok(None);
        }
        let (id, old_path, old_title): (i64, String, String) = tx.query_row(
            "SELECT id, path, title FROM documents
              WHERE content_sha256 = ?1 AND valid_to IS NULL AND source = 'vault'
                AND path <> '' AND path <> ?2 AND id <> ?3",
            params![digest, rel, exclude_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        if !path_is_absent(root, &old_path) {
            return Ok(None);
        }
        Ok(Some((id, old_title)))
    }

    /// Replace a document's chunk vectors AND stamp `embedded_sha256` — in ONE transaction.
    ///
    /// `embedded_of` is the SHA-256 of the body these vectors were actually built from. It is a
    /// parameter and not `content_sha256` read back from the row, and that distinction is the
    /// whole safety property:
    ///
    /// 🔴 Embedding happens OUTSIDE this transaction and takes seconds per batch. In that gap an
    /// ingest can rewrite the document. Stamping `embedded_sha256 = content_sha256` would then
    /// certify the NEW body's hash against the OLD body's vectors — the document leaves the
    /// queue looking fresh while carrying vectors for text it no longer contains. That is
    /// exactly the bug this column exists to detect, reintroduced by the fix for it.
    ///
    /// So the stamp is conditional: it only lands `WHERE content_sha256 IS NULL OR
    /// content_sha256 = embedded_of`. 🔴 And if it does NOT land, the WHOLE replacement rolls
    /// back — the vectors are not kept either. Keeping them looks harmless and is not:
    ///
    ///   1. body A is embedded correctly; both hashes are A and the vectors are A's.
    ///   2. an ingest writes body B. `content_sha256` becomes B, `embedded_sha256` stays A.
    ///   3. the embedder reads B and starts work.
    ///   4. another ingest restores body A. Both hashes are A again.
    ///   5. the embedder returns with B's vectors. The stamp is correctly rejected — but if the
    ///      vectors were committed anyway, the row now reads body A, hashes A/A, vectors B.
    ///
    /// The document is not stale by any test the store can apply, so nothing will ever re-embed
    /// it: correct vectors were deleted and replaced with wrong ones, permanently and silently.
    /// Rolling back leaves A's vectors in place, which is the only safe outcome.
    ///
    /// **Returns whether the stamp landed**, so a caller can count and report the rejections
    /// rather than assume they cannot happen.
    ///
    /// `COALESCE(content_sha256, ?2)` also heals the legacy state: a row that predates the hash
    /// columns has `content_sha256 IS NULL`, and stamping `embedded_sha256 = content_sha256`
    /// there would write NULL, leave the document permanently queued, and make `recall embed`
    /// loop on it forever. Filling the content hash from what was embedded ends that.
    ///
    /// Deletes first, for the reason [`Store::delete_chunk_embeddings`] exists: an upsert only
    /// overwrites the indices it is handed, so a note that shrank from 8 chunks to 3 would keep
    /// chunks 3..8 of its previous body and go on matching text it no longer contains. The
    /// delete and the inserts share this transaction, so a failure part-way through cannot
    /// leave a document with a partial vector set.
    pub fn replace_chunk_embeddings(
        &self,
        doc_id: i64,
        dim: usize,
        chunks: &[(Vec<u8>, ChunkPlacement<'_>)],
        embedded_of: &str,
    ) -> Result<bool> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM bge_chunk_embeddings WHERE doc_id = ?1",
            params![doc_id],
        )?;
        {
            let mut ins = tx.prepare(
                "INSERT INTO bge_chunk_embeddings
                     (doc_id, chunk_index, vec, dim, heading_path, char_start, char_end)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for (chunk_index, (vec, place)) in chunks.iter().enumerate() {
                ins.execute(params![
                    doc_id,
                    chunk_index as i64,
                    vec,
                    dim as i64,
                    place.heading_path,
                    place.char_start as i64,
                    place.char_end as i64,
                ])?;
            }
        }
        // Stamp LAST, inside the same transaction, and only if the row still holds the body
        // these vectors were built from. See the doc comment: an unconditional stamp certifies
        // whatever the row says NOW, which after a concurrent ingest is a different document.
        //
        // This UPDATE touches neither `title` nor `body`, so it does not fire `documents_au`
        // and writes nothing to the FTS index — see that trigger's definition.
        let stamped = tx.execute(
            "UPDATE documents
                SET content_sha256  = COALESCE(content_sha256, ?2),
                    embedded_sha256 = ?2
              WHERE id = ?1
                AND (content_sha256 IS NULL OR content_sha256 = ?2)",
            params![doc_id, embedded_of],
        )?;
        if stamped != 1 {
            // Rejected: these vectors describe a body this row no longer holds. Drop the whole
            // transaction — dropping a rusqlite transaction without committing rolls it back —
            // so the document keeps the vectors it already had. See the doc comment for why
            // committing them "anyway" is the dangerous option and not the conservative one.
            return Ok(false);
        }
        tx.commit()?;
        Ok(true)
    }

    /// Drop every chunk vector of one document. Returns the number of rows removed.
    ///
    /// Required before re-writing a document's chunks: `upsert_chunk_embedding` only
    /// overwrites the indices it is given, so a note that shrank from 8 chunks to 3
    /// would keep chunks 3..8 of its PREVIOUS body — passages the document no longer
    /// contains, still scoring in every search. Same failure class as the orphaned
    /// `embeddings2` rows that collapsed recall@5.
    pub fn delete_chunk_embeddings(&self, doc_id: i64) -> Result<usize> {
        Ok(self.conn.execute(
            "DELETE FROM bge_chunk_embeddings WHERE doc_id = ?1",
            params![doc_id],
        )?)
    }

    /// Valid docs whose chunk vectors are missing OR STALE — the work queue for `recall embed`.
    ///
    /// The per-chunk twin of [`Store::docs_missing_embeddings`]. Using the OLD queue while
    /// writing the NEW table would never drain: every document would stay "missing" forever
    /// and the loop would re-embed the same batch until the process was killed.
    ///
    /// 🔴 This used to ask only "does this document have ANY vectors?", never "are they
    /// CURRENT?". `ingest_dir`'s update arm keeps a document's id, so editing a note left its
    /// old vectors attached to it permanently — the note went on being retrieved by passages it
    /// no longer contained, and no counter anywhere reported it.
    ///
    /// Three cases, and all three must be caught:
    ///   1. never embedded          — no row in `bge_chunk_embeddings`
    ///   2. embedded before the hash columns existed — `embedded_sha256 IS NULL`
    ///   3. embedded, then edited   — the two hashes differ
    ///
    /// Case 2 is why this is not written as `content_sha256 <> embedded_sha256`, nor as
    /// `content_sha256 IS NOT embedded_sha256`. `<>` against NULL is NULL, which WHERE treats
    /// as false, and `NULL IS NOT NULL` is false outright — so either form on its own silently
    /// excludes every document in the existing corpus, which is exactly the set that needs
    /// re-embedding most. The `IS NULL` branch is not optional.
    pub fn docs_missing_chunk_embeddings(
        &self,
        limit: usize,
    ) -> Result<Vec<(i64, String, String)>> {
        let needs = needs_chunk_embedding_clause();
        let mut stmt = self.conn.prepare(&format!(
            "SELECT d.id, d.title, d.body
               FROM documents d
              WHERE {needs}
              ORDER BY d.id
              LIMIT ?1"
        ))?;
        let rows = stmt.query_map(params![limit as i64], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Every count a guarded mutation checks, from ONE statement.
    ///
    /// One statement so the numbers share a snapshot. Read separately, a writer landing between
    /// them can make a guard compare figures that never coexisted — and a guard that can be
    /// fooled by a race is not a guard.
    pub fn guard_counts(&self) -> Result<GuardCounts> {
        let needs = needs_chunk_embedding_clause();
        Ok(self.conn.query_row(
            &format!(
                "SELECT
                   (SELECT count(*) FROM documents WHERE valid_to IS NULL),
                   (SELECT count(*) FROM bge_chunk_embeddings),
                   (SELECT count(*) FROM documents WHERE content_sha256 IS NULL),
                   (SELECT count(*) FROM documents d WHERE {needs}),
                   (SELECT count(*) FROM documents d
                     WHERE d.valid_to IS NULL AND d.content_sha256 IS NOT NULL
                       AND d.embedded_sha256 IS NULL
                       AND EXISTS (SELECT 1 FROM bge_chunk_embeddings c WHERE c.doc_id = d.id)),
                   (SELECT count(*) FROM bge_chunk_embeddings c
                     LEFT JOIN documents d ON d.id = c.doc_id WHERE d.id IS NULL)"
            ),
            [],
            |r| {
                Ok(GuardCounts {
                    live_docs: r.get(0)?,
                    chunk_vectors: r.get(1)?,
                    content_sha_null: r.get(2)?,
                    queue: r.get(3)?,
                    stampable: r.get(4)?,
                    orphan_vectors: r.get(5)?,
                })
            },
        )?)
    }

    /// Record that a document's EXISTING vectors were built from its current body.
    ///
    /// 🔴 This does NOT embed anything. It writes the missing RECORD of what the vectors already
    /// stored were built from. It is only correct when those vectors really are current, which is
    /// a claim about the world that this function cannot check — the caller must have verified it
    /// (see `--verify-sample`). That is why it is not exposed as a bare command.
    ///
    /// Deliberately narrow: only live documents that HAVE vectors, HAVE a content hash, and have
    /// NO stamp. A document with no vectors must stay queued; a document whose stamp already
    /// disagrees with its content hash is genuinely stale and must not be silenced.
    ///
    /// Touches neither `title` nor `body`, so `documents_au` does not fire and the FTS index is
    /// not rewritten.
    pub fn backfill_embedded_stamp(&self) -> Result<usize> {
        let tx = self.conn.unchecked_transaction()?;
        let n = tx.execute(
            "UPDATE documents
                SET embedded_sha256 = content_sha256
              WHERE valid_to IS NULL
                AND content_sha256 IS NOT NULL
                AND embedded_sha256 IS NULL
                AND EXISTS (SELECT 1 FROM bge_chunk_embeddings c WHERE c.doc_id = documents.id)",
            [],
        )?;
        tx.commit()?;
        Ok(n)
    }

    /// How many live documents still need (re-)embedding — the size of the queue, exactly.
    ///
    /// Shares [`NEEDS_CHUNK_EMBEDDING`] with `docs_missing_chunk_embeddings`, so the number a
    /// run prints and the work it actually has cannot drift apart. `count_chunk_embedded_docs`
    /// is NOT a substitute: it counts documents that have any chunk row at all, which after the
    /// staleness fix includes every stale document, and using it as "already done" is what made
    /// a run report more documents finished than the corpus contains.
    pub fn count_docs_needing_chunk_embeddings(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            &format!(
                "SELECT count(*) FROM documents d WHERE {}",
                needs_chunk_embedding_clause()
            ),
            [],
            |r| r.get(0),
        )?)
    }

    /// Live document count and outstanding-embedding count, from ONE statement.
    ///
    /// Two separate queries are two snapshots: an ingest between them can make the queue larger
    /// than the corpus and print "2 of 1 still queued", or make a corpus-minus-queue subtraction
    /// go negative. One SELECT with two scalar subqueries is evaluated in a single read
    /// transaction, so the pair is always internally consistent.
    pub fn embedding_progress(&self) -> Result<(i64, i64)> {
        let needs = needs_chunk_embedding_clause();
        Ok(self.conn.query_row(
            &format!(
                "SELECT (SELECT count(*) FROM documents WHERE valid_to IS NULL),
                        (SELECT count(*) FROM documents d WHERE {needs})"
            ),
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?)
    }

    /// How many live DOCUMENTS have at least one chunk vector.
    ///
    /// Counts documents, not rows, so it stays comparable with `count_embeddings()` and
    /// with `count_valid()` in the progress line. The row count is several times higher
    /// and reporting it as "docs vectorized" would read as progress past 100%.
    pub fn count_chunk_embedded_docs(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(DISTINCT c.doc_id)
             FROM bge_chunk_embeddings c
             JOIN documents d ON d.id = c.doc_id
             WHERE d.valid_to IS NULL",
            [],
            |r| r.get(0),
        )?)
    }

    /// Every live chunk vector, as `(doc_id, vector)` with doc_id REPEATED once per chunk.
    ///
    /// The per-chunk twin of [`Store::load_all_embeddings`]: same tier gate, same query-time
    /// path exclusion, same deterministic ordering requirement. Callers must aggregate to one
    /// score per doc — see [`crate::bge::best_per_doc`].
    pub fn load_all_chunk_embeddings(&self) -> Result<Vec<(i64, Vec<f32>)>> {
        self.load_chunk_embeddings_filtered(None)
    }

    /// [`Store::load_all_chunk_embeddings`] restricted to some documents: what the warm daemon
    /// re-reads after a refresh re-embeds them, so its in-RAM matrix follows the database
    /// without a 17 s full reload. Same tier gate and path exclusion — one query, two callers.
    pub fn load_chunk_embeddings_for_docs(&self, ids: &[i64]) -> Result<Vec<(i64, Vec<f32>)>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }
        self.load_chunk_embeddings_filtered(Some(ids))
    }

    fn load_chunk_embeddings_filtered(&self, only: Option<&[i64]>) -> Result<Vec<(i64, Vec<f32>)>> {
        let exclude = PathExcludeList::load();
        // i64 ids formatted as integers: not injectable, and SQLite caps bound parameters.
        let only_clause = only
            .map(|ids| format!(" AND c.doc_id IN ({})", ids.iter().map(i64::to_string).collect::<Vec<_>>().join(",")))
            .unwrap_or_default();
        let sql = format!(
            // LIMIT is 1,000,000 rather than the 200,000 used for `embeddings`: that guard
            // rail counts DOCUMENTS there and CHUNKS here, and the vault averages several
            // chunks per document, so keeping 200,000 would silently drop most of the corpus
            // out of the in-RAM matrix.
            //
            // ⚠️ It is a CEILING, not an allocation — the matrix is grown row by row, so this
            // costs nothing until the corpus actually reaches it. Do not size a RAM estimate
            // off this number: the real row count is (docs SURVIVING the tier gate and the
            // path exclusion below) x chunks-per-doc, which is a small fraction of the raw
            // document count. At 384-dim f32 the ceiling would be ~1.5 GB; the live matrix is
            // far smaller.
            "SELECT c.doc_id, d.path, c.vec
             FROM bge_chunk_embeddings c
             JOIN documents d ON d.id = c.doc_id
             WHERE d.valid_to IS NULL{}{}
             ORDER BY c.doc_id, c.chunk_index
             LIMIT 1000000",
            TierFilter::load().sql_clause("d"),
            only_clause
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Vec<u8>>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (id, path, bytes) = r?;
            if is_query_excluded_path(&exclude, &path) {
                continue;
            }
            out.push((id, crate::bge::from_bytes(&bytes)));
        }
        Ok(out)
    }

    /// Legacy pooled vectors for docs that have NO chunk vectors yet.
    ///
    /// 🔴 THIS IS WHAT MAKES A PARTIAL RE-EMBED SAFE. `BgeSearch::load` originally preferred the
    /// chunk table whenever it was non-EMPTY, which is only correct if the re-embed is atomic —
    /// and it is not. Observed live: only a few hundred documents had chunk rows, so the query matrix
    /// silently shrank from over 100k pooled vectors to a few thousand chunks and the semantic channel was
    /// serving 0.7% of the corpus while looking perfectly healthy. Emptiness is the wrong test.
    ///
    /// Unioning the two instead makes the rollout MONOTONIC: every document keeps whatever vector
    /// it has, and each one upgrades from pooled to per-chunk the moment it is re-embedded.
    /// Coverage can only go up, and it never matters how far through the embed run you are.
    pub fn load_pooled_embeddings_without_chunks(&self) -> Result<Vec<(i64, Vec<f32>)>> {
        let exclude = PathExcludeList::load();
        let sql = format!(
            "SELECT e.doc_id, d.path, e.vec
             FROM embeddings e
             JOIN documents d ON d.id = e.doc_id
             WHERE d.valid_to IS NULL{}
               AND NOT EXISTS (
                   SELECT 1 FROM bge_chunk_embeddings c WHERE c.doc_id = e.doc_id
               )
             ORDER BY e.doc_id
             LIMIT 200000",
            TierFilter::load().sql_clause("d")
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Vec<u8>>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (id, path, bytes) = r?;
            if is_query_excluded_path(&exclude, &path) {
                continue;
            }
            out.push((id, crate::bge::from_bytes(&bytes)));
        }
        Ok(out)
    }

    /// Valid docs that still lack a vector — the work queue for `recall embed`.
    /// Returns (id, body) so the caller can embed without a second lookup.
    /// Returns (id, title, body) with title SEPARATE, not pre-joined.
    ///
    /// It used to return `d.title || ' ' || d.body` as one string. That was fine while the
    /// embedder swallowed a whole document, but the chunker needs the title on its own so it
    /// can prepend it to EVERY chunk - otherwise chunk 4 of a long note is an anonymous
    /// fragment with nothing to say what it belongs to.
    ///
    /// 🔴 SUPERSEDED by [`Store::docs_missing_chunk_embeddings`] on the embed path — it joins
    /// the legacy one-row-per-doc `embeddings` table, which nothing writes any more. Kept
    /// (with its table) so an index built before the per-chunk change can still be inspected
    /// and re-embedded from either side during the rollout.
    #[allow(dead_code)]
    pub fn docs_missing_embeddings(&self, limit: usize) -> Result<Vec<(i64, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT d.id, d.title, d.body
             FROM documents d
             LEFT JOIN embeddings e ON e.doc_id = d.id
             WHERE d.valid_to IS NULL AND e.doc_id IS NULL
             ORDER BY d.id
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Does any SEARCHABLE document (live, allowed tier, path not excluded — exactly what
    /// `fts_search` can return) hold `term`?
    pub fn term_searchable(&self, term: &str) -> Result<bool> {
        self.any_term_searchable(std::slice::from_ref(&term.to_string()))
    }

    /// [`Store::term_searchable`] for an OR of terms, in one FTS5 query. Stops at the first
    /// searchable row.
    pub fn any_term_searchable(&self, terms: &[String]) -> Result<bool> {
        Ok(self.searchable_rows(terms, 1)? > 0)
    }

    /// Does one SEARCHABLE document hold EVERY term (an FTS5 AND)? A phantom probe must not.
    pub fn some_doc_holds_all(&self, terms: &[String]) -> Result<bool> {
        let expr = terms
            .iter()
            .map(|t| t.replace('"', ""))
            .filter(|t| !t.is_empty())
            .map(|t| format!("\"{t}\""))
            .collect::<Vec<_>>()
            .join(" AND ");
        if expr.is_empty() {
            return Ok(false);
        }
        let exclude = PathExcludeList::load();
        let sql = format!(
            "SELECT d.path FROM documents_fts JOIN documents d ON d.id = documents_fts.rowid \
             WHERE documents_fts MATCH ?1 AND d.valid_to IS NULL{}{}",
            TierFilter::load().sql_clause("d"),
            // Scoped: the abstain judge's answer would otherwise reveal whether a term exists in
            // a box this reader cannot see (the mail `holds()` oracle, Codex tenancy r2).
            crate::scope::clause("d")
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params![expr])?;
        while let Some(r) = rows.next()? {
            let path: String = r.get(0)?;
            if !is_query_excluded_path(&exclude, &path) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Every SEARCHABLE document's id (live, allowed tier, path not excluded) in (path, id) order, and a
    /// fingerprint of exactly that set: FNV-1a over each (path, content) identity (`content_sha256`, else the
    /// body itself, never a timestamp). Any insert, tombstone, revive, rename, in-place edit, tier or exclude
    /// change that alters what fts5 can return changes it (Codex abstain-cal r1 #2: count / max / sum of ids
    /// collided on different sets), while two indexes of one vault, which number docs differently, agree
    /// (Codex det1 #2/#3). One scan of the live rows. Also returns a bottom-[`SKETCH_K`] MinHash sketch of the
    /// same identities, so two corpus states can be compared ([`sketch_similarity`]) without storing either.
    ///
    /// Cached per view (the query + exclude list) on this connection until the database changes:
    /// `PRAGMA data_version` moves on another connection's commit, `total_changes()` on this
    /// connection's own writes. The daemon looks this up on every near-miss query, and on a 70k-doc
    /// vault the scan is the bulk of that lookup.
    pub fn searchable_snapshot(&self) -> Result<Snapshot> {
        let exclude = PathExcludeList::load();
        let sql = format!(
            "SELECT d.id, d.path, coalesce(d.content_sha256, d.body, '') FROM documents d              WHERE d.valid_to IS NULL{}{} ORDER BY d.path, d.id",
            TierFilter::load().sql_clause("d"),
            // Scoped: the phantom null is calibrated over what THIS reader can search (Codex tenancy
            // b1), and the fingerprint then differs per reader view by construction.
            crate::scope::clause("d")
        );
        let view = format!("{sql}\n{}", exclude.patterns.join("\n"));
        let version: (i64, i64) = self.conn.query_row(
            "SELECT (SELECT data_version FROM pragma_data_version), total_changes()",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        // Inside a transaction the token does not move on a rollback, so never cache there (Codex
        // conc r1): the calibrator's SAVEPOINT scans fresh; the request path runs in autocommit.
        if !self.conn.is_autocommit() {
            return self.scan_searchable(&exclude, &sql);
        }
        if let Some((v, snap)) = self.snapshots.borrow().get(&view) {
            if *v == version {
                return Ok(snap.clone());
            }
        }
        let snap = self.scan_searchable(&exclude, &sql)?;
        let mut cache = self.snapshots.borrow_mut();
        if cache.len() >= SNAPSHOT_CACHE_VIEWS {
            cache.clear(); // ponytail: wholesale eviction; a daemon serves a handful of views
        }
        cache.insert(view, (version, snap.clone()));
        Ok(snap)
    }

    fn scan_searchable(&self, exclude: &PathExcludeList, sql: &str) -> Result<Snapshot> {
        #[cfg(test)]
        tests::SNAPSHOT_SCANS.with(|c| c.set(c.get() + 1));
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut ids = Vec::new();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut sketch = std::collections::BTreeSet::new();
        while let Some(r) = rows.next()? {
            let path: String = r.get(1)?;
            if is_query_excluded_path(&exclude, &path) {
                continue;
            }
            let id: i64 = r.get(0)?;
            let ver: String = r.get(2)?;
            let mut item: u64 = 0xcbf2_9ce4_8422_2325;
            // (path, content) identity, not the doc id: two indexes of one vault number docs differently
            for b in path.as_bytes().iter().chain(&[0xfe]).chain(ver.as_bytes()).chain(&[0xff]) {
                h = (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
                item = (item ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
            }
            // splitmix64 finaliser: FNV's low bits are too regular for a MinHash
            item = (item ^ (item >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            item = (item ^ (item >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            sketch.insert(item ^ (item >> 31));
            if sketch.len() > SKETCH_K {
                sketch.pop_last();
            }
            ids.push(id);
        }
        Ok((format!("{}:{h:016x}", ids.len()), ids, sketch.into_iter().collect()))
    }

    /// Every SEARCHABLE document as (id, path, content version), ordered by (path, id): an order that does not
    /// depend on the order the files were enumerated and indexed in (rg walks a directory differently each run,
    /// so doc ids differ between two fresh indexes of the same vault). Used to sample phantom probes.
    pub fn searchable_by_path(&self) -> Result<Vec<(i64, String, String)>> {
        let exclude = PathExcludeList::load();
        let sql = format!(
            "SELECT d.id, d.path, coalesce(d.content_sha256, d.body, '') FROM documents d              WHERE d.valid_to IS NULL{}{} ORDER BY d.path, d.id",
            TierFilter::load().sql_clause("d"),
            crate::scope::clause("d") // the same scoped corpus as searchable_snapshot
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            let path: String = r.get(1)?;
            if !is_query_excluded_path(&exclude, &path) {
                out.push((r.get(0)?, path, r.get(2)?));
            }
        }
        Ok(out)
    }

    /// Run `f` inside one read snapshot (a SAVEPOINT), so a calibration's fingerprint, document
    /// list, bodies and probe judgements all see the same corpus state (Codex abstain-cal r1 #3).
    pub fn in_snapshot<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        self.conn.execute_batch("SAVEPOINT abstain_cal")?;
        let out = f();
        self.conn.execute_batch("RELEASE abstain_cal")?;
        out
    }

    /// A stored abstain calibration (`query::phantom_null`) for `key`, if any.
    pub fn calibration_get(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM abstain_calibration WHERE key = ?1", params![key], |r| r.get(0))
            .optional()?)
    }

    /// The newest stored abstain calibration whose key starts with `prefix` (a calibration version).
    pub fn calibration_latest(&self, prefix: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT value FROM abstain_calibration WHERE substr(key, 1, length(?1)) = ?1                  ORDER BY rowid DESC LIMIT 1",
                params![prefix],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// Store an abstain calibration, keeping the [`CALIBRATION_KEEP`] newest keys (a process with
    /// another tier / exclude policy has its own key and must not evict this one every query).
    pub fn calibration_put(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO abstain_calibration(key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        // Retention is per FAMILY: the key minus its fingerprint, i.e. version + reader policy. The
        // fingerprint is `<count>:<hash>` (`searchable_snapshot`), so the family ends before the
        // SECOND-to-last ':' (Codex: cutting at the last ':' kept the doc count in the family, so
        // every count change opened a new bucket). Four agent views calibrating in turn therefore
        // cannot evict the operator's null and make every view recompute ~200 judged probes per
        // query (CodeRabbit, tenancy PR). A global ceiling still bounds the table.
        let family = key
            .rfind(':')
            .and_then(|last| key[..last].rfind(':'))
            .map_or(key, |i| &key[..=i]);
        self.conn.execute(
            "DELETE FROM abstain_calibration
              WHERE substr(key, 1, length(?1)) = ?1
                AND rowid NOT IN (SELECT rowid FROM abstain_calibration
                                   WHERE substr(key, 1, length(?1)) = ?1
                                   ORDER BY rowid DESC LIMIT ?2)",
            params![family, CALIBRATION_KEEP as i64],
        )?;
        self.conn.execute(
            "DELETE FROM abstain_calibration WHERE rowid NOT IN              (SELECT rowid FROM abstain_calibration ORDER BY rowid DESC LIMIT ?1)",
            params![(CALIBRATION_KEEP * 8) as i64],
        )?;
        Ok(())
    }

    /// Searchable documents holding `term`, counted up to `cap` (the abstain WEIGHT's df). Only
    /// searchable text counts (Codex loop2-r1 #6: excluded copies of a term used to lower its weight
    /// and could flip a verdict without any searchable content changing).
    pub fn term_searchable_df(&self, term: &str, cap: usize) -> Result<usize> {
        self.searchable_rows(std::slice::from_ref(&term.to_string()), cap)
    }

    /// Count searchable rows matching any of `terms`, stopping at `cap` eligible rows. Eligibility
    /// is applied to every row BEFORE it is counted, never after a SQL `LIMIT` (Codex loop2-r1 #5).
    ///
    /// EXACT, with no scan budget (loop2 r3, Codex r2 HIGH #2): the round-2 budget turned 50,000
    /// excluded occurrences into `df = cap`, so excluded text made an unknown entity "findable" and
    /// a typo a required literal — excluded documents vetoed or admitted queries again. The scan
    /// only reads live rows (`valid_to IS NULL`), so it is bounded by the live corpus, and it stops
    /// at `cap` eligible rows; only a term held almost entirely by excluded documents reads far.
    pub(crate) fn searchable_rows(&self, terms: &[String], cap: usize) -> Result<usize> {
        let expr = terms
            .iter()
            .map(|t| t.replace('"', ""))
            .filter(|t| !t.is_empty())
            .map(|t| format!("\"{t}\""))
            .collect::<Vec<_>>()
            .join(" OR ");
        if expr.is_empty() || cap == 0 {
            return Ok(0);
        }
        let exclude = PathExcludeList::load();
        let sql = format!(
            "SELECT d.path FROM documents_fts JOIN documents d ON d.id = documents_fts.rowid              WHERE documents_fts MATCH ?1 AND d.valid_to IS NULL{}{}",
            TierFilter::load().sql_clause("d"),
            // Scoped like `some_doc_holds_all`: df over boxes the reader cannot see would let the
            // abstain verdict reveal that a private term exists.
            crate::scope::clause("d")
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params![expr])?;
        let mut n = 0usize;
        while let Some(r) = rows.next()? {
            let path: String = r.get(0)?;
            if !is_query_excluded_path(&exclude, &path) {
                n += 1;
                if n == cap {
                    break;
                }
            }
        }
        Ok(n)
    }

    /// Does document `id` contain at least one of `terms` (an FTS5 OR, porter-stemmed)? Used for
    /// a typo's one-edit corrections: the supporting hit must hold a correction itself.
    pub fn doc_matches_any(&self, id: i64, terms: &[String]) -> Result<bool> {
        let expr = terms
            .iter()
            .map(|t| t.replace('"', ""))
            .filter(|t| !t.is_empty())
            .map(|t| format!("\"{t}\""))
            .collect::<Vec<_>>()
            .join(" OR ");
        if expr.is_empty() {
            return Ok(false);
        }
        crate::query::prof::time("doc_matches_any", || {
            Ok(self
                .conn
                .query_row(
                    "SELECT 1 FROM documents_fts WHERE documents_fts MATCH ?1 AND rowid = ?2",
                    params![expr, id],
                    |_| Ok(()),
                )
                .optional()?
                .is_some())
        })
    }

    /// Does document `id` contain `term` (title or body), as the porter-stemmed index sees it?
    pub fn doc_matches_term(&self, id: i64, term: &str) -> Result<bool> {
        let expr = format!("\"{}\"", term.replace('"', ""));
        crate::query::prof::time("doc_matches_term", || {
            Ok(self
                .conn
                .query_row(
                    "SELECT 1 FROM documents_fts WHERE documents_fts MATCH ?1 AND rowid = ?2",
                    params![expr, id],
                    |_| Ok(()),
                )
                .optional()?
                .is_some())
        })
    }

    /// Lowercased `title || ' ' || body` for a set of doc ids, for lexical reranking.
    ///
    /// `Hit` carries only a truncated `snippet`, which is the wrong input for a coverage score —
    /// a query term present in the document but outside the snippet window would read as absent
    /// and the reranker would punish a document for the snippet's framing rather than its
    /// content.
    ///
    /// Batched at 500 to stay under SQLite's variable limit (999 by default). Lowercasing here
    /// rather than per-term at the call site keeps it one pass over the text instead of one pass
    /// per query term.
    pub fn bodies_for_ids(&self, ids: &[i64]) -> Result<HashMap<i64, String>> {
        Ok(self
            .title_body_rows(ids)?
            .into_iter()
            .map(|(id, title, body)| (id, format!("{title} {body}").to_lowercase()))
            .collect())
    }

    /// Raw `(id, title, body)` rows for `ids`, unchanged text (the cross-encoder reads case and
    /// punctuation). Batched at 500 like [`Store::bodies_for_ids`].
    pub fn title_body_rows(&self, ids: &[i64]) -> Result<Vec<(i64, String, String)>> {
        let mut out = Vec::with_capacity(ids.len());
        for batch in ids.chunks(500) {
            let placeholders = std::iter::repeat_n("?", batch.len())
                .collect::<Vec<_>>()
                .join(", ");
            // Hydration is the last line: an id that reached here from any channel still renders
            // nothing unless this reader may see its scope.
            let sql = format!(
                "SELECT id, title, body FROM documents WHERE id IN ({placeholders}){}",
                crate::scope::clause("documents")
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map(rusqlite::params_from_iter(batch.iter()), |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })?;
            for row in rows {
                out.push(row?);
            }
        }
        Ok(out)
    }

    /// Valid indexed docs from an explicit path list, WHETHER OR NOT they already have vectors.
    ///
    /// The re-embed queue. [`Store::docs_missing_chunk_embeddings_for_paths`] deliberately skips
    /// anything already embedded, which is right for a backfill and useless after a CHUNKER
    /// change: every document is already "done" by that test, so a corrected chunker would never
    /// be applied to a single one of them.
    ///
    /// 🔴 Prefer this over deleting the rows first and re-running the backfill queue. A document
    /// with no chunk rows falls back to its pooled vector — and ~19,000 live documents have no
    /// pooled vector at all, so a delete-then-embed would drop them out of semantic search
    /// entirely for the length of the run. `upsert_chunk_embedding` is a delete-then-insert per
    /// DOCUMENT, so going through this queue keeps every document continuously searchable.
    pub fn docs_for_paths(
        &self,
        root: &Path,
        paths: &[PathBuf],
    ) -> Result<Vec<(i64, String, String, String)>> {
        self.docs_for_paths_inner(root, paths, false)
    }

    /// Valid indexed docs from an explicit path list that still lack CHUNK vectors.
    ///
    /// The per-chunk twin of [`Store::docs_missing_embeddings_for_paths`], and the reason it
    /// exists is the same trap [`Store::docs_missing_chunk_embeddings`] documents: the pooled
    /// queue asks `embeddings` whether a document is done, so pairing it with a writer that
    /// fills `bge_chunk_embeddings` would re-embed the same documents forever while the query
    /// path saw nothing new.
    ///
    /// 🔴 Returns `title` and `body` SEPARATELY, which is the whole point. Its pooled sibling
    /// pre-joins them as `title || ' ' || body` in SQL, and the chunker cannot accept that:
    /// [`crate::bge::chunk_for_embedding_spans`] needs the title on its own to build the
    /// `"Title > H1 > H2"` prefix, and needs `body` offsets to line up with the stored
    /// `char_start`/`char_end` provenance. Handing it a pre-joined string would shift every
    /// offset by `title.len() + 1` and render hits with the wrong text.
    ///
    /// Ordering, de-duplication, case-folding and the 512-wide `IN (...)` batching all match the
    /// pooled version so the two behave identically from the caller's side.
    pub fn docs_missing_chunk_embeddings_for_paths(
        &self,
        root: &Path,
        paths: &[PathBuf],
    ) -> Result<Vec<(i64, String, String, String)>> {
        self.docs_for_paths_inner(root, paths, true)
    }

    /// Shared body of the two path-list queues. `only_missing` toggles the one clause that
    /// differs, so ordering, de-duplication, case-folding and batching cannot drift apart.
    fn docs_for_paths_inner(
        &self,
        root: &Path,
        paths: &[PathBuf],
        only_missing: bool,
    ) -> Result<Vec<(i64, String, String, String)>> {
        let mut seen = HashSet::new();
        let mut requested = Vec::new();
        for raw in paths {
            let rel = normalize_requested_path(root, raw)?;
            let rel_key = rel.replace('\\', "/").to_lowercase();
            if !seen.insert(rel_key) {
                continue;
            }
            let query_key = rel.replace('\\', "/").to_ascii_lowercase();
            requested.push((query_key, rel));
        }

        const LOOKUP_BATCH_SIZE: usize = 512;
        let mut rows_by_key = HashMap::new();
        for batch in requested.chunks(LOOKUP_BATCH_SIZE) {
            let placeholders = std::iter::repeat_n("?", batch.len())
                .collect::<Vec<_>>()
                .join(", ");
            let missing_clause = if only_missing {
                "AND NOT EXISTS (SELECT 1 FROM bge_chunk_embeddings c WHERE c.doc_id = d.id)"
            } else {
                ""
            };
            let sql = format!(
                "SELECT lower(replace(d.path, '\\', '/')), d.id, d.title, d.body
                 FROM documents d
                 WHERE lower(replace(d.path, '\\', '/')) IN ({placeholders})
                   AND d.valid_to IS NULL
                   {missing_clause}
                 ORDER BY d.id"
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map(
                rusqlite::params_from_iter(batch.iter().map(|(key, _)| key)),
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                },
            )?;
            for row in rows {
                let (key, id, title, body) = row?;
                rows_by_key.entry(key).or_insert((id, title, body));
            }
        }

        let mut out = Vec::new();
        for (query_key, rel) in requested {
            if let Some((id, title, body)) = rows_by_key.remove(&query_key) {
                out.push((id, title, body, rel));
            }
        }
        Ok(out)
    }

    /// Valid indexed docs from an explicit path list that still lack vectors.
    /// This is the safe repair path for embedding known coverage gaps without
    /// draining the older broad missing-embedding queue first.
    pub fn docs_missing_embeddings_for_paths(
        &self,
        root: &Path,
        paths: &[PathBuf],
    ) -> Result<Vec<(i64, String, String)>> {
        let mut seen = HashSet::new();
        let mut requested = Vec::new();
        for raw in paths {
            let rel = normalize_requested_path(root, raw)?;
            let rel_key = rel.replace('\\', "/").to_lowercase();
            if !seen.insert(rel_key) {
                continue;
            }
            let query_key = rel.replace('\\', "/").to_ascii_lowercase();
            requested.push((query_key, rel));
        }

        const LOOKUP_BATCH_SIZE: usize = 512;
        let mut rows_by_key = HashMap::new();
        for batch in requested.chunks(LOOKUP_BATCH_SIZE) {
            let placeholders = std::iter::repeat_n("?", batch.len())
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT lower(replace(d.path, '\\', '/')), d.id, d.title || ' ' || d.body
                 FROM documents d
                 LEFT JOIN embeddings e ON e.doc_id = d.id
                 WHERE lower(replace(d.path, '\\', '/')) IN ({placeholders})
                   AND d.valid_to IS NULL
                   AND e.doc_id IS NULL
                 ORDER BY d.id"
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map(
                rusqlite::params_from_iter(batch.iter().map(|(key, _)| key)),
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            )?;
            for row in rows {
                let (key, id, text) = row?;
                rows_by_key.entry(key).or_insert((id, text));
            }
        }

        let mut out = Vec::new();
        for (query_key, rel) in requested {
            if let Some((id, text)) = rows_by_key.remove(&query_key) {
                out.push((id, text, rel));
            }
        }
        Ok(out)
    }

    /// Load every stored vector for valid docs (id + decoded f32s) for brute-force
    /// cosine search. Holds the whole matrix in RAM — fine for ~100k × 384 f32.
    ///
    /// ORDER BY e.doc_id is required for determinism: without it the DB can
    /// return rows in arbitrary order, making tie-break rankings and ANN index
    /// builds non-reproducible across runs (eval ground-truth oracle is sensitive
    /// to this ordering).
    pub fn load_all_embeddings(&self) -> Result<Vec<(i64, Vec<f32>)>> {
        let exclude = PathExcludeList::load();
        // The tier gate is applied HERE, at matrix-load time, not after scoring. The daemon
        // holds this matrix resident for the life of the process, so a post-scoring filter
        // would keep paying RAM and cosine cost for rows that can never be returned.
        let sql = format!(
            // LIMIT 200_000 is a guard rail against unbounded RAM growth if the
            // corpus unexpectedly scales beyond current ~100k doc range; it is
            // not expected to be hit under normal operation.
            // At 384-dim f32 this caps RAM at ~300 MB (200k x 384 x 4 B); exceeding it would silently drop excess docs from the in-RAM matrix.
            "SELECT e.doc_id, d.path, e.vec
             FROM embeddings e
             JOIN documents d ON d.id = e.doc_id
             WHERE d.valid_to IS NULL{}
             ORDER BY e.doc_id
             LIMIT 200000",
            TierFilter::load().sql_clause("d")
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Vec<u8>>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (id, path, bytes) = r?;
            if is_query_excluded_path(&exclude, &path) {
                continue;
            }
            out.push((id, crate::bge::from_bytes(&bytes)));
        }
        Ok(out)
    }

    /// The subset of `ids` the current reader may see (`scope::current()`), or `None` for the
    /// operator (everything; no query run). The pre-fusion tenancy gate: vector channels score an
    /// unscoped resident matrix, so their candidates pass through here before any judge, dedup or
    /// fusion step can see them (Codex tenancy r1: dedup lends ids and scores across copies).
    pub fn allowed_ids(&self, ids: &[i64]) -> Result<Option<HashSet<i64>>> {
        let clause = crate::scope::clause("documents");
        if clause.is_empty() {
            return Ok(None);
        }
        let mut out = HashSet::with_capacity(ids.len());
        for batch in ids.chunks(500) {
            // i64 ids formatted as integers: not injectable.
            let list = batch.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
            let mut stmt = self
                .conn
                .prepare(&format!("SELECT id FROM documents WHERE id IN ({list}){clause}"))?;
            for id in stmt.query_map([], |r| r.get::<_, i64>(0))? {
                out.insert(id?);
            }
        }
        Ok(Some(out))
    }

    /// Fetch a single doc as a Hit (for assembling bge-channel results by id).
    pub fn hit_by_id(&self, id: i64) -> Result<Option<Hit>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT id, path, title, substr(body,1,200) FROM documents WHERE id = ?1 AND valid_to IS NULL{}",
            crate::scope::clause("documents")
        ))?;
        let mut rows = stmt.query_map(params![id], |r| {
            Ok(Hit {
                id: r.get(0)?,
                path: r.get(1)?,
                title: r.get(2)?,
                snippet: r.get(3)?,
                score: 0.0,
            })
        })?;
        match rows.next() {
            Some(h) => Ok(Some(h?)),
            None => Ok(None),
        }
    }

    /// Deterministic local graph v0: return valid docs near FTS seed docs by path.
    ///
    /// This is intentionally read-only and uses only indexed database metadata.
    /// It is not the final entity/relation KG, but it gives the graph channel a
    /// measurable adjacency source instead of an empty stub.
    pub fn graph_path_neighbors(&self, seeds: &[Hit], k: usize) -> Result<Vec<Hit>> {
        if seeds.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let mut out = Vec::new();
        let mut seen: HashSet<i64> = seeds.iter().map(|hit| hit.id).collect();

        for seed in seeds {
            if out.len() >= k {
                break;
            }

            let Some(prefix) = graph_parent_prefix(&seed.path) else {
                continue;
            };

            let pattern = format!("{}%", escape_like_prefix(&prefix));
            let limit = (k - out.len()) as i64;
            let mut stmt = self.conn.prepare(&format!(
                "SELECT id, path, title, substr(body,1,200)
                 FROM documents
                 WHERE valid_to IS NULL
                   AND path LIKE ?1 ESCAPE '\\'{}
                 ORDER BY length(path) ASC, id DESC
                 LIMIT ?2",
                crate::scope::clause("documents")
            ))?;
            let rows = stmt.query_map(params![pattern, limit], |r| {
                Ok(Hit {
                    id: r.get(0)?,
                    path: r.get(1)?,
                    title: r.get(2)?,
                    snippet: r.get(3)?,
                    score: 0.0,
                })
            })?;

            for row in rows {
                let hit = row?;
                if seen.insert(hit.id) {
                    out.push(hit);
                    if out.len() >= k {
                        break;
                    }
                }
            }
        }

        Ok(out)
    }

    pub fn count_embeddings(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM embeddings e JOIN documents d ON d.id = e.doc_id WHERE d.valid_to IS NULL",
            [],
            |r| r.get(0),
        )?)
    }

    /// Count live embedded documents whose path now matches PathExcludeList.
    ///
    /// This is a dry-run helper for stale exclude-policy drift. It reads only
    /// paths, never bodies, and does not mutate the store.
    pub fn count_exclude_policy_embeddings(&self) -> Result<i64> {
        let exclude = PathExcludeList::load();
        let mut stmt = self.conn.prepare(
            "SELECT d.path
             FROM embeddings e
             JOIN documents d ON d.id = e.doc_id
             WHERE d.valid_to IS NULL",
        )?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut count = 0_i64;
        for row in rows {
            if is_query_excluded_path(&exclude, &row?) {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Count embeddings orphaned from tombstoned documents (valid_to IS NOT NULL).
    /// Used by `recall gc` dry-run and apply for reporting.
    pub fn count_orphaned_embeddings(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM embeddings e
             JOIN documents d ON d.id = e.doc_id
             WHERE d.valid_to IS NOT NULL",
            [],
            |r| r.get(0),
        )?)
    }

    /// Delete embeddings for all tombstoned documents.
    /// NEVER touches documents whose valid_to IS NULL (live).
    /// Returns the number of embedding rows deleted.
    pub fn delete_orphaned_embeddings(&self) -> Result<usize> {
        let n = self.conn.execute(
            "DELETE FROM embeddings WHERE doc_id IN (
                 SELECT d.id FROM documents d WHERE d.valid_to IS NOT NULL
             )",
            [],
        )?;
        self.conn
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
            .ok();
        Ok(n)
    }
}

fn graph_parent_prefix(path: &str) -> Option<String> {
    let path = path.trim_matches('/');
    let (dir, _name) = path.rsplit_once('/')?;
    if dir.is_empty() {
        None
    } else {
        Some(format!("{}/", dir))
    }
}

fn escape_like_prefix(prefix: &str) -> String {
    let mut out = String::with_capacity(prefix.len());
    for ch in prefix.chars() {
        if matches!(ch, '\\' | '%' | '_') {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

/// Tokens the abstain judge reads, and the MATCH its evidence search uses ([`fts_query_tokens`],
/// [`Store::fts_judge_ids`]). Unchanged at 64: every floor in query.rs (FTS5_MATCH_CAP,
/// LONG_QUERY_MIN_SHARE, BGE_MIN_SUPPORT, SUBJECT_MAX_FINDABLE) was calibrated on it.
const MAX_FTS_TOKENS: usize = 64;
/// [`Store::prune_common_tokens`] never drops below this many tokens: the rarest ones always stay.
const FTS_KEEP_RAREST: usize = 3;
/// [`Store::prune_common_tokens`] only drops a token held by more than this many fts rows (card v2:
/// a fraction-only cut cost LoCoMo session -7.8 pp on DEV at 10%, its stores hold 20-700 documents).
const FTS_PRUNE_MIN_DF: i64 = 5000;

/// The DEV-picked cut, on by default for stores above the floor (v3 TEST measured exactly this).
const FTS_COMMON_FRAC_DEFAULT: f64 = 0.25;

/// `RECALL_FTS_COMMON_FRAC`: unset = [`FTS_COMMON_FRAC_DEFAULT`]; a fraction in (0, 1] = that cut;
/// anything else (`0`, `off`, garbage) = prune disabled.
fn fts_common_frac() -> Option<f64> {
    parse_common_frac(std::env::var("RECALL_FTS_COMMON_FRAC").ok().as_deref())
}

fn parse_common_frac(v: Option<&str>) -> Option<f64> {
    match v {
        None => Some(FTS_COMMON_FRAC_DEFAULT),
        Some(v) => v.trim().parse::<f64>().ok().filter(|f| *f > 0.0 && *f <= 1.0),
    }
}

#[cfg(test)]
mod common_frac_tests {
    use super::{parse_common_frac, FTS_COMMON_FRAC_DEFAULT};

    #[test]
    fn unset_is_the_default_and_zero_or_junk_disables() {
        assert_eq!(parse_common_frac(None), Some(FTS_COMMON_FRAC_DEFAULT));
        assert_eq!(parse_common_frac(Some("0.5")), Some(0.5));
        assert_eq!(parse_common_frac(Some(" 0.1 ")), Some(0.1));
        for off in ["0", "0.0", "off", "", "-1", "2"] {
            assert_eq!(parse_common_frac(Some(off)), None, "{off:?} disables the prune");
        }
    }
}

/// Tokens whose df is at most `max_df`, plus the [`FTS_KEEP_RAREST`] rarest; original order. Pure.
fn keep_rare(tokens: Vec<String>, df: &[i64], max_df: f64) -> Vec<String> {
    let mut order: Vec<usize> = (0..tokens.len()).collect();
    order.sort_by_key(|&i| (df[i], i));
    let rarest: std::collections::HashSet<usize> = order.into_iter().take(FTS_KEEP_RAREST).collect();
    tokens
        .into_iter()
        .enumerate()
        .filter(|(i, _)| rarest.contains(i) || df[*i] as f64 <= max_df)
        .map(|(_, t)| t)
        .collect()
}
/// Tokens the RANKING search matches (backlog #8, PLAN.md section 2). 256 covers 1,186 of 1,406
/// ArguAna queries in full; the longest question of a private evaluation set is 18. SQLite FTS5 rejects ~1,000+ terms
/// ("query too large"); test `fts_search_accepts_a_max_match_tokens_query` proves 256 is safe.
pub(crate) const MAX_MATCH_TOKENS: usize = 256;

/// bm25 column weights for `documents_fts`, whose columns are `(0 = title, 1 = body)`
/// — see the `CREATE VIRTUAL TABLE` above. `bm25()` defaults every unspecified weight to
/// 1.0, so `BM25_TITLE_WEIGHT = 8.0` is the only behavioural change from the shipped
/// `bm25(documents_fts)`. Only the RATIO matters: fts5 sums the per-column weighted term
/// frequencies, so scaling both weights leaves the ordering untouched.
///
/// Tuned 2026-08-19 by sweeping 1 → 128 over a frozen private evaluation set:
/// MRR climbs monotonically and saturates at w ≈ 32 (identical at 32/48/64/128), R@5
/// peaks at 16-24, and R@10 never moves outside the noise floor — this is a re-ordering
/// win inside the already-retrieved set, NOT a coverage win.
///
/// 🔴 8.0 is deliberately BELOW the metric optimum. The gain is not evenly distributed:
/// on the questions whose text shares a content word with the answer document's
/// title it is large (MRR +0.060 at w=8 on the high-overlap stratum), and on those that
/// share NO word with the title it is NEGATIVE at every weight ≥ 4 (MRR -0.016 at w=8,
/// 95% CI excludes 0). Raising the weight buys more of the first and more of the second.
/// 8.0 keeps 72% of the w=16 MRR gain (56% of the w ≥ 32 asymptote) while costing the
/// no-overlap stratum 2 questions at depth 10 instead of the 3 that every weight ≥ 16 costs.
///
/// The degenerate end is real and is what bounds this: `bm25(documents_fts, 1.0, 0.0)`
/// (title only) scores 96% of its top-10 on a title term and collapses to R@1 0.143 /
/// R@10 0.286 against the control's 0.343 / 0.629. Do not chase the plateau to infinity.
const BM25_TITLE_WEIGHT: f64 = 8.0;
/// Companion to [`BM25_TITLE_WEIGHT`]; 1.0 is the fts5 default for an unweighted column.
const BM25_BODY_WEIGHT: f64 = 1.0;

/// Build a robust FTS5 MATCH expression: quote each alphanumeric token and OR
/// them together. OR maximises recall (the metric we optimise) and quoting
/// neutralises punctuation that would otherwise be FTS5 syntax errors.
/// At most `MAX_FTS_TOKENS` tokens are included so large inputs (e.g. a whole
/// document pasted as a query) cannot blow FTS5's internal term limit.
fn fts_match_expr(query: &str) -> String {
    fts_match_expr_capped(query, MAX_MATCH_TOKENS)
}

fn fts_match_expr_capped(query: &str, cap: usize) -> String {
    fts_tokens_capped(query, cap)
        .iter()
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// The lowercased tokens [`fts_match_expr`] ORs together: alphanumeric runs of >= 2 bytes, the
/// first [`MAX_FTS_TOKENS`] only. The abstain rules (`query::confidence_terms`) judge exactly
/// these tokens, so a token FTS5 never searched can never veto or support a result.
pub(crate) fn fts_query_tokens(query: &str) -> Vec<String> {
    fts_tokens_capped(query, MAX_FTS_TOKENS)
}

fn fts_tokens_capped(query: &str, cap: usize) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .take(cap)
        .map(str::to_lowercase)
        .collect()
}

/// True when the ranking MATCH holds more tokens than the judge's: only then do the two differ.
pub(crate) fn exceeds_judge_cap(query: &str) -> bool {
    fts_tokens_capped(query, MAX_FTS_TOKENS + 1).len() > MAX_FTS_TOKENS
}

fn derive_title(rel: &str, body: &str) -> String {
    for line in body.lines().take(40) {
        let l = line.trim();
        if let Some(h) = l.strip_prefix("# ") {
            return h.trim().to_string();
        }
    }
    rel.rsplit('/').next().unwrap_or(rel).to_string()
}

fn read_markdown_text(path: &Path) -> Result<String> {
    Ok(stored_form(std::fs::read(path)?))
}

/// The text recall STORES for a file with these bytes: lossy UTF-8, then (opt-in
/// `scrub.store_redacted`, on for new installs) the scrub layers. The one definition behind every
/// ingest path AND `recall stored-form`, so a caller that must confirm "the index holds this
/// file's current text" (memcap) asks recall instead of re-deriving the redaction.
pub fn stored_form(bytes: Vec<u8>) -> String {
    let text = match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(&e.into_bytes()).into_owned(),
    };
    crate::scrub::scrub_stored(text)
}

fn normalize_requested_path(root: &Path, raw: &Path) -> Result<String> {
    if raw.components().any(|c| matches!(c, Component::ParentDir)) {
        anyhow::bail!("path-list entry escapes vault root: {}", raw.display());
    }

    let rel = if raw.is_absolute() {
        raw.strip_prefix(root)
            .with_context(|| format!("path-list entry outside vault root: {}", raw.display()))?
            .to_path_buf()
    } else {
        raw.to_path_buf()
    };

    Ok(rel
        .to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches('/')
        .to_string())
}

fn resolve_ingest_abs_path(root: &Path, rel: &str) -> Result<PathBuf> {
    fn join_slash_path(base: &Path, rel: &str) -> PathBuf {
        let mut out = base.to_path_buf();
        for part in rel.split('/').filter(|part| !part.is_empty()) {
            out.push(part);
        }
        out
    }

    if rel == ".claude" || rel.starts_with(".claude/") {
        let home = std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from)
            .context("USERPROFILE/HOME is required to resolve .claude path-list entries")?;
        return Ok(join_slash_path(&home, rel));
    }
    Ok(join_slash_path(root, rel))
}

// ── Path exclude filter ────────────────────────────────────────────────────────
//
// Snapshot mirror dirs (e.g. `tools/state/skill-radar/disabled-loader-roots-*`)
// contain verbatim copies of skill files already indexed at their canonical
// locations. Without filtering, a single "Optimization Reference" skill floods
// the top-5 results as 3+ identical documents under different paths.
//
// Rules: a vault-relative path is excluded when ANY of its slash-separated
// segments matches a builtin pattern OR one of the user-supplied patterns in
// RECALL_EXCLUDE_EXTRA (comma-separated prefix strings).
//
// Matching is deliberately simple — no regex, no glob library (zero new deps):
//   • exact segment match  → "disabled-loader-roots-2026-05-09-active-finder-only"
//   • prefix match         → "disabled-loader-roots"   (matches any date suffix)
//   • prefix match         → "disabled-"  (any tool-dump segment starting with "disabled-")
//
// PRECISION REQUIREMENT: patterns must only match confirmed vault tool-dump /
// mirror-copy artifacts, NOT user content folders.  Over-broad prefixes like
// "archive-", "snapshot-", "old-", "bak-" would silently drop legitimate user
// folders (e.g. "archive-2024-notes/", "old-research/").  Such generic names
// belong in RECALL_EXCLUDE_EXTRA where the user opts in explicitly.
//
// The RECALL_EXCLUDE_EXTRA env var lets users extend the list without a recompile:
//   RECALL_EXCLUDE_EXTRA="archive-2023,bak-old-skills,old-"

/// Built-in path-segment patterns that identify confirmed vault tool-dump /
/// mirror-copy directories.  Each entry is matched as an exact segment OR a
/// prefix of a segment (lowercased).
///
/// KEEP THIS LIST TIGHT — only add entries for vault-tooling artifacts whose
/// names cannot collide with user content.  Generic words like "archive",
/// "backup", "old", "snapshot" must NOT appear here as open-ended prefixes;
/// route them to RECALL_EXCLUDE_EXTRA instead.
const BUILTIN_EXCLUDE_PREFIXES: &[&str] = &[
    // skill-radar snapshot copies (the confirmed culprit):
    //   tools/state/skill-radar/disabled-loader-roots-2026-05-09-active-finder-only/…
    "disabled-loader-roots",
    // Any other tool-dump segment prefixed "disabled-" (vault loader convention).
    // This prefix is tool-specific — user notes are never named "disabled-*".
    "disabled-",
];

/// The complete, closed set of values `documents.tier` may hold. Every tier a caller asks
/// for is matched against this list and the *constant* is what reaches SQL — a request
/// string is never interpolated into a query.
pub const KNOWN_TIERS: &[&str] = &["BRAIN", "SKILLS", "VENDOR"];

/// Pure function of a vault-relative path → the `documents.tier` value it belongs in, or
/// `None` when no rule matches.
///
/// **This mirrored a private `apply_tiers.py::tier_of()` rule-for-rule** — same
/// order (first match wins: SKILLS, then VENDOR, then the BRAIN allowlist), same depth
/// guards, same case-insensitivity (that script compiles with `re.I`; here the whole path is
/// lower-cased once before any comparison). Do not invent new rules here — if a rule needs to
/// change, change it in both places or the same path will be classified two different ways
/// depending on whether it arrived via `recall index` or a manual `apply_tiers.py --apply`.
///
/// `None` is a NULL write, never the literal `"UNCLASSIFIED"` — [`KNOWN_TIERS`] is a closed
/// set and the query gate is `tier IS NULL OR tier IN (...)`; writing an unknown string would
/// match neither branch and delete that row from search.
///
/// Called from every INSERT site that lands a new `documents` row (`Store::insert`, the
/// `ingest_dir` insert branch, the `ingest_paths` insert branch) so a document is tagged the
/// moment it lands — never from an UPDATE branch. This function only ever supplies the value
/// for a brand-new row; nothing here ever overwrites a tier already stored on an existing row.
pub fn tier_for_path(rel_path: &str) -> Option<&'static str> {
    let norm = rel_path.replace('\\', "/");
    // `recall save` rows have no vault path: they are the user's own words, checked before any
    // rule because an empty string would otherwise fall through to `None`.
    if norm.trim().is_empty() {
        return Some("BRAIN");
    }
    tier_by_rules(&norm.to_lowercase(), tier_rules())
}

/// First matching rule wins; no match is `None` (NULL), which the query gate always serves.
fn tier_by_rules(p: &str, rules: &[TierRule]) -> Option<&'static str> {
    rules.iter().find(|r| r.matches(p)).map(|r| r.tier)
}

/// One path rule for `documents.tier`. Paths are vault-relative, `/`-separated, lower-cased.
#[derive(Debug, Clone, PartialEq)]
pub struct TierRule {
    pub tier: &'static str,
    pub kind: TierMatch,
    pub value: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TierMatch {
    /// path starts with `value`
    Prefix,
    /// path contains `value` anywhere
    Contains,
    /// `^value[^/]+/`: a genuine subdirectory under `value`, not a file sitting directly in it
    SubdirAfter,
    /// `^value[^/]+/raw/`: a subdirectory under `value` whose next segment is `raw`
    RawSubdirAfter,
    /// `^\d\d-[^/]*/`: inside a numbered top folder (`10-area/...`); `value` unused
    NumberedFolder,
    /// `^\d\d-[^/]*\.(md|qmd)$`: a numbered note at the root; `value` unused
    NumberedRootFile,
    /// `^[^/]+\.(md|qmd)$`: any note at the root; `value` unused
    RootNoteFile,
    /// every path; `value` unused (a catch-all, normally last)
    Any,
}

impl TierRule {
    fn matches(&self, p: &str) -> bool {
        let v = self.value.as_str();
        match self.kind {
            TierMatch::Prefix => p.starts_with(v),
            TierMatch::Contains => p.contains(v),
            TierMatch::SubdirAfter => tier_path_has_subdir_after(p, v),
            TierMatch::RawSubdirAfter => tier_path_has_raw_subdir_after(p, v),
            TierMatch::NumberedFolder => tier_path_is_numbered_vault_folder(p),
            TierMatch::NumberedRootFile => tier_path_is_numbered_root_file(p),
            TierMatch::RootNoteFile => tier_path_is_root_note_file(p),
            TierMatch::Any => true,
        }
    }
}

/// The public default: only layout-independent rules (agent skill folders, dependency and build
/// trees, tool mirror copies). Vault-specific rules belong in the PRIVATE rules file
/// ([`private_rules_path`]), which replaces this list entirely when it has a `tiers` array.
fn default_tier_rules() -> Vec<TierRule> {
    let r = |tier, kind, value: &str| TierRule { tier, kind, value: value.to_string() };
    use TierMatch::*;
    vec![
        r("SKILLS", Prefix, "skills/"),
        r("SKILLS", Prefix, ".claude/skills/"),
        r("SKILLS", Prefix, ".agents/skills/"),
        r("VENDOR", Contains, "/node_modules/"),
        r("VENDOR", Contains, "/.venv"),
        r("VENDOR", Contains, "/target/"),
        r("VENDOR", Contains, ".pytest_cache"),
        r("VENDOR", Contains, "disabled-loader-roots"),
        r("VENDOR", Contains, "worktrees/"),
        // Everything else is the user's own writing. Without this the default left every note
        // NULL, which serves the same rows but moved 4 of 300 SciFact fts5 top-10 lists against
        // the old build (which tagged them BRAIN); with it the public benchmark is unchanged.
        r("BRAIN", Any, ""),
    ]
}

/// `~/.recall/private-rules.json`: per-user rules that describe one person's vault and must not
/// ship in the repository. Optional; a missing file means the public defaults.
pub fn private_rules_path() -> PathBuf {
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
    PathBuf::from(home).join(".recall").join("private-rules.json")
}

/// Parse the `tiers` array of a private rules file: `[{"tier":"VENDOR","match":"prefix","value":"x/"}, ...]`.
/// `Ok(None)` when the file has no `tiers` key. Any unknown tier or match kind is an error, never
/// skipped: a silently dropped rule re-tags documents.
pub fn parse_tier_rules(text: &str) -> Result<Option<Vec<TierRule>>> {
    let v: serde_json::Value = serde_json::from_str(text).context("private rules: not valid JSON")?;
    let Some(arr) = v.get("tiers") else { return Ok(None) };
    let arr = arr.as_array().context("private rules: `tiers` must be an array")?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, r) in arr.iter().enumerate() {
        let field = |k: &str| r.get(k).and_then(|x| x.as_str()).unwrap_or("");
        let tier = KNOWN_TIERS
            .iter()
            .copied()
            .find(|t| *t == field("tier"))
            .with_context(|| format!("private rules: tiers[{i}].tier must be one of {}", KNOWN_TIERS.join(", ")))?;
        let kind = match field("match") {
            "prefix" => TierMatch::Prefix,
            "contains" => TierMatch::Contains,
            "subdir_after" => TierMatch::SubdirAfter,
            "raw_subdir_after" => TierMatch::RawSubdirAfter,
            "numbered_folder" => TierMatch::NumberedFolder,
            "numbered_root_file" => TierMatch::NumberedRootFile,
            "root_note_file" => TierMatch::RootNoteFile,
            "any" => TierMatch::Any,
            other => anyhow::bail!("private rules: tiers[{i}].match '{other}' is not a known match kind"),
        };
        let value = field("value").to_lowercase();
        if matches!(kind, TierMatch::Prefix | TierMatch::Contains | TierMatch::SubdirAfter | TierMatch::RawSubdirAfter)
            && value.is_empty()
        {
            anyhow::bail!("private rules: tiers[{i}] needs a non-empty value");
        }
        out.push(TierRule { tier, kind, value });
    }
    Ok(Some(out))
}

/// The tier rules in force: the private file's list if it has one, else the public default.
/// Loaded once. [`check_private_rules`] is the preflight that turns a bad file into an error
/// before any command runs, so this never sees one.
static TIER_RULES: std::sync::OnceLock<Vec<TierRule>> = std::sync::OnceLock::new();

fn tier_rules() -> &'static [TierRule] {
    TIER_RULES.get_or_init(|| {
        if cfg!(test) {
            return default_tier_rules(); // tests never read this machine's private file
        }
        std::fs::read_to_string(private_rules_path())
            .ok()
            .and_then(|t| parse_tier_rules(&t).ok().flatten())
            .unwrap_or_else(default_tier_rules)
    })
}

/// Vault-relative locations that exist only in one person's setup. Both optional; absent means the
/// feature that needs them is off (or asks for the key), never a guessed default path.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PrivateSettings {
    /// recall's own development folder inside the vault (eval scripts, benchmark artifacts). When
    /// set: `<it>/eval/results/` is kept out of query results, and the embedding benchmark gate,
    /// `recall bench` output and the eval-script doctor checks use it.
    pub recall_dev_dir: Option<String>,
    /// The `recall bench` fixtures file.
    pub bench_fixtures: Option<String>,
    /// `recall eval` queries for this corpus; empty = the generic default list.
    pub eval_queries: Vec<String>,
}

/// Parse the `recall_dev_dir` / `bench_fixtures` keys: vault-relative, `/`-separated, no `..`.
pub fn parse_private_settings(text: &str) -> Result<PrivateSettings> {
    let v: serde_json::Value = serde_json::from_str(text).context("private rules: not valid JSON")?;
    let get = |k: &str| -> Result<Option<String>> {
        match v.get(k) {
            None => Ok(None),
            Some(serde_json::Value::String(x)) => {
                // canonical form, so the query-exclusion prefix below matches what the path resolves to
                let x = x
                    .replace('\\', "/")
                    .split('/')
                    .filter(|seg| !seg.is_empty() && *seg != ".")
                    .collect::<Vec<_>>()
                    .join("/");
                if x.is_empty() || x.contains(':') || x.split('/').any(|seg| seg == "..") {
                    anyhow::bail!("private rules: `{k}` must be a vault-relative path without ..");
                }
                Ok(Some(x))
            }
            Some(_) => anyhow::bail!("private rules: `{k}` must be a string"),
        }
    };
    let eval_queries = match v.get("eval_queries") {
        None => Vec::new(),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .map(|q| q.as_str().filter(|q| !q.trim().is_empty()).map(str::to_string))
            .collect::<Option<Vec<_>>>()
            .context("private rules: `eval_queries` must be a list of non-empty strings")?,
        Some(_) => anyhow::bail!("private rules: `eval_queries` must be a list of non-empty strings"),
    };
    Ok(PrivateSettings {
        recall_dev_dir: get("recall_dev_dir")?,
        bench_fixtures: get("bench_fixtures")?,
        eval_queries,
    })
}

static PRIVATE_SETTINGS: std::sync::OnceLock<PrivateSettings> = std::sync::OnceLock::new();

fn private_settings() -> &'static PrivateSettings {
    PRIVATE_SETTINGS.get_or_init(|| {
        if cfg!(test) {
            return PrivateSettings::default(); // tests never read this machine's private file
        }
        std::fs::read_to_string(private_rules_path())
            .ok()
            .and_then(|t| parse_private_settings(&t).ok())
            .unwrap_or_default()
    })
}

/// `<vault>/<recall_dev_dir>` when the private rules file names one.
pub fn recall_dev_dir() -> Option<PathBuf> {
    private_settings().recall_dev_dir.as_ref().map(|r| vault_root().join(r))
}

/// `recall eval` queries from the private rules file (empty when it has none).
pub fn private_eval_queries() -> &'static [String] {
    &private_settings().eval_queries
}

/// `<vault>/<bench_fixtures>` when the private rules file names one.
pub fn bench_fixtures_path() -> Option<PathBuf> {
    private_settings().bench_fixtures.as_ref().map(|r| vault_root().join(r))
}

/// Preflight or refuse: an unreadable or invalid private rules file stops recall with the reason.
/// The validated list is frozen here, so a file edited or deleted later (a long-running daemon)
/// cannot swap in different rules between this check and the first insert.
pub fn check_private_rules() -> Result<()> {
    let path = private_rules_path();
    let (rules, paths) = match std::fs::read_to_string(&path) {
        Ok(text) => (
            parse_tier_rules(&text).with_context(|| path.display().to_string())?,
            parse_private_settings(&text).with_context(|| path.display().to_string())?,
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (None, PrivateSettings::default()),
        Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
    };
    let _ = TIER_RULES.set(rules.unwrap_or_else(default_tier_rules));
    let _ = PRIVATE_SETTINGS.set(paths);
    Ok(())
}

/// `^prefix[^/]+/` depth guard: true when `p` starts with `prefix` and there is at least one
/// more non-empty path segment before the next `/` — a genuine subdirectory, not a file
/// sitting directly in the `prefix` folder.
fn tier_path_has_subdir_after(p: &str, prefix: &str) -> bool {
    match p.strip_prefix(prefix) {
        Some(rest) => matches!(rest.find('/'), Some(idx) if idx > 0),
        None => false,
    }
}

/// `^prefix[^/]+/raw/` — a subdirectory under `prefix` whose next segment is literally `raw`.
fn tier_path_has_raw_subdir_after(p: &str, prefix: &str) -> bool {
    match p.strip_prefix(prefix) {
        Some(rest) => match rest.find('/') {
            Some(idx) if idx > 0 => rest[idx + 1..].starts_with("raw/"),
            _ => false,
        },
        None => false,
    }
}

/// `^\d\d-[^/]*/` — a numbered vault folder (`10-area/…`, `20-work/…`): the literal `-`
/// at position 2, and at least one more path segment after it.
fn tier_path_is_numbered_vault_folder(p: &str) -> bool {
    let first = p.split('/').next().unwrap_or("");
    let b = first.as_bytes();
    b.len() >= 3 && b[0].is_ascii_digit() && b[1].is_ascii_digit() && b[2] == b'-' && p.len() > first.len()
}

/// `^\d\d-[^/]*\.(md|qmd)$` — a numbered vault-style file sitting at the root (no directory).
fn tier_path_is_numbered_root_file(p: &str) -> bool {
    if p.contains('/') {
        return false;
    }
    let b = p.as_bytes();
    b.len() >= 3
        && b[0].is_ascii_digit()
        && b[1].is_ascii_digit()
        && b[2] == b'-'
        && (p.ends_with(".md") || p.ends_with(".qmd"))
}

/// `^[^/]+\.(md|qmd)$` — any loose note at the vault root (no directory).
fn tier_path_is_root_note_file(p: &str) -> bool {
    !p.contains('/') && (p.ends_with(".md") || p.ends_with(".qmd"))
}

/// Tiers served by default when `RECALL_TIERS` is unset: the user's own notes plus the
/// third-party text they collected, but NOT the agent skill library.
const DEFAULT_TIERS: &[&str] = &["BRAIN", "VENDOR"];

/// Which slices of the corpus the QUERY path is allowed to return.
///
/// `documents.tier` splits the corpus three ways: BRAIN (notes the user wrote), SKILLS (the
/// agent skill library, mirrored in from every tool that clones it) and VENDOR (textbooks,
/// cloned repos, other people's text kept on purpose).
///
/// Skills used to compete with the user's own notes and WIN: a generic "backup and recovery"
/// query returned a backup skill at rank 1, ahead of every note written on the subject.
/// Thousands of densely-keyworded how-to files are enough mass to crowd the top-k of any
/// generic phrasing. Skills now live in their own
/// index (`~/.recall/skills.db`, queried by `skill_find.py`), so the default excludes them
/// here rather than letting two different kinds of document fight over the same ranking.
///
/// **A NULL tier is always allowed.** A freshly ingested document has no tier until the
/// tagger runs, and a gate that hid untagged rows would make new notes silently
/// unsearchable — a much worse failure than a skill ranking too high. Fail open.
pub struct TierFilter {
    /// `None` = serve everything (no clause emitted at all).
    allowed: Option<Vec<&'static str>>,
}

impl TierFilter {
    /// Precedence: `RECALL_TIERS` → `~/.recall/tiers.txt` → [`DEFAULT_TIERS`].
    ///
    /// The file exists for the same reason `exclude.txt` does: retrieval policy belongs in a
    /// readable, diffable file rather than only inside whichever script last launched the
    /// daemon. This one also degrades safely — with no file and no variable the compiled
    /// brain+vendor default applies, which is the intended policy anyway.
    ///
    /// Unrecognised names are ignored rather than fatal, and if that leaves nothing at all
    /// the default is restored: a typo must not silently return zero results forever.
    pub fn load() -> Self {
        if let Ok(spec) = std::env::var("RECALL_TIERS") {
            return Self::from_spec(&spec);
        }
        if let Ok(text) = std::fs::read_to_string(tiers_file_path()) {
            let spec: String = text
                .lines()
                .map(|l| l.split('#').next().unwrap_or("").trim())
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(",");
            if !spec.is_empty() {
                return Self::from_spec(&spec);
            }
        }
        Self::from_list(DEFAULT_TIERS)
    }

    pub fn from_spec(spec: &str) -> Self {
        let spec = spec.trim();
        if spec.is_empty() || spec.eq_ignore_ascii_case("all") {
            return TierFilter { allowed: None };
        }
        let mut allowed: Vec<&'static str> = Vec::new();
        for part in spec.split(',') {
            let part = part.trim();
            if let Some(known) = KNOWN_TIERS
                .iter()
                .find(|k| k.eq_ignore_ascii_case(part))
                .copied()
            {
                if !allowed.contains(&known) {
                    allowed.push(known);
                }
            }
        }
        if allowed.is_empty() {
            return Self::from_list(DEFAULT_TIERS);
        }
        TierFilter {
            allowed: Some(allowed),
        }
    }

    fn from_list(list: &[&'static str]) -> Self {
        TierFilter {
            allowed: Some(list.to_vec()),
        }
    }

    /// SQL fragment appended to an existing WHERE clause, e.g.
    /// `" AND (d.tier IS NULL OR d.tier IN ('BRAIN','VENDOR'))"`, or `""` when unfiltered.
    /// Only [`KNOWN_TIERS`] constants are ever interpolated.
    pub fn sql_clause(&self, alias: &str) -> String {
        match &self.allowed {
            None => String::new(),
            Some(list) => {
                let quoted: Vec<String> = list.iter().map(|t| format!("'{t}'")).collect();
                format!(
                    " AND ({alias}.tier IS NULL OR {alias}.tier IN ({}))",
                    quoted.join(",")
                )
            }
        }
    }

    /// Human-readable form for status output and warnings.
    pub fn describe(&self) -> String {
        match &self.allowed {
            None => "all".to_string(),
            Some(list) => list.join(","),
        }
    }
}

// ── Tombstone-on-index (`--prune-missing`) ───────────────────────────────────

/// Refuse to tombstone when this enumeration came in more than this fraction BELOW the
/// recent baseline.
///
/// `enumerate_docs` shells out to `rg --files` behind a 5-second wall-clock deadline
/// (`ENUM_TIMEOUT`) and silently falls back to a narrower recursive walk if rg is missing,
/// errors, or loses that race — an AV intercept, a loaded machine, a stalled network path.
/// Either way the enumerated set comes back SHORT, and a short set is row-for-row
/// indistinguishable from "the user deleted those files". Tombstoning off one would mark
/// thousands of live documents dead in a single pass.
///
/// 10% is chosen because the two magnitudes are nowhere near each other: ordinary vault churn
/// is a few hundred notes a week against ~70k live docs (well under 1%), while an enumeration
/// failure loses whole directory subtrees — the 2026-08-18 measurement had 93,965 files
/// deleted at once. Anything between those is worth a human look, which is what refusing buys.
const PRUNE_MAX_ENUM_SHRINK: f64 = 0.10;

/// The baseline is the MAX enumerated count over this many recent successful runs — not the
/// single previous one.
///
/// If "previous" meant "last", one truncated enumeration recorded as the baseline would
/// authorise the next equally-truncated run, and the gate would quietly ratchet itself open
/// one bad run at a time. Taking the max over a short window means a low reading can never
/// lower the bar, while a genuine, permanent shrink still becomes the new normal after a few
/// runs instead of requiring the override forever.
const PRUNE_BASELINE_RUNS: usize = 5;

/// Operator override for [`prune_gate_verdict`], mirroring `RECALL_CONFIRM_RESET`.
const PRUNE_SHRINK_OVERRIDE_ENV: &str = "RECALL_CONFIRM_PRUNE_SHRINK";

/// What `--prune-missing` should do once it has decided which documents are gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneMode {
    /// Analyse and write the list. Touches no row.
    Dry,
    /// Analyse, write the list, and set `valid_to` on every confirmed-absent document.
    Apply,
}

/// The outcome of one [`Store::prune_missing`] run.
#[derive(Debug)]
pub struct PruneReport {
    pub enumerated: usize,
    pub used_fallback: bool,
    pub baseline: Option<i64>,
    pub live_vault_docs: usize,
    /// Live vault docs absent from the enumerated set. **Candidates, not victims** — this
    /// number includes every excluded, gitignored and oversized file that is still on disk.
    pub not_enumerated: usize,
    /// The subset the filesystem confirms is gone. The only rows ever tombstoned.
    pub missing: Vec<String>,
    /// Rows actually updated (0 in [`PruneMode::Dry`]).
    pub tombstoned: usize,
    pub list_path: PathBuf,
}

/// The prune sanity gate, as a pure decision so every branch is testable without a filesystem.
///
/// `Ok(())` means the enumeration is trustworthy enough to tombstone against. `Err(reason)`
/// means it is not, and the caller must change nothing and exit non-zero.
pub(crate) fn prune_gate_verdict(
    enumerated: usize,
    baseline: Option<i64>,
    used_fallback: bool,
) -> std::result::Result<(), String> {
    // The override exists because a real, permanent shrink would otherwise deadlock: the gate
    // refuses, so no run is recorded, so the baseline never comes down. It is deliberately an
    // env var and not a flag — same shape as RECALL_CONFIRM_RESET, and awkward enough to type
    // that it cannot be reached by muscle memory.
    if std::env::var(PRUNE_SHRINK_OVERRIDE_ENV).as_deref() == Ok("yes") {
        return Ok(());
    }

    let Some(baseline) = baseline else {
        // Nothing to compare against. Allowed, because the per-file filesystem check in
        // `prune_missing` is an independent authority and refusing forever on a fresh store
        // would make the feature unreachable. The caller prints this state prominently.
        return Ok(());
    };

    let floor = (baseline as f64 * (1.0 - PRUNE_MAX_ENUM_SHRINK)).floor() as i64;
    if (enumerated as i64) < floor {
        return Err(format!(
            "enumerated {enumerated} docs, more than {:.0}% below the baseline of {baseline} \
             (floor {floor}) — a truncated enumeration looks exactly like a mass deletion",
            PRUNE_MAX_ENUM_SHRINK * 100.0
        ));
    }

    if used_fallback && (enumerated as i64) < baseline {
        return Err(format!(
            "the rg enumeration failed and the fallback walk returned {enumerated} docs, \
             below the baseline of {baseline} — the walk skips node_modules/target/dotdirs \
             and does not read .gitignore, so its shortfall says nothing about the disk"
        ));
    }

    Ok(())
}

/// `true` only when the filesystem positively reports that `rel` does not exist under `root`.
///
/// Every other outcome — permission denied, an IO error, a path the OS will not parse —
/// returns `false`. Tombstoning is a mutation, and "I could not tell" must never authorise
/// one. `symlink_metadata` rather than `metadata` for the same reason: a symlink whose target
/// has gone is still an entry the user created, and the conservative answer is "present".
fn path_is_absent(root: &Path, rel: &str) -> bool {
    let p = Path::new(rel);
    // An absolute path, a drive prefix, or a `..` escape cannot be resolved against the root
    // safely, and a wrong resolution here deletes a live document from search. Treat as present.
    if p.is_absolute()
        || p.components()
            .any(|c| matches!(c, Component::ParentDir | Component::Prefix(_)))
    {
        return false;
    }
    // 🔴 Resolve with the SAME function that created the row. `resolve_ingest_abs_path` sends a
    // `.claude/...` path-list entry to USERPROFILE, NOT to the vault root; a bare `root.join()`
    // here therefore looked for it in the wrong place, found nothing, and answered "absent" for
    // every such document forever.
    //
    // Two things read that answer and both are destructive:
    //   * move detection treats a permanently-absent document as a permanently-open MOVE
    //     candidate, so the first vault file with a byte-identical body takes its id and all of
    //     its vectors, and the original is left at no path at all - no tombstone, nothing to
    //     find it by;
    //   * `prune_missing` tombstones it.
    //
    // Exposure when this was found was zero rows, because the live corpus happened to hold no
    // `.claude/` documents at the time. `recall index --paths` with a `.claude/` entry is a
    // supported, tested route to creating one.
    let Ok(abs) = resolve_ingest_abs_path(root, rel) else {
        // Cannot resolve it (no USERPROFILE/HOME) - the safe answer is "present", same as the
        // escape cases above. Never let a resolution failure delete a document.
        return false;
    };
    match std::fs::symlink_metadata(abs) {
        Ok(_) => false,
        Err(e) => e.kind() == std::io::ErrorKind::NotFound,
    }
}

/// Write the would-tombstone / did-tombstone list, and return where it landed.
///
/// Written in BOTH modes on purpose. In `Dry` it is the thing a human reads before approving;
/// in `Apply` it is the record of what was done and the input to undoing it.
#[allow(clippy::too_many_arguments)]
fn write_prune_list(
    list_dir: &Path,
    mode: PruneMode,
    missing: &[(i64, String)],
    enumerated: usize,
    baseline: Option<i64>,
    used_fallback: bool,
    not_enumerated: usize,
    live_vault_docs: usize,
) -> Result<PathBuf> {
    std::fs::create_dir_all(list_dir)
        .with_context(|| format!("create prune list dir {}", list_dir.display()))?;
    let verb = match mode {
        PruneMode::Dry => "would-tombstone",
        PruneMode::Apply => "tombstoned",
    };
    let path = list_dir.join(format!("prune-missing-{}-{}.txt", verb, Store::now()));
    let mut f = std::fs::File::create(&path)
        .with_context(|| format!("write prune list {}", path.display()))?;

    writeln!(f, "# recall --prune-missing={}", match mode {
        PruneMode::Dry => "dry",
        PruneMode::Apply => "apply",
    })?;
    writeln!(f, "# enumerated:      {enumerated}{}", if used_fallback { "  (FALLBACK walk, not rg)" } else { "" })?;
    writeln!(f, "# baseline:        {}", baseline.map(|b| b.to_string()).unwrap_or_else(|| "none recorded".into()))?;
    writeln!(f, "# live vault docs: {live_vault_docs}")?;
    writeln!(f, "# not enumerated:  {not_enumerated}  (candidates — includes excluded/gitignored/oversized files still on disk)")?;
    writeln!(f, "# FILE IS GONE:    {}  (filesystem-confirmed absent — the rows below)", missing.len())?;
    writeln!(f, "#")?;
    writeln!(f, "# Format: <doc_id>\\t<vault-relative path>")?;
    writeln!(f, "# Reversible: UPDATE documents SET valid_to = NULL WHERE id IN (<the ids below>);")?;
    for (id, p) in missing {
        writeln!(f, "{id}\t{p}")?;
    }
    f.flush()?;
    Ok(path)
}

/// A compiled exclude list loaded once per `ingest_dir` call.
/// Zero-allocation hot path: `is_excluded` only borrows string slices.
/// Size of the [`Store::searchable_snapshot`] MinHash sketch.
pub const SKETCH_K: usize = 256;

/// Estimated Jaccard similarity of the two corpus states two bottom-k sketches (ascending) came
/// from: of the k smallest hashes of their union, the share present in both. Empty vs empty is 1.
pub fn sketch_similarity(a: &[u64], b: &[u64]) -> f64 {
    let k = SKETCH_K.min(a.len().max(b.len()));
    if k == 0 {
        return 1.0;
    }
    let (sa, sb): (std::collections::BTreeSet<u64>, std::collections::BTreeSet<u64>) =
        (a.iter().copied().collect(), b.iter().copied().collect());
    let union: Vec<u64> = sa.union(&sb).copied().take(k).collect();
    union.iter().filter(|x| sa.contains(x) && sb.contains(x)).count() as f64 / union.len() as f64
}

/// Abstain calibrations kept per store ([`Store::calibration_put`]).
pub const CALIBRATION_KEEP: usize = 4;

pub struct PathExcludeList {
    patterns: Vec<String>,
}

impl PathExcludeList {
    /// Build from builtin patterns + `~/.recall/exclude.txt` + `RECALL_EXCLUDE_EXTRA`.
    ///
    /// The file exists so the policy is VISIBLE and version-controllable. Until now the
    /// real list lived only inside `binecall-daemon-silent.vbs` as a comma-joined string
    /// in an environment variable — invisible to anyone reading this repository, and
    /// impossible to diff. Anything that starts the daemon by another route (a scheduled
    /// task, a shim, a bare `recall daemon up`) silently got a different search policy.
    ///
    /// Both sources work and are additive; the environment variable is still the right tool
    /// for a one-off override.
    ///
    /// (Note for future debugging: when excluded paths reappear in results, the cause is
    /// far more likely to be a STALE DAEMON BINARY than a missing variable — see the
    /// 2026-08-05 incident in CLAUDE.md. Verify what the running process is, not what the
    /// source says.)
    pub fn load() -> Self {
        let mut patterns: Vec<String> = BUILTIN_EXCLUDE_PREFIXES
            .iter()
            .map(|s| s.to_string())
            .collect();

        for s in Self::patterns_from_file(&exclude_file_path()) {
            patterns.push(s);
        }

        if let Ok(extra) = std::env::var("RECALL_EXCLUDE_EXTRA") {
            for seg in extra.split(',') {
                let s = seg.trim().to_lowercase();
                if !s.is_empty() {
                    patterns.push(s);
                }
            }
        }

        patterns.sort();
        patterns.dedup();
        PathExcludeList { patterns }
    }

    /// One pattern per line; `#` starts a comment; blank lines ignored. A missing or
    /// unreadable file is not an error — it just contributes nothing.
    fn patterns_from_file(path: &std::path::Path) -> Vec<String> {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Vec::new();
        };
        text.lines()
            .map(|l| l.split('#').next().unwrap_or("").trim().to_lowercase())
            .filter(|l| !l.is_empty())
            .collect()
    }

    /// Build from an explicit pattern slice — used in unit tests to avoid
    /// mutating global process environment (which is not thread-safe in tests).
    #[cfg(test)]
    pub(crate) fn with_extra(extra: &[&str]) -> Self {
        let mut patterns: Vec<String> = BUILTIN_EXCLUDE_PREFIXES
            .iter()
            .map(|s| s.to_string())
            .collect();
        for p in extra {
            let s = p.trim().to_lowercase();
            if !s.is_empty() {
                patterns.push(s);
            }
        }
        PathExcludeList { patterns }
    }

    /// Returns `true` when `rel_path` (vault-relative, forward-slash normalised)
    /// contains a segment that matches any loaded pattern.
    ///
    /// Matching: a segment matches a pattern when the segment (lowercased) either
    /// equals the pattern exactly or starts with the pattern.
    pub fn is_excluded(&self, rel_path: &str) -> bool {
        self.matching_pattern(rel_path).is_some()
    }

    /// The first loaded pattern that excludes `rel_path` (the `recall ui` scan preview groups by it).
    pub fn matching_pattern(&self, rel_path: &str) -> Option<&str> {
        self.patterns.iter().map(String::as_str).find(|p| segment_match(p, rel_path))
    }
}

/// The exclude rule for ONE pattern: some path segment, lower-cased, starts with it (so `foo`
/// also matches `foo-bar` and `foo.md`).
pub fn segment_match(pat: &str, rel_path: &str) -> bool {
    rel_path.split('/').any(|seg| seg.to_lowercase().starts_with(pat))
}

pub(crate) fn is_query_excluded_path(exclude: &PathExcludeList, rel_path: &str) -> bool {
    if exclude.is_excluded(rel_path) {
        return true;
    }
    let p = rel_path.replace('\\', "/").to_lowercase();
    p.starts_with("eval/results/")
        || private_settings()
            .recall_dev_dir
            .as_ref()
            .is_some_and(|d| p.starts_with(&format!("{}/eval/results/", d.to_lowercase())))
}

/// The outcome of one [`enumerate_docs`] call.
///
/// `used_fallback` is not diagnostic decoration — it changes what the enumerated set MEANS.
/// `rg --files` respects `.gitignore` and skips hidden directories; the fallback walk skips a
/// different set (`node_modules`, `target`, `vendor`, anything dot-prefixed, anything past
/// `MAX_DEPTH`) and does not read `.gitignore` at all. The two therefore return genuinely
/// different sets over the same tree, so a count that shrank across that boundary says nothing
/// about the disk. `prune_missing` treats it as disqualifying for exactly that reason.
pub(crate) struct Enumeration {
    pub files: Vec<PathBuf>,
    pub used_fallback: bool,
    /// Wall time of the whole enumeration (rg, or rg's failure plus the fallback walk).
    pub elapsed_ms: u64,
}

/// What a missing ripgrep costs, in one sentence, with the fix.
pub(crate) const RG_MISSING: &str = "ripgrep (rg) not found on PATH: the rg query channel is OFF and \
`recall index --prune-missing` is disabled. Install it: winget install BurntSushi.ripgrep.MSVC (Windows) \
| brew install ripgrep (macOS) | sudo apt-get install ripgrep (Debian/Ubuntu)";

/// Whether `bin --version` runs and exits 0 within 3 s (a wedged binary is killed, not waited on).
pub(crate) fn rg_available_at(bin: &str) -> bool {
    let Ok(mut child) = std::process::Command::new(bin)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        return false;
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
        }
    }
}

pub(crate) fn rg_available() -> bool {
    rg_available_at("rg")
}

/// What one `ingest_dir` pass did to the files it saw. `restored` is a file that was deleted and
/// came back; `moved` is a rename.
#[derive(Debug, Default, Clone, Copy)]
pub struct IngestCounts {
    pub new: usize,
    pub updated: usize,
    pub moved: usize,
    pub restored: usize,
    pub unchanged: usize,
}

/// Print [`RG_MISSING`] to stderr at most once per process, so a missing rg is never silent.
pub(crate) fn warn_rg_missing() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| eprintln!("{} {RG_MISSING}", crate::reply::paint_err(crate::reply::WARN, "recall: warning:")));
}

/// rg flags that stop PARENT and GLOBAL ignore rules from hiding notes. A vault under a repo whose
/// .gitignore is `*` (a home-folder repo), or under a `~/.rgignore`, otherwise indexes partly with
/// exit 0. The vault's OWN .gitignore/.ignore still apply to folders (the `*.md` globs whitelist
/// matching FILES, so a rule naming one note does not hide it).
const RG_OWN_IGNORES_ONLY: &[&str] = &["--no-ignore-parent", "--no-ignore-global"];

/// `rg --files` for .md/.qmd under `root` behind a wall-clock deadline, so a hung rg (NFS stall, AV
/// intercept, wedged child) can't freeze indexing. 120 s, not 5: a real 30k-note vault took ~5 s, and
/// a timeout makes prune REFUSE (fallback enumerations are not trusted). `None` = missing / errored /
/// timed out; the caller falls back to the walk.
fn rg_files(root: &Path, extra: &[&str], loud: bool) -> Option<Vec<u8>> {
    const ENUM_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
    let mut child = match std::process::Command::new("rg")
        .args(["--no-config", "--files", "-g", "*.md", "-g", "*.qmd"])
        .args(extra)
        .arg(root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => {
            if loud {
                warn_rg_missing();
            }
            return None;
        }
    };

    // Drain stdout on a separate thread. `rg --files` over the 100k+ file vault
    // emits several MB — far past the ~64 KB OS pipe buffer. Without a concurrent
    // reader, rg blocks writing to a full pipe, never exits, and always trips
    // ENUM_TIMEOUT below, silently forcing the slow recursive walk every index.
    // With a reader draining live, rg streams to completion (or is killed on a
    // real timeout, which closes the pipe and ends the read). Mirrors rg_channel.
    let mut out_pipe = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = std::io::Read::read_to_end(&mut out_pipe, &mut buf);
        buf
    });

    let deadline = std::time::Instant::now() + ENUM_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let buf = reader.join().unwrap_or_default();
                // ponytail: exit 1 + no output is ambiguous (an empty vault, or rg's ignore rules
                // hid every file, e.g. a vault under a dotdir) and stays a fallback. Cost: prune
                // refuses on a vault with zero markdown files left.
                if !status.success() && loud {
                    eprintln!("recall: rg enumeration failed ({status}); using the slower fallback walk, and prune will refuse");
                }
                return status.success().then_some(buf);
            }
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            _ => {
                // Timed out or wait error — kill, join the reader so its
                // thread/handle can't leak, and fall back to the walk.
                if loud {
                    eprintln!("recall: rg enumeration timed out; using the slower fallback walk, and prune will refuse");
                }
                let _ = child.kill();
                let _ = reader.join();
                return None;
            }
        }
    }
}

/// Enumerate candidate docs (.md/.qmd) under `root` via ripgrep.
///
/// The vault contains ~100k+ markdown files spread across cloned repos and
/// tool caches; a hand-rolled std::fs recursion is too slow and pins memory.
/// `rg --files` walks in parallel, respects .gitignore, and skips hidden dirs
/// by default — enumerating the whole tree in seconds. If rg is missing or
/// fails, fall back to the recursive walk so indexing still works offline.
///
/// 🔴 **The returned set is ALREADY FILTERED by `exclude`.** It is the set of files this
/// indexer would ingest — not the set of files that exist. Anything that diffs live
/// documents against this list is measuring "would we ingest it today", and an excluded
/// document is a real file on disk that simply is not wanted in results. `prune_missing`
/// re-checks the filesystem for precisely this reason; see the comment there.
fn enumerate_docs(root: &Path, exclude: &PathExcludeList) -> Result<Enumeration> {
    // Spawn rg with a wall-clock deadline so a hung rg (NFS stall, AV intercept,
    // wedged child) can't freeze indexing. Mirrors rg_channel in query.rs.
    //
    // 120 s, not 5: a real 30k-note vault took ~5 s for `rg --files` on *.md alone, and a
    // timeout now makes prune REFUSE (fallback enumerations are not trusted),
    // so a deadline at the edge would stop the nightly prune. It only has to catch a hung rg.
    let started = std::time::Instant::now();
    let elapsed_ms = || started.elapsed().as_millis() as u64;

    // Returns Some(stdout bytes) on a clean rg run, None if rg is missing /
    // errored / exited non-zero / timed out — caller then uses the walk fallback.
    let rg_stdout = rg_files(root, RG_OWN_IGNORES_ONLY, true);
    match rg_stdout {
        Some(buf) => {
            let text = String::from_utf8_lossy(&buf);
            let root_norm = root.to_string_lossy().replace('\\', "/");
            let raw_count = text.lines().filter(|l| !l.trim().is_empty()).count();
            let files = text
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .filter(|l| {
                    // Compute vault-relative path for the exclude check.
                    let abs_norm = l.replace('\\', "/");
                    let rel = abs_norm
                        .strip_prefix(root_norm.as_str())
                        .map(|s| s.trim_start_matches('/'))
                        .unwrap_or(abs_norm.as_str());
                    !exclude.is_excluded(rel)
                })
                .map(PathBuf::from)
                .collect::<Vec<_>>();
            // Notes the vault's own ignore rules hide are not indexed; say how many, never silently.
            if let Some(all) = rg_files(root, &["--no-ignore"], false) {
                let hidden = all.split(|&b| b == b'\n').filter(|l| !l.is_empty()).count().saturating_sub(raw_count);
                if hidden > 0 {
                    eprintln!("{} {hidden} note(s) are hidden by ignore rules inside the vault (folders its own .gitignore/.ignore/.rgignore exclude) and are NOT indexed (a .gitignore/.ignore/.rgignore under {})", crate::reply::paint_err(crate::reply::WARN, "recall: warning:"), root.display());
                }
            }
            Ok(Enumeration {
                files,
                used_fallback: false,
                elapsed_ms: elapsed_ms(),
            })
        }
        None => {
            // rg unavailable / errored — fall back to the slow but reliable walk.
            let mut files = Vec::new();
            collect_docs(root, &mut files, exclude)?;
            Ok(Enumeration {
                files,
                used_fallback: true,
                elapsed_ms: elapsed_ms(),
            })
        }
    }
}

fn collect_docs(dir: &Path, out: &mut Vec<PathBuf>, exclude: &PathExcludeList) -> Result<()> {
    // Iterative walk with an explicit work stack + depth cap. The recursive form
    // could overflow the call stack on a pathologically deep tree and (because
    // `is_dir()` follows symlinks) loop forever on a symlink cycle when this
    // rg-absent fallback runs. The cap bounds both without changing which files
    // match — no real vault approaches this depth.
    const MAX_DEPTH: usize = 64;
    let mut stack: Vec<(PathBuf, usize)> = vec![(dir.to_path_buf(), 0)];
    while let Some((cur, depth)) = stack.pop() {
        let rd = match std::fs::read_dir(&cur) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for entry in rd.flatten() {
            let p = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if p.is_dir() {
                if depth + 1 > MAX_DEPTH {
                    continue;
                }
                // skip noisy / non-knowledge dirs — the vault root contains cloned
                // repos and tool caches whose file counts dwarf the actual notes.
                if name.starts_with('.')
                    || matches!(
                        name.as_ref(),
                        "node_modules"
                            | "target"
                            | "deps"
                            | "dist"
                            | "build"
                            | "out"
                            | "vendor"
                            | "venv"
                            | "__pycache__"
                            | ".venv"
                            | ".git"
                            | ".obsidian"
                            | ".fingerprint"
                            | ".next"
                            | ".cache"
                            | ".cargo"
                            | ".rustup"
                            | "site-packages"
                    )
                {
                    continue;
                }
                // Also skip snapshot mirror dirs before descending into them.
                if exclude.is_excluded(&name.to_lowercase()) {
                    continue;
                }
                stack.push((p, depth + 1));
            } else if let Some(ext) = p.extension() {
                if ext == "md" || ext == "qmd" {
                    out.push(p);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        /// Full scans [`Store::scan_searchable`] ran on this thread.
        pub(super) static SNAPSHOT_SCANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        static ENUMERATION_ORDER: std::cell::RefCell<Option<(bool, Vec<PathBuf>)>> = const { std::cell::RefCell::new(None) };
    }

    /// Override the actual list consumed by ingest_dir on this test thread only.
    pub(super) fn arrange_enumeration(mut files: Vec<PathBuf>) -> Vec<PathBuf> {
        ENUMERATION_ORDER.with(|order| {
            if let Some((reverse, observed)) = order.borrow_mut().as_mut() {
                files.sort();
                if *reverse { files.reverse(); }
                *observed = files.clone();
            }
        });
        files
    }

    struct EnumerationOrder;
    impl EnumerationOrder {
        fn set(reverse: bool) -> Self {
            ENUMERATION_ORDER.with(|order| {
                assert!(order.borrow().is_none());
                *order.borrow_mut() = Some((reverse, Vec::new()));
            });
            Self
        }
        fn observed(&self) -> Vec<PathBuf> {
            ENUMERATION_ORDER.with(|order| order.borrow().as_ref().unwrap().1.clone())
        }
    }
    impl Drop for EnumerationOrder {
        fn drop(&mut self) {
            ENUMERATION_ORDER.with(|order| *order.borrow_mut() = None);
        }
    }

    /// The per-view snapshot cache: reused until a commit on ANY connection, a write on this one,
    /// or a different view (exclude list); always equal to a fresh scan.
    #[test]
    fn searchable_snapshot_is_cached_until_the_database_or_the_view_changes() {
        let _env = crate::query::tests::env_lock();
        let path = std::env::temp_dir().join(format!("recall-snapcache-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let (a, b) = (Store::open_at(&path).unwrap(), Store::open_at(&path).unwrap());
        a.insert("n1.md", "n1", "alpha note", "vault").unwrap();
        let scans = || SNAPSHOT_SCANS.with(|c| c.get());
        let fresh = || Store::open_at(&path).unwrap().searchable_snapshot().unwrap();
        let s0 = scans();
        let f1 = a.searchable_snapshot().unwrap();
        assert_eq!(a.searchable_snapshot().unwrap(), f1);
        assert_eq!(scans(), s0 + 1, "the second lookup is served from the cache");
        assert_eq!(fresh(), f1);
        b.insert("n2.md", "n2", "beta note", "vault").unwrap(); // another connection commits
        let f2 = a.searchable_snapshot().unwrap();
        assert_ne!(f2.0, f1.0, "another connection's commit invalidates");
        assert_eq!(fresh(), f2);
        a.insert("zz-skip/n3.md", "n3", "gamma note", "vault").unwrap(); // this connection writes
        let f3 = a.searchable_snapshot().unwrap();
        assert_ne!(f3.0, f2.0, "this connection's own write invalidates");
        let saved = std::env::var_os("RECALL_EXCLUDE_EXTRA");
        std::env::set_var("RECALL_EXCLUDE_EXTRA", "zz-skip");
        let f4 = a.searchable_snapshot().unwrap();
        let f4_fresh = fresh();
        match saved {
            Some(v) => std::env::set_var("RECALL_EXCLUDE_EXTRA", v),
            None => std::env::remove_var("RECALL_EXCLUDE_EXTRA"),
        }
        assert_eq!(f4.0, f2.0, "another exclude list is another view: n3 excluded again");
        assert_eq!(f4, f4_fresh);
        assert_eq!(a.searchable_snapshot().unwrap(), f3, "and the first view is still cached");
        drop((a, b));
        for ext in ["db", "db-wal", "db-shm"] {
            let _ = std::fs::remove_file(path.with_extension(ext));
        }
    }

    // ── content_sha256 backfill ───────────────────────────────────────────────

    /// The backfill must produce the SHA-256 of the body, be idempotent, and be resumable.
    #[test]
    fn backfill_hashes_every_body_exactly_once() {
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();

        let a = s.insert("a.md", "A", "body one", "test").unwrap();
        let b = s.insert("b.md", "B", "body two", "test").unwrap();
        let c = s.insert("c.md", "C", "", "test").unwrap();

        // `insert` now hashes at insert time, so a freshly written row has nothing to backfill.
        // The backfill exists for rows written BEFORE the column did - blank them to reproduce
        // that state, which is the only state it is ever asked to repair.
        s.conn
            .execute("UPDATE documents SET content_sha256 = NULL", [])
            .unwrap();

        // A limit smaller than the corpus must make progress, not stall.
        assert_eq!(s.backfill_content_hashes(2).unwrap(), 2);
        assert_eq!(s.backfill_content_hashes(2).unwrap(), 1);
        // Nothing left: a finished corpus is a no-op, which is what stops the caller's loop.
        assert_eq!(s.backfill_content_hashes(2).unwrap(), 0);

        let got = |id: i64| -> String {
            s.conn
                .query_row(
                    "SELECT content_sha256 FROM documents WHERE id = ?1",
                    params![id],
                    |r| r.get(0),
                )
                .unwrap()
        };
        assert_eq!(got(a), sha256_hex(b"body one"));
        assert_eq!(got(b), sha256_hex(b"body two"));
        // An empty body still gets a hash - the SHA-256 of the empty string. It must not be
        // left NULL, or it would be re-selected forever.
        assert_eq!(got(c), sha256_hex(b""));
    }

    /// A row that already carries a hash must never be rewritten.
    ///
    /// Two separate defences, asserted separately because they fail separately:
    ///  1. the SELECT filters on `content_sha256 IS NULL`, so such a row is never even a
    ///     candidate - the backfill reports no progress and leaves the value alone;
    ///  2. the UPDATE re-asserts `content_sha256 IS NULL`.
    ///
    /// To be precise about which defence does what, because it is easy to state this wrongly:
    /// TRANSACTION ISOLATION is what protects against a concurrent writer - a commit between
    /// the SELECT and the UPDATE fails the WAL snapshot and rolls the batch back with an
    /// error; it does not quietly skip a row and report zero. Defence 2 protects against
    /// something else: this statement being reused outside that transaction discipline, or
    /// the SELECT filter being loosened.
    ///
    /// Defence 2 is therefore unreachable through `backfill_content_hashes` in a single-
    /// threaded test - the SELECT has already excluded the row - so it is asserted against the
    /// shipped statement itself. Neither test exercises two concurrent connections.
    #[test]
    fn backfill_never_overwrites_an_existing_hash() {
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = s.insert("a.md", "A", "original", "test").unwrap();
        s.conn
            .execute(
                "UPDATE documents SET content_sha256 = 'SENTINEL' WHERE id = ?1",
                params![id],
            )
            .unwrap();

        // Defence 1: not a candidate, so no progress is reported and the value survives.
        assert_eq!(s.backfill_content_hashes(10).unwrap(), 0);
        let after: String = s
            .conn
            .query_row(
                "SELECT content_sha256 FROM documents WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(after, "SENTINEL", "the backfill must only ever fill a NULL");

        // Defence 2: THE statement the backfill ships (`BACKFILL_UPDATE_SQL`, not a copy of
        // it), run against the same row, must affect ZERO rows. Deleting the predicate from the
        // const fails this assertion, which is the whole point of sharing it.
        let affected = s
            .conn
            .execute(BACKFILL_UPDATE_SQL, params!["CLOBBERED", id])
            .unwrap();
        assert_eq!(affected, 0, "the UPDATE's own guard must reject a hashed row");
        let still: String = s
            .conn
            .query_row(
                "SELECT content_sha256 FROM documents WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(still, "SENTINEL");
    }

    /// `documents_au` must not fire for a write that touches neither title nor body - that
    /// narrowing is the whole reason the backfill is cheap - and must still fire for a body edit.
    ///
    /// A row COUNT over the shadow table is too weak for either half: equal counts do not prove
    /// equal contents, and a changed count after an edit does not prove the OLD tokens were
    /// removed. So the metadata half compares the shadow table's full contents byte for byte,
    /// and the body half asserts on what the index actually answers - the old term must stop
    /// matching and the new term must start.
    #[test]
    fn metadata_update_does_not_touch_the_fts_index() {
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = s
            .insert("a.md", "Title", "aardvark searchable body", "test")
            .unwrap();

        // Full contents of the FTS shadow table, not just its row count.
        let fts_dump = |s: &Store| -> Vec<(i64, Vec<u8>)> {
            let mut st = s
                .conn
                .prepare("SELECT id, block FROM documents_fts_data ORDER BY id")
                .unwrap();
            let v = st
                .query_map([], |r| {
                    Ok((r.get(0)?, r.get::<_, Option<Vec<u8>>>(1)?.unwrap_or_default()))
                })
                .unwrap()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap();
            v
        };
        let matches = |s: &Store, term: &str| -> i64 {
            s.conn
                .query_row(
                    "SELECT count(*) FROM documents_fts WHERE documents_fts MATCH ?1",
                    params![term],
                    |r| r.get(0),
                )
                .unwrap()
        };

        assert_eq!(matches(&s, "aardvark"), 1, "fixture must be indexed to start");

        let before = fts_dump(&s);
        s.conn
            .execute(
                "UPDATE documents SET content_sha256 = 'x' WHERE id = ?1",
                params![id],
            )
            .unwrap();
        assert_eq!(
            fts_dump(&s),
            before,
            "a metadata write must leave the FTS index byte-identical"
        );
        assert_eq!(matches(&s, "aardvark"), 1);

        // ...but a body edit must both remove the old tokens and add the new ones.
        s.conn
            .execute(
                "UPDATE documents SET body = 'pangolin different body' WHERE id = ?1",
                params![id],
            )
            .unwrap();
        assert_eq!(
            matches(&s, "aardvark"),
            0,
            "the old term must stop matching after a body edit"
        );
        assert_eq!(
            matches(&s, "pangolin"),
            1,
            "the new term must start matching after a body edit"
        );
    }

    // ── PathExcludeList unit tests ────────────────────────────────────────────

    #[test]
    fn exclude_disabled_loader_roots_exact_known_path() {
        // The confirmed real-world culprit path that produced duplicate top-5 results.
        let ex = PathExcludeList::load();
        assert!(
            ex.is_excluded(
                "tools/state/skill-radar/disabled-loader-roots-2026-05-09-active-finder-only/optimization-reference.md"
            ),
            "disabled-loader-roots-* snapshot mirror must be excluded"
        );
    }

    #[test]
    fn exclude_disabled_loader_roots_any_date_suffix() {
        // The prefix match must fire regardless of the date stamp variant.
        let ex = PathExcludeList::load();
        assert!(ex.is_excluded(
            "tools/state/skill-radar/disabled-loader-roots-2027-01-01-foo/skill.md"
        ));
    }

    #[test]
    fn exclude_disabled_prefix_in_any_segment() {
        // "disabled-*" appearing anywhere in the path (not just first segment).
        let ex = PathExcludeList::load();
        assert!(ex.is_excluded("some/deep/path/disabled-loader-roots-2026/file.md"));
    }

    #[test]
    fn allow_canonical_skill_path() {
        // The real canonical skill path must NOT be excluded.
        let ex = PathExcludeList::load();
        assert!(
            !ex.is_excluded(
                "skills/optimization-suite/ln-810-performance-optimization-coordinator/SKILL.md"
            ),
            "canonical skill path must NOT be excluded"
        );
    }

    #[test]
    fn allow_ordinary_notes() {
        let ex = PathExcludeList::load();
        assert!(!ex.is_excluded("projects/recall/plan.md"));
        assert!(!ex.is_excluded("general/Patterns.md"));
    }

    #[test]
    fn exclude_disabled_prefix_variants() {
        // Any path segment starting with "disabled-" is a vault tool-dump artifact.
        let ex = PathExcludeList::load();
        assert!(ex.is_excluded("path/to/disabled-loader-roots-2026-03-01/readme.md"));
        assert!(ex.is_excluded("path/to/disabled-something-else/file.md"));
    }

    // ── Precision / false-positive tests ─────────────────────────────────────
    // These verify that over-broad prefixes (snapshot-, archive-, old-, bak-)
    // do NOT fire on legitimate user content folders.

    #[test]
    fn keep_legitimate_archive_folder() {
        // A user's real archive dir must NOT be silently dropped.
        let ex = PathExcludeList::load();
        assert!(
            !ex.is_excluded("projects/archive-2024-notes/summary.md"),
            "user archive folder must NOT be excluded by builtin rules"
        );
        assert!(
            !ex.is_excluded("archive-research/deep-dive.md"),
            "user archive folder at vault root must NOT be excluded"
        );
    }

    #[test]
    fn keep_legitimate_snapshot_folder() {
        // "snapshot-" is a common user naming convention for point-in-time notes.
        let ex = PathExcludeList::load();
        assert!(
            !ex.is_excluded("areas/snapshot-weekly/2026-01-02.md"),
            "user snapshot folder must NOT be excluded by builtin rules"
        );
    }

    #[test]
    fn keep_legitimate_old_folder() {
        // "old-" prefixed folders often hold historical reference notes.
        let ex = PathExcludeList::load();
        assert!(
            !ex.is_excluded("old-research/llm-landscape.md"),
            "user old- folder must NOT be excluded by builtin rules"
        );
        assert!(
            !ex.is_excluded("docs/old-architecture/v1.md"),
            "user old- subfolder must NOT be excluded by builtin rules"
        );
    }

    #[test]
    fn keep_legitimate_bak_folder() {
        let ex = PathExcludeList::load();
        assert!(
            !ex.is_excluded("bak-2025-configs/settings.md"),
            "user bak- folder must NOT be excluded by builtin rules"
        );
    }

    #[test]
    fn exclude_extra_patterns_extend_list() {
        // with_extra() mirrors what RECALL_EXCLUDE_EXTRA does at runtime but
        // avoids mutating the global process env (not thread-safe under cargo test).
        let ex = PathExcludeList::with_extra(&["my-archive", "stale-copies"]);

        assert!(ex.is_excluded("path/my-archive/skill.md"));
        assert!(ex.is_excluded("path/stale-copies/skill.md"));
        // canonical path still allowed
        assert!(!ex.is_excluded("projects/recall/PRD.md"));
    }

    #[test]
    fn exclude_extra_patterns_are_case_insensitive() {
        // Patterns are lowercased on load; path segments are lowercased on match.
        let ex = PathExcludeList::with_extra(&["MyArchive"]);
        assert!(ex.is_excluded("some/MyArchive/note.md"));
        assert!(ex.is_excluded("some/myarchive/note.md"));
        assert!(ex.is_excluded("some/MYARCHIVE/note.md"));
    }

    #[test]
    fn empty_path_not_excluded() {
        let ex = PathExcludeList::load();
        assert!(!ex.is_excluded(""));
    }

    #[test]
    fn single_segment_path_not_excluded() {
        let ex = PathExcludeList::load();
        assert!(!ex.is_excluded("README.md"));
    }

    // ── Per-chunk BGE vector tests ────────────────────────────────────────────

    /// Write one chunk vector with empty provenance, for the tests that are about the vector
    /// rows rather than about where a chunk came from.
    fn put_chunk(store: &Store, doc_id: i64, chunk_index: usize, vec: &[u8]) {
        let nowhere = ChunkPlacement {
            heading_path: "",
            char_start: 0,
            char_end: 0,
        };
        store
            .upsert_chunk_embedding(doc_id, chunk_index, vec, 2, &nowhere)
            .unwrap();
    }

    #[test]
    fn chunk_provenance_round_trips_and_slices_the_original_body() {
        // The point of char_start/char_end: a stored hit must be re-checkable against the note
        // it came from. If the offsets do not slice back to the chunk text, they are worse than
        // absent — they would render a passage the chunk never contained.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        // Wide enough to split into several chunks under the test tokenizer's token budget.
        // ⚠️ Must exceed `bge::CHUNK_MAX_TOKENS` in TOTAL, not just the per-chunk target: the
        // whole-document floor emits ONE span for any document the model can read in one pass,
        // so a fixture under the window would silently stop exercising multi-chunk provenance.
        let filler = "word ".repeat(400);
        let body =
            &format!("# Title\n\n## Section A\n\nalpha {filler}\n\n## Section B\n\nbeta {filler}");
        let id = store.insert("n.md", "Title", body, "vault").unwrap();

        let spans =
            crate::bge::chunk_for_embedding_spans("Title", body, &crate::bge::WordTokenizer);
        assert!(spans.len() > 1, "need several chunks, got {}", spans.len());
        for (i, s) in spans.iter().enumerate() {
            store
                .upsert_chunk_embedding(id, i, &[0u8; 8], 2, &s.into())
                .unwrap();
        }

        let mut stmt = store
            .conn
            .prepare(
                "SELECT chunk_index, heading_path, char_start, char_end
                 FROM bge_chunk_embeddings WHERE doc_id = ?1 ORDER BY chunk_index",
            )
            .unwrap();
        let rows: Vec<(usize, String, usize, usize)> = stmt
            .query_map(params![id], |r| {
                Ok((
                    r.get::<_, i64>(0)? as usize,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)? as usize,
                    r.get::<_, i64>(3)? as usize,
                ))
            })
            .unwrap()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(rows.len(), spans.len());
        for (idx, heading, start, end) in rows {
            let span = &spans[idx];
            assert_eq!(heading, span.heading_path);
            assert_eq!((start, end), (span.char_start, span.char_end));
            // BYTE offsets into the body. `text` is the SOFTENED passage (long unbroken runs get
            // spaces inserted so they tokenize), so the two match once spaces are removed.
            //
            // The prefix is one line ("Title > Section B") followed by a blank line, and it VARIES
            // per chunk because it carries the heading path in scope — so split on the first
            // blank line rather than trimming a fixed literal.
            let passage = span.text.split_once("\n\n").map(|(_, p)| p).unwrap_or(&span.text);
            assert_eq!(
                body[start..end].replace(' ', ""),
                passage.replace(' ', ""),
                "chunk {idx} offsets are wrong"
            );
        }
        // Somewhere in there a chunk must know which section it sits under.
        assert!(
            spans.iter().any(|s| s.heading_path.contains("Section B")),
            "heading chain was never captured: {:?}",
            spans.iter().map(|s| &s.heading_path).collect::<Vec<_>>()
        );
    }

    /// 🔴 THE REGRESSION TEST for the partial-re-embed coverage collapse.
    ///
    /// A re-embed runs for hours, so "the chunk table is non-empty" does NOT mean "the chunk table
    /// is authoritative". Observed live at a few hundred documents: preferring the chunk table
    /// outright dropped the query matrix from 136,589 vectors to 4,082, and nothing looked wrong
    /// because the fts5 rescue tier kept answering. The union must keep every document reachable.
    #[test]
    fn a_partially_rechunked_store_still_covers_every_document() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];

        // Three docs with pooled vectors; only ONE has been re-embedded into chunks.
        let a = store.insert("a.md", "A", "body a", "vault").unwrap();
        let b = store.insert("b.md", "B", "body b", "vault").unwrap();
        let c = store.insert("c.md", "C", "body c", "vault").unwrap();
        for id in [a, b, c] {
            store.upsert_embedding(id, &dummy, 2).unwrap();
        }
        put_chunk(&store, a, 0, &dummy);
        put_chunk(&store, a, 1, &dummy);

        let chunks = store.load_all_chunk_embeddings().unwrap();
        let pooled = store.load_pooled_embeddings_without_chunks().unwrap();

        // The re-embedded doc contributes its chunks and must NOT also appear as pooled, or it
        // would be scored twice under two different representations.
        assert_eq!(chunks.iter().filter(|(id, _)| *id == a).count(), 2);
        assert!(
            !pooled.iter().any(|(id, _)| *id == a),
            "a re-chunked doc must not also come back as a pooled row"
        );

        // Every document is still reachable.
        let mut covered: Vec<i64> = chunks
            .iter()
            .chain(pooled.iter())
            .map(|(id, _)| *id)
            .collect();
        covered.sort_unstable();
        covered.dedup();
        assert_eq!(
            covered,
            vec![a, b, c],
            "a partial re-embed lost documents from the query matrix"
        );
    }

    #[test]
    fn chunk_embeddings_table_holds_many_rows_per_doc() {
        // The whole point of the table: `embeddings` is keyed `doc_id INTEGER PRIMARY KEY`
        // and physically cannot hold a second vector for one document. If this ever fails,
        // long documents are being pooled back into one row again.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = store.insert("long.md", "Long", "body", "cli").unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        for i in 0..5 {
            put_chunk(&store, id, i, &dummy);
        }
        let rows: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM bge_chunk_embeddings WHERE doc_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(rows, 5);
        // Counts DOCUMENTS, not rows — one doc with five chunks is one doc.
        assert_eq!(store.count_chunk_embedded_docs().unwrap(), 1);
    }

    #[test]
    fn upsert_chunk_embedding_replaces_the_same_index_not_appends() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = store.insert("a.md", "A", "body", "cli").unwrap();
        put_chunk(&store, id, 0, &[1u8; 8]);
        put_chunk(&store, id, 0, &[2u8; 8]);
        let loaded = store.load_all_chunk_embeddings().unwrap();
        assert_eq!(loaded.len(), 1, "re-embedding must not duplicate a chunk");
    }

    #[test]
    fn delete_chunk_embeddings_clears_a_shrunk_documents_tail() {
        // A note that shrank from 5 chunks to 2 must not keep chunks 2..5 of its previous
        // body — those passages no longer exist and would go on matching queries.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = store.insert("a.md", "A", "body", "cli").unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        for i in 0..5 {
            put_chunk(&store, id, i, &dummy);
        }
        assert_eq!(store.delete_chunk_embeddings(id).unwrap(), 5);
        for i in 0..2 {
            put_chunk(&store, id, i, &dummy);
        }
        assert_eq!(store.load_all_chunk_embeddings().unwrap().len(), 2);
    }

    #[test]
    fn edited_document_is_reported_as_needing_embedding() {
        // The bug this whole change exists to remove: the queue asked "does this document have
        // ANY vectors?" and never "are they CURRENT?". A note edited after being embedded kept
        // its old vectors forever and went on matching text it no longer contains.
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = s.insert("a.md", "T", "original body", "test").unwrap();
        put_chunk(&s, id, 0, &[0u8; 8]);

        // A document that HAS been embedded, at its current content, is not stale.
        s.conn
            .execute(
                "UPDATE documents SET content_sha256 = ?1, embedded_sha256 = ?1 WHERE id = ?2",
                params![sha256_hex(b"original body"), id],
            )
            .unwrap();
        assert!(
            s.docs_missing_chunk_embeddings(10).unwrap().is_empty(),
            "an embedded, unedited document must not be queued"
        );

        // Edited: the content hash moves, the embedded hash does not. It MUST be re-queued.
        s.conn
            .execute(
                "UPDATE documents SET body = ?1, content_sha256 = ?2 WHERE id = ?3",
                params!["EDITED body", sha256_hex(b"EDITED body"), id],
            )
            .unwrap();
        let stale = s.docs_missing_chunk_embeddings(10).unwrap();
        assert_eq!(stale.len(), 1, "an edited document must be re-embedded");
        assert_eq!(stale[0].0, id);
        assert_eq!(stale[0].2, "EDITED body", "the queue must hand back the NEW body");
    }

    #[test]
    fn embedded_before_the_hash_columns_existed_is_reported_as_stale() {
        // The NULL case. Every document in the live corpus is in exactly this state the moment
        // the columns are added: it has vectors, and no record of what they were built from.
        // A predicate written as `content_sha256 <> embedded_sha256` — or as
        // `content_sha256 IS NOT embedded_sha256` — silently excludes ALL of them, because
        // `<>` against NULL is NULL and `NULL IS NOT NULL` is false.
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = s.insert("a.md", "T", "body", "test").unwrap();
        put_chunk(&s, id, 0, &[0u8; 8]);
        s.conn
            .execute(
                "UPDATE documents SET content_sha256 = NULL, embedded_sha256 = NULL WHERE id = ?1",
                params![id],
            )
            .unwrap();
        let stale = s.docs_missing_chunk_embeddings(10).unwrap();
        assert_eq!(
            stale.len(),
            1,
            "a document with vectors but no embedded_sha256 must be re-embedded"
        );
    }

    #[test]
    fn writing_chunks_stamps_embedded_sha256_in_the_same_transaction() {
        // The stamp is a CLAIM about the vectors: "these were built from this body". A missing
        // stamp is the SAFE direction — it leaves the document queued. The dangerous direction
        // is a stamp that survives a vector replacement it does not describe, which is what
        // `an_edit_then_undo_during_embed_does_not_replace_good_vectors` covers.
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = s.insert("a.md", "T", "the body", "test").unwrap();
        s.backfill_content_hashes(10).unwrap();

        let stale_before = s.docs_missing_chunk_embeddings(10).unwrap().len();
        assert_eq!(stale_before, 1, "never embedded, so it starts stale");

        let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 8 };
        s.replace_chunk_embeddings(id, 2, &[(vec![0u8; 8], place)], &sha256_hex(b"the body"))
            .unwrap();

        let stamped: Option<String> = s
            .conn
            .query_row(
                "SELECT embedded_sha256 FROM documents WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stamped.as_deref(), Some(sha256_hex(b"the body").as_str()));
        assert!(
            s.docs_missing_chunk_embeddings(10).unwrap().is_empty(),
            "after writing vectors the document must leave the queue"
        );
    }

    #[test]
    fn replacing_chunks_clears_the_tail_of_a_shrunk_document() {
        // Same discipline delete_chunk_embeddings existed for: an upsert only overwrites the
        // indices it is handed, so a note that shrank from 5 chunks to 1 would keep 4 chunks of
        // its previous body and go on matching text it no longer contains.
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = s.insert("a.md", "T", "body", "test").unwrap();
        for i in 0..5 {
            put_chunk(&s, id, i, &[0u8; 8]);
        }
        let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 4 };
        s.replace_chunk_embeddings(id, 2, &[(vec![1u8; 8], place)], &sha256_hex(b"body"))
            .unwrap();
        let n: i64 = s
            .conn
            .query_row(
                "SELECT count(*) FROM bge_chunk_embeddings WHERE doc_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "the shrunk document's old chunk tail must be gone");
    }

    #[test]
    fn docs_missing_chunk_embeddings_skips_docs_with_any_chunk() {
        // Guards the embed loop against an infinite re-run: a document the loop has just
        // finished must LEAVE the queue, or `recall embed` re-embeds the same batch until it
        // is killed.
        //
        // 🔴 Updated with the staleness fix, and deliberately NOT weakened. Under the old
        // contract "has any chunk row" was enough to leave the queue, so `put_chunk` alone
        // satisfied it. Under the new one a document also has to carry a matching
        // `embedded_sha256` — which is the whole point, and which is why this must go through
        // `replace_chunk_embeddings`, the path the embed loop actually uses. Writing chunk rows
        // by hand and expecting the document to drain would be asserting the old bug.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let done = store.insert("done.md", "D", "body d", "cli").unwrap();
        let todo = store.insert("todo.md", "T", "body t", "cli").unwrap();
        store.backfill_content_hashes(10).unwrap();
        let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 6 };
        store
            .replace_chunk_embeddings(done, 2, &[(vec![0u8; 8], place)], &sha256_hex(b"body d"))
            .unwrap();

        let work = store.docs_missing_chunk_embeddings(10).unwrap();
        let ids: Vec<i64> = work.iter().map(|(id, _, _)| *id).collect();
        assert_eq!(ids, vec![todo]);
    }

    fn move_probe_dir(tag: &str) -> std::path::PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let d = std::env::temp_dir().join(format!("recall_move_{tag}_{}_{stamp}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn live_id_like(s: &Store, frag: &str) -> Option<i64> {
        s.conn
            .query_row(
                "SELECT id FROM documents WHERE path LIKE ?1 AND valid_to IS NULL",
                params![format!("%{frag}%")],
                |r| r.get(0),
            )
            .ok()
    }

    fn vec_count(s: &Store, id: i64) -> i64 {
        s.conn
            .query_row(
                "SELECT count(*) FROM bge_chunk_embeddings WHERE doc_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap()
    }

    /// The wide->narrow `documents_au` migration must actually run on a PRE-EXISTING database.
    ///
    /// Every other test builds its fixture with `Store::open_at(":memory:")`, which gets the
    /// narrow trigger straight from `CREATE TRIGGER IF NOT EXISTS` — so no test ever constructed
    /// the wide trigger, and the migration block could be replaced with `if false {}` with all
    /// 269 tests still green. It is the only code that applies the change to the 7.1 GB file,
    /// and the measured 10.2x / ~419 MB benefit exists only if it runs.
    #[test]
    fn the_wide_documents_au_trigger_is_migrated_to_the_narrow_one() {
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        // Put the OLD wide trigger back, exactly as a pre-upgrade database has it.
        s.conn
            .execute_batch(
                "DROP TRIGGER documents_au;
                 CREATE TRIGGER documents_au AFTER UPDATE ON documents BEGIN
                     INSERT INTO documents_fts(documents_fts, rowid, title, body)
                     VALUES ('delete', old.id, old.title, old.body);
                     INSERT INTO documents_fts(rowid, title, body)
                     VALUES (new.id, new.title, new.body);
                 END;",
            )
            .unwrap();
        let wide: String = s
            .conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name='documents_au'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(!wide.contains("UPDATE OF"), "fixture must start WIDE");

        // Re-running init_schema is what `Store::open` does on every open.
        s.init_schema().unwrap();

        let after: String = s
            .conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name='documents_au'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            after.contains("AFTER UPDATE OF title, body"),
            "the migration must narrow an existing wide trigger, got: {after}"
        );

        // And it must be idempotent - a second open must not thrash it.
        s.init_schema().unwrap();
        let again: String = s
            .conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name='documents_au'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(after, again, "the migration must be a no-op the second time");
    }

    /// The `path_is_absent` guard in move detection must be load-bearing on its own.
    ///
    /// It was not: deleting it left all 269 tests green, because `unique_incoming` already
    /// suppressed the only copy case any test built — `incoming_digests` counts every enumerated
    /// file, so a copy whose original is still inside the tree always has count >= 2.
    ///
    /// The case that isolates this guard is a candidate the walk NEVER SEES: a document whose
    /// file lives outside the vault root, so it contributes nothing to `incoming_digests` and
    /// the count is 1. `.claude/` path-list entries are exactly that — they resolve under the
    /// HOME, not the root. If `path_is_absent` is removed, or resolves them against the root as
    /// it used to, that live document is stolen by the first vault file with the same body.
    #[test]
    fn a_candidate_whose_file_still_exists_outside_the_root_is_not_stolen() {
        let root = move_probe_dir("outside-root");
        let home = move_probe_dir("outside-home");
        let claude = home.join(".claude").join("skills").join("dup");
        std::fs::create_dir_all(&claude).unwrap();
        let body = "# Shared\n\nthis exact body exists in two places";
        std::fs::write(claude.join("SKILL.md"), body).unwrap();

        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        // The `.claude` document, as `ingest_paths` would have created it: a live vault row
        // whose file is real but lives outside `root`.
        let outside = s
            .insert(".claude/skills/dup/SKILL.md", "Shared", body, "vault")
            .unwrap();
        let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 4 };
        s.replace_chunk_embeddings(outside, 2, &[(vec![0u8; 8], place)], &sha256_hex(body.as_bytes()))
            .unwrap();

        // A vault file with the SAME body appears. The walk sees only this one, so
        // `unique_incoming` is 1 and cannot help; only `path_is_absent` answering "the candidate
        // is still on disk" stops the theft.
        std::fs::write(root.join("copy.md"), body).unwrap();

        let prev = std::env::var_os("USERPROFILE");
        // SAFETY: single-threaded test; restored below.
        unsafe { std::env::set_var("USERPROFILE", &home) };
        let res = s.ingest_dir(&root);
        match prev {
            Some(v) => unsafe { std::env::set_var("USERPROFILE", v) },
            None => unsafe { std::env::remove_var("USERPROFILE") },
        }
        res.unwrap();

        let still_there: i64 = s
            .conn
            .query_row(
                "SELECT count(*) FROM documents WHERE id = ?1 AND valid_to IS NULL AND path = '.claude/skills/dup/SKILL.md'",
                params![outside],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            still_there, 1,
            "a document whose file STILL EXISTS must not be moved onto another file's path"
        );
        assert_eq!(vec_count(&s, outside), 1, "and it must keep its vectors");
        let copy_id = live_id_like(&s, "copy.md").expect("the new file must be indexed");
        assert_ne!(copy_id, outside, "the new file must be its OWN document");

        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&home).ok();
    }

    /// The embed queue must not contain documents the QUERY path refuses to return.
    ///
    /// `TierFilter`'s default is `["BRAIN","VENDOR"]`, so a SKILLS document is dropped in SQL
    /// before the loader's Rust filter runs — no search can ever reach it. Measured on the live
    /// corpus: **7,377 live SKILLS documents / 69,204 chunk rows** were being queued for
    /// embedding into a matrix that would never serve them.
    ///
    /// A `tier IS NULL` document must still be queued: the tier clause is
    /// `tier IS NULL OR tier IN (...)`, and 47 live rows have no tier at all.
    #[test]
    fn the_embed_queue_excludes_tiers_the_query_path_will_not_serve() {
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let brain = s.insert("projects/a.md", "A", "brain body", "vault").unwrap();
        let skills = s.insert("skills/b/SKILL.md", "B", "skills body", "vault").unwrap();
        let untiered = s.insert("c.md", "C", "untiered body", "vault").unwrap();
        // Set tiers explicitly rather than relying on `tier_for_path`, so this test is about the
        // QUEUE and not about path classification.
        s.conn
            .execute("UPDATE documents SET tier='BRAIN'  WHERE id=?1", params![brain])
            .unwrap();
        s.conn
            .execute("UPDATE documents SET tier='SKILLS' WHERE id=?1", params![skills])
            .unwrap();
        s.conn
            .execute("UPDATE documents SET tier=NULL     WHERE id=?1", params![untiered])
            .unwrap();

        let queued: Vec<i64> = s
            .docs_missing_chunk_embeddings(50)
            .unwrap()
            .iter()
            .map(|(id, _, _)| *id)
            .collect();

        assert!(queued.contains(&brain), "a BRAIN document must be queued");
        assert!(
            queued.contains(&untiered),
            "an untiered document must be queued - the clause is `tier IS NULL OR tier IN (...)`"
        );
        assert!(
            !queued.contains(&skills),
            "a SKILLS document must NOT be queued: the query path will never return it, so \
             embedding it is work that can never be used"
        );

        // The counter and the queue must agree, or a run reports work it will not do.
        assert_eq!(
            s.count_docs_needing_chunk_embeddings().unwrap() as usize,
            queued.len(),
            "the count and the queue must be the same predicate"
        );
    }

    /// 🔴 A rename must be detected on a database that has NOT been backfilled.
    ///
    /// This is the production state, and it is the one every other move test skips. Adding
    /// `content_sha256` to an existing database leaves it NULL on every row; `ingest_dir`'s
    /// `unchanged` arm writes nothing, so a document nobody has edited since the migration stays
    /// NULL indefinitely. Both move predicates are `content_sha256 = ?1`, and **NULL never
    /// equals anything**, so move detection finds no candidates, reports `0 moved (renamed)`,
    /// and is indistinguishable from "no renames happened".
    ///
    /// Every other move test passes because it builds its fixture with `insert`/`ingest_dir` on
    /// a fresh database, where the hash IS written at insert time. On a real, older database the
    /// column was NULL on every row, so the feature would have shipped inert.
    #[test]
    fn a_rename_is_detected_on_a_database_that_was_never_backfilled() {
        let dir = move_probe_dir("nobackfill");
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();

        let a = dir.join("before.md");
        std::fs::write(&a, "# T\n\na body unique to the no-backfill test").unwrap();
        s.ingest_dir(&dir).unwrap();
        let id = live_id_like(&s, "before").expect("indexed");
        let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 4 };
        s.replace_chunk_embeddings(
            id,
            2,
            &[(vec![0u8; 8], place)],
            &sha256_hex(b"# T\n\na body unique to the no-backfill test"),
        )
        .unwrap();

        // 🔴 Reproduce the production state: the column exists and is NULL, exactly as
        // `add_column_if_missing` leaves it on a pre-existing database.
        s.conn
            .execute("UPDATE documents SET content_sha256 = NULL", [])
            .unwrap();

        std::fs::rename(&a, dir.join("after.md")).unwrap();
        s.ingest_dir(&dir).unwrap();

        let id2 = live_id_like(&s, "after").expect("re-indexed");
        assert_eq!(
            id2, id,
            "a rename must be detected even when the corpus was never backfilled"
        );
        assert_eq!(vec_count(&s, id), 1, "and the vectors must come with it");
        assert_eq!(live_id_like(&s, "before"), None, "old path must not stay live");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// 🔴 GATE 1. A rename must KEEP the document and its vectors.
    ///
    /// Identity was path equality alone, so renaming a note left its old row live and pointing at
    /// a file that no longer exists, and created a second row with ZERO vectors - the note lost
    /// semantic search until a full re-index.
    ///
    /// A real filesystem is required: the deciding fact is that the OLD PATH IS GONE. Two writes
    /// at two paths do not describe a rename, because the first file would still be there.
    #[test]
    fn renaming_a_file_keeps_its_id_and_vectors() {
        let dir = move_probe_dir("rename");
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();

        let a = dir.join("old-name.md");
        std::fs::write(&a, "# T\n\na body that is unique to this test").unwrap();
        s.ingest_dir(&dir).unwrap();
        let id = live_id_like(&s, "old-name").expect("indexed");
        let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 4 };
        s.replace_chunk_embeddings(id, 2, &[(vec![0u8; 8], place)], &sha256_hex(b"# T\n\na body that is unique to this test"))
            .unwrap();

        // a REAL rename: the old path no longer exists on disk
        std::fs::rename(&a, dir.join("new-name.md")).unwrap();
        s.ingest_dir(&dir).unwrap();

        let id2 = live_id_like(&s, "new-name").expect("re-indexed under the new name");
        assert_eq!(id, id2, "a rename is a MOVE - the id must be kept");
        assert_eq!(vec_count(&s, id), 1, "vectors must survive a rename");
        // Count alone is too weak: assert the BYTES and the provenance columns are the ones
        // that were written, not a fresh row that merely happens to exist.
        let (bytes, dim, hp, cs, ce): (Vec<u8>, i64, String, i64, i64) = s
            .conn
            .query_row(
                "SELECT vec, dim, heading_path, char_start, char_end
                   FROM bge_chunk_embeddings WHERE doc_id = ?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .unwrap();
        assert_eq!(bytes, vec![0u8; 8], "the original vector bytes must survive");
        assert_eq!((dim, hp.as_str(), cs, ce), (2, "", 0, 4), "and its placement");

        // Row counts do not prove the document is still FINDABLE. Ask the index.
        let hits: i64 = s
            .conn
            .query_row(
                "SELECT count(*) FROM documents_fts WHERE documents_fts MATCH ?1",
                params!["unique"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(hits, 1, "the moved document must still be findable by search");
        // ...and the FTS index must still agree with the documents table after the move.
        s.conn
            .execute_batch("INSERT INTO documents_fts(documents_fts, rank) VALUES('integrity-check', 1);")
            .expect("FTS external-content integrity must hold after a move");
        assert_eq!(
            live_id_like(&s, "old-name"),
            None,
            "the old path must not still be live - that is the row that used to be orphaned"
        );
        let live: i64 = s
            .conn
            .query_row("SELECT count(*) FROM documents WHERE valid_to IS NULL", [], |r| r.get(0))
            .unwrap();
        assert_eq!(live, 1, "a rename must leave exactly ONE live document");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A rename that CHANGES the derived title must invalidate the embedding.
    ///
    /// `derive_title` falls back to the FILENAME when a note has no H1, so renaming such a note
    /// changes the title — and the title is part of the text that was embedded. `content_sha256`
    /// covers the body only, so nothing else in the system would notice. The move must therefore
    /// clear `embedded_sha256`, and the proof is that the real work queue returns the document.
    ///
    /// The plain-rename gate above deliberately uses a note WITH an H1, so its title does not
    /// move; between them the two branches of the move are both covered.
    #[test]
    fn a_rename_that_changes_the_derived_title_requeues_the_document() {
        let dir = move_probe_dir("titlechange");
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();

        // NO H1 anywhere, so the title comes from the filename.
        let body = "just a paragraph with no heading at all";
        let a = dir.join("first-name.md");
        std::fs::write(&a, body).unwrap();
        s.ingest_dir(&dir).unwrap();
        let id = live_id_like(&s, "first-name").expect("indexed");
        let t1: String = s
            .conn
            .query_row("SELECT title FROM documents WHERE id = ?1", params![id], |r| r.get(0))
            .unwrap();

        let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 4 };
        s.replace_chunk_embeddings(id, 2, &[(vec![0u8; 8], place)], &sha256_hex(body.as_bytes()))
            .unwrap();
        assert!(
            s.docs_missing_chunk_embeddings(10).unwrap().is_empty(),
            "embedded and certified, so nothing queued"
        );

        std::fs::rename(&a, dir.join("second-name.md")).unwrap();
        s.ingest_dir(&dir).unwrap();

        let id2 = live_id_like(&s, "second-name").expect("moved");
        assert_eq!(id2, id, "still a move - the id is kept");
        let t2: String = s
            .conn
            .query_row("SELECT title FROM documents WHERE id = ?1", params![id], |r| r.get(0))
            .unwrap();
        assert_ne!(t1, t2, "fixture is only meaningful if the title actually changed");
        assert_eq!(vec_count(&s, id), 1, "the vectors are kept - they are not wrong, just stale");

        let stale = s.docs_missing_chunk_embeddings(10).unwrap();
        assert_eq!(
            stale.len(),
            1,
            "a title change must re-queue the document through the REAL work queue"
        );
        assert_eq!(stale[0].0, id);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// 🔴 GATE 2. A COPY must produce TWO documents. If this ever yields 1, the identity-theft
    /// bug is live: a copied note has taken the original's id, and the original has silently
    /// stopped being findable.
    #[test]
    fn copying_a_file_creates_a_second_document() {
        let dir = move_probe_dir("copy");
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();

        std::fs::write(dir.join("one.md"), "# T\n\nidentical body").unwrap();
        s.ingest_dir(&dir).unwrap();
        // BOTH files now exist, with byte-identical bodies.
        std::fs::write(dir.join("two.md"), "# T\n\nidentical body").unwrap();
        s.ingest_dir(&dir).unwrap();

        let n: i64 = s
            .conn
            .query_row("SELECT count(*) FROM documents WHERE valid_to IS NULL", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 2, "a copy must NEVER take another document's identity");
        assert!(live_id_like(&s, "one.md").is_some());
        assert!(live_id_like(&s, "two.md").is_some());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// 🔴 A rename AND a copy of the same content in ONE index run must not let the COPY take
    /// the original's identity.
    ///
    /// This is the case that forced move resolution out of the per-file loop and into a pass that
    /// runs after the whole enumeration. Rename `a.md` to `z.md`, then copy `z.md` to `b.md`, and
    /// index once. File by file each of them looks exactly like a plain rename: there is one
    /// candidate in the database and `a.md` really is gone. Whichever file the walk visits first
    /// would have claimed the id and the vectors, and the real renamed file would have got a
    /// fresh id with none. Only "exactly ONE incoming file carries this content" separates them,
    /// and that is not knowable until every file has been read.
    #[test]
    fn a_rename_and_a_copy_in_one_run_never_steal_the_identity() {
        let dir = move_probe_dir("renamecopy");
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();

        let a = dir.join("a.md");
        std::fs::write(&a, "# T\n\nthe body that gets copied around").unwrap();
        s.ingest_dir(&dir).unwrap();
        let orig = live_id_like(&s, "a.md").expect("indexed");
        let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 4 };
        s.replace_chunk_embeddings(
            orig,
            2,
            &[(vec![0u8; 8], place)],
            &sha256_hex(b"# T\n\nthe body that gets copied around"),
        )
        .unwrap();

        // rename a.md -> z.md, THEN copy z.md -> b.md, all before the next index
        std::fs::rename(&a, dir.join("z.md")).unwrap();
        std::fs::copy(dir.join("z.md"), dir.join("b.md")).unwrap();
        s.ingest_dir(&dir).unwrap();

        let z = live_id_like(&s, "z.md").expect("z indexed");
        let b = live_id_like(&s, "b.md").expect("b indexed");
        assert_ne!(z, b, "two files, two documents");
        // Two incoming files carry this content, so which one moved is UNKNOWABLE. Neither may
        // claim the original id - ambiguity must produce new documents, not a guess.
        assert_ne!(b, orig, "the COPY must never take the original's id");
        assert_ne!(z, orig, "and neither may the rename, while it is ambiguous");
        let live: i64 = s
            .conn
            .query_row("SELECT count(*) FROM documents WHERE valid_to IS NULL", [], |r| r.get(0))
            .unwrap();
        assert_eq!(live, 3, "the stranded original plus the two new files");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// 🔴 Renaming a live note ONTO a path some other note used to occupy must move the LIVE
    /// note, not revive the tombstone.
    ///
    /// `ingest_dir` revives a tombstoned row when a file reappears at its path. Taken first, that
    /// lookup wins over move detection: the tombstone (which has NO vectors) comes back, and the
    /// real document stays live at a path that no longer exists. Move detection is therefore
    /// consulted first, and suppresses the revive when a live candidate exists.
    #[test]
    fn renaming_onto_a_tombstoned_path_moves_the_live_note_not_the_tombstone() {
        let dir = move_probe_dir("tombcollide");
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();

        // an older note at target.md, later deleted and tombstoned
        std::fs::write(dir.join("target.md"), "# Old\n\nan older unrelated note").unwrap();
        s.ingest_dir(&dir).unwrap();
        let tomb = live_id_like(&s, "target.md").unwrap();
        std::fs::remove_file(dir.join("target.md")).unwrap();
        s.conn
            .execute(
                "UPDATE documents SET valid_to = '1' WHERE id = ?1",
                params![tomb],
            )
            .unwrap();

        // the live note we actually care about, with vectors
        let src = dir.join("source.md");
        std::fs::write(&src, "# Live\n\nthe note that is about to be renamed").unwrap();
        s.ingest_dir(&dir).unwrap();
        let live_id = live_id_like(&s, "source.md").unwrap();
        let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 4 };
        s.replace_chunk_embeddings(
            live_id,
            2,
            &[(vec![0u8; 8], place)],
            &sha256_hex(b"# Live\n\nthe note that is about to be renamed"),
        )
        .unwrap();

        // rename it ONTO the tombstoned path
        std::fs::rename(&src, dir.join("target.md")).unwrap();
        s.ingest_dir(&dir).unwrap();

        let now_at_target = live_id_like(&s, "target.md").expect("indexed");
        assert_eq!(
            now_at_target, live_id,
            "the LIVE note must move onto the path, not the tombstone be revived"
        );
        assert_ne!(now_at_target, tomb, "the tombstone must NOT have been revived");
        assert_eq!(vec_count(&s, live_id), 1, "and it must keep its vectors");
        assert_eq!(
            live_id_like(&s, "source.md"),
            None,
            "nothing may still be live at the old path"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// 🔴 The move-versus-revive decision must not depend on the order the walk returns files.
    ///
    /// The counterexample that killed the first fix: `src.md` and a RIVAL note hold identical
    /// content H, and the destination path carries an unrelated tombstone. Rename `src.md` onto
    /// that path and edit the rival away from H in the same run. If the decision is taken while
    /// walking, then reaching the destination BEFORE the rival sees two live candidates for H —
    /// the source and the not-yet-updated rival — calls it ambiguous, revives the tombstone, and
    /// never reconsiders. Reaching them the other way round moves the document correctly. Same
    /// inputs, different answer.
    ///
    /// Two identical fixtures inject opposite orders into the actual list consumed by ingest_dir.
    /// The observed consumed lists must differ; no filesystem iteration-order promise is needed.
    /// Disabling the deferral turns this test red and names the failing order.
    #[test]
    fn move_versus_revive_does_not_depend_on_enumeration_order() {
        let mut observed_orders: Vec<bool> = Vec::new();
        for reverse in [false, true] {
            let rival_name = "rival.md";
            let dir = move_probe_dir(if reverse { "order-reverse" } else { "order-forward" });
            let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();

            // an unrelated note at target.md, later tombstoned
            std::fs::write(dir.join("target.md"), "# Old\n\nsomething unrelated entirely").unwrap();
            s.ingest_dir(&dir).unwrap();
            let tomb = live_id_like(&s, "target.md").unwrap();
            std::fs::remove_file(dir.join("target.md")).unwrap();
            s.conn
                .execute("UPDATE documents SET valid_to = '1' WHERE id = ?1", params![tomb])
                .unwrap();

            // the source and a RIVAL with byte-identical content H
            let shared = "# Same\n\nthese two notes are byte identical for now";
            std::fs::write(dir.join("src.md"), shared).unwrap();
            std::fs::write(dir.join(rival_name), shared).unwrap();
            s.ingest_dir(&dir).unwrap();
            let src_id = live_id_like(&s, "src.md").unwrap();
            let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 4 };
            s.replace_chunk_embeddings(src_id, 2, &[(vec![0u8; 8], place)], &sha256_hex(shared.as_bytes()))
                .unwrap();

            // in ONE run: rename src.md onto the tombstoned path, and edit the rival away from H
            std::fs::rename(dir.join("src.md"), dir.join("target.md")).unwrap();
            std::fs::write(dir.join(rival_name), "# Same\n\nthe rival has been edited to J").unwrap();

            let order = EnumerationOrder::set(reverse);
            s.ingest_dir(&dir).unwrap();
            let consumed = order.observed();
            drop(order);
            assert_eq!(consumed.len(), 2);
            let target_at = consumed.iter().position(|p| p.file_name().unwrap() == "target.md").unwrap();
            let rival_at = consumed.iter().position(|p| p.file_name().unwrap() == rival_name).unwrap();
            observed_orders.push(target_at < rival_at);

            let at_target = live_id_like(&s, "target.md").expect("indexed");
            assert_eq!(
                at_target, src_id,
                "the LIVE note must move onto the path with reverse order = {reverse}"
            );
            assert_ne!(at_target, tomb, "the tombstone must not be revived");
            assert_eq!(vec_count(&s, src_id), 1, "and the vectors must come with it");
            assert_eq!(live_id_like(&s, "src.md"), None, "nothing left live at the old path");

            std::fs::remove_dir_all(&dir).ok();
        }
        assert_eq!(observed_orders, [false, true], "the ingester must consume both orders");
    }

    /// Ambiguity is never resolved by guessing. Two live documents share a body, both are then
    /// deleted, and a third file appears with that same body: there is no way to know which one
    /// moved, so it must become a NEW document rather than steal either identity.
    #[test]
    fn an_ambiguous_content_match_is_never_treated_as_a_move() {
        let dir = move_probe_dir("ambig");
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();

        std::fs::write(dir.join("a.md"), "# T\n\nshared body text").unwrap();
        std::fs::write(dir.join("b.md"), "# T\n\nshared body text").unwrap();
        s.ingest_dir(&dir).unwrap();
        let a = live_id_like(&s, "a.md").unwrap();
        let b = live_id_like(&s, "b.md").unwrap();

        std::fs::remove_file(dir.join("a.md")).unwrap();
        std::fs::remove_file(dir.join("b.md")).unwrap();
        std::fs::write(dir.join("c.md"), "# T\n\nshared body text").unwrap();
        s.ingest_dir(&dir).unwrap();

        let c = live_id_like(&s, "c.md").expect("indexed");
        assert_ne!(c, a, "must not steal a.md's identity");
        assert_ne!(c, b, "must not steal b.md's identity");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// An EMPTY body must never be move-matched. Measured on a real corpus: of its live
    /// documents there is exactly ONE content hash shared by more than one row, and it is the
    /// SHA-256 of the empty string, shared by several rows. Matching on it would let any new empty
    /// file inherit an arbitrary one of them.
    #[test]
    fn an_empty_body_is_never_treated_as_a_move() {
        let dir = move_probe_dir("empty");
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();

        std::fs::write(dir.join("blank-one.md"), "").unwrap();
        s.ingest_dir(&dir).unwrap();
        let first = live_id_like(&s, "blank-one").expect("indexed");

        std::fs::remove_file(dir.join("blank-one.md")).unwrap();
        std::fs::write(dir.join("blank-two.md"), "").unwrap();
        s.ingest_dir(&dir).unwrap();

        let second = live_id_like(&s, "blank-two").expect("indexed");
        assert_ne!(
            first, second,
            "an empty file must not inherit another empty file's identity"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A rename that ALSO edits the body is NOT caught by a content-hash fallback, by
    /// construction: the hash it would match on has changed. This test pins that as the KNOWN
    /// limitation rather than leaving it to be discovered. Catching it needs a durable
    /// identifier (frontmatter `uid:` or the NTFS file id), which this change does not add.
    #[test]
    fn rename_plus_edit_is_a_new_document_and_that_is_the_known_limitation() {
        let dir = move_probe_dir("renamedit");
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();

        let a = dir.join("before.md");
        std::fs::write(&a, "# T\n\noriginal unique body").unwrap();
        s.ingest_dir(&dir).unwrap();
        let id = live_id_like(&s, "before").unwrap();

        std::fs::remove_file(&a).unwrap();
        std::fs::write(dir.join("after.md"), "# T\n\nCHANGED unique body").unwrap();
        s.ingest_dir(&dir).unwrap();

        let id2 = live_id_like(&s, "after").unwrap();
        assert_ne!(
            id, id2,
            "documented limitation: rename+edit is not detectable from content alone"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn editing_a_real_file_on_disk_requeues_it_end_to_end() {
        // END TO END, through the REAL ingest entry point and real files on disk. The unit
        // tests above drive the schema directly, which proves the QUERY is right but not that
        // the ingest path maintains the hash it depends on. This is the whole cycle:
        //
        //   index -> embed -> (edit the file) -> index -> MUST be queued again
        //
        // The bug being guarded: `ingest_dir`'s update arm keeps the document id, so before
        // this change an edited note silently kept vectors built from text it no longer had.
        use std::fs;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "recall_stale_e2e_{}_{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let note = root.join("note.md");
        fs::write(&note, "# Title\n\nthe original body text").unwrap();

        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        s.ingest_dir(&root).unwrap();

        let id: i64 = s
            .conn
            .query_row(
                "SELECT id FROM documents WHERE path LIKE '%note%' AND valid_to IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        // The ingest path itself must set the content hash - not the backfill.
        let h1: Option<String> = s
            .conn
            .query_row(
                "SELECT content_sha256 FROM documents WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(h1.is_some(), "ingest must write content_sha256 on insert");

        // Embed it, the way the embed loop does.
        let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 4 };
        s.replace_chunk_embeddings(
            id,
            2,
            &[(vec![0u8; 8], place)],
            &sha256_hex(b"# Title\n\nthe original body text"),
        )
        .unwrap();
        assert!(
            s.docs_missing_chunk_embeddings(10).unwrap().is_empty(),
            "freshly embedded, so nothing to do"
        );

        // Now EDIT the file on disk and re-index it.
        fs::write(&note, "# Title\n\nCOMPLETELY DIFFERENT body text now").unwrap();
        s.ingest_dir(&root).unwrap();

        let id2: i64 = s
            .conn
            .query_row(
                "SELECT id FROM documents WHERE path LIKE '%note%' AND valid_to IS NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(id2, id, "an edit updates in place - the id must survive");

        let h2: Option<String> = s
            .conn
            .query_row(
                "SELECT content_sha256 FROM documents WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_ne!(h1, h2, "the content hash must move when the body changes");

        let vectors_kept: i64 = s
            .conn
            .query_row(
                "SELECT count(*) FROM bge_chunk_embeddings WHERE doc_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(vectors_kept, 1, "the vectors are still attached - that is the bug");

        let stale = s.docs_missing_chunk_embeddings(10).unwrap();
        assert_eq!(
            stale.len(),
            1,
            "an edited file MUST be re-queued for embedding"
        );
        assert_eq!(stale[0].0, id);
        assert!(
            stale[0].2.contains("COMPLETELY DIFFERENT"),
            "the queue must hand back the NEW body, not the embedded one"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_document_rewritten_mid_embed_is_not_certified_fresh() {
        // 🔴 Embedding happens OUTSIDE the write transaction and takes seconds. If an ingest
        // rewrites the document in that gap, stamping `embedded_sha256 = content_sha256` would
        // certify the NEW body against the OLD body's vectors and drop it out of the queue -
        // the staleness bug, reintroduced by its own fix. The stamp must be withheld instead.
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = s.insert("a.md", "T", "body AS READ", "test").unwrap();
        let read_hash = sha256_hex(b"body AS READ");

        // the ingest that lands while the embedder is busy
        s.conn
            .execute(
                "UPDATE documents SET body = ?1, content_sha256 = ?2 WHERE id = ?3",
                params!["body AFTER EDIT", sha256_hex(b"body AFTER EDIT"), id],
            )
            .unwrap();

        let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 4 };
        let stamped = s
            .replace_chunk_embeddings(id, 2, &[(vec![0u8; 8], place)], &read_hash)
            .unwrap();
        assert!(!stamped, "the stamp must be withheld when the row moved");

        let embedded: Option<String> = s
            .conn
            .query_row(
                "SELECT embedded_sha256 FROM documents WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(embedded, None, "nothing may certify vectors it did not build");
        let vecs: i64 = s
            .conn
            .query_row(
                "SELECT count(*) FROM bge_chunk_embeddings WHERE doc_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(vecs, 0, "a rejected stamp must roll the vector write back too");
        assert_eq!(
            s.docs_missing_chunk_embeddings(10).unwrap().len(),
            1,
            "the document must STAY queued and be re-embedded from its current body"
        );
    }

    #[test]
    fn an_edit_then_undo_during_embed_does_not_replace_good_vectors() {
        // The A -> B -> A interleaving. This is the case that makes committing the vectors after
        // a rejected stamp unsafe: the row ends up looking perfectly fresh (both hashes A) while
        // holding vectors built from B, and NOTHING will ever select it for repair.
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = s.insert("a.md", "T", "body A", "test").unwrap();
        let hash_a = sha256_hex(b"body A");

        // 1. correctly embedded at body A
        let good = ChunkPlacement { heading_path: "", char_start: 0, char_end: 6 };
        assert!(s
            .replace_chunk_embeddings(id, 2, &[(vec![0xAA; 8], good)], &hash_a)
            .unwrap());

        // 2. ingest writes body B; 3. the embedder reads B and starts work
        s.conn
            .execute(
                "UPDATE documents SET body = ?1, content_sha256 = ?2 WHERE id = ?3",
                params!["body B", sha256_hex(b"body B"), id],
            )
            .unwrap();
        // 4. ingest restores body A - both hashes are A again, so the row looks untouched
        s.conn
            .execute(
                "UPDATE documents SET body = ?1, content_sha256 = ?2 WHERE id = ?3",
                params!["body A", &hash_a, id],
            )
            .unwrap();

        // 5. the embedder returns with B's vectors
        let bad = ChunkPlacement { heading_path: "", char_start: 0, char_end: 6 };
        let stamped = s
            .replace_chunk_embeddings(id, 2, &[(vec![0xBB; 8], bad)], &sha256_hex(b"body B"))
            .unwrap();
        assert!(!stamped, "vectors for body B must not be certified against body A");

        let vec_bytes: Vec<u8> = s
            .conn
            .query_row(
                "SELECT vec FROM bge_chunk_embeddings WHERE doc_id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            vec_bytes,
            vec![0xAA; 8],
            "body A's CORRECT vectors must survive - nothing would ever repair them"
        );
        let embedded: Option<String> = s
            .conn
            .query_row(
                "SELECT embedded_sha256 FROM documents WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(embedded.as_deref(), Some(hash_a.as_str()));
        assert!(
            s.docs_missing_chunk_embeddings(10).unwrap().is_empty(),
            "and the document is genuinely fresh, because its vectors really are body A's"
        );
    }

    #[test]
    fn a_legacy_row_with_no_content_hash_still_drains() {
        // Every row is in this state on a database that predates the hash columns and has not
        // been backfilled. Stamping `embedded_sha256 = content_sha256` there writes NULL, the
        // document stays queued, and `recall embed` loops on it forever. The write must heal
        // the missing content hash instead.
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = s.insert("a.md", "T", "legacy body", "test").unwrap();
        s.conn
            .execute("UPDATE documents SET content_sha256 = NULL WHERE id = ?1", params![id])
            .unwrap();
        assert_eq!(s.docs_missing_chunk_embeddings(10).unwrap().len(), 1);

        let place = ChunkPlacement { heading_path: "", char_start: 0, char_end: 4 };
        let stamped = s
            .replace_chunk_embeddings(id, 2, &[(vec![0u8; 8], place)], &sha256_hex(b"legacy body"))
            .unwrap();
        assert!(stamped, "a legacy row must be certifiable, not skipped forever");
        assert!(
            s.docs_missing_chunk_embeddings(10).unwrap().is_empty(),
            "the queue MUST drain - otherwise embed never terminates"
        );
    }

    #[test]
    fn a_failed_chunk_write_rolls_back_the_delete() {
        // Atomicity asserted by FORCING a failure, not by checking a happy path - a version
        // built from three autocommitted statements passes the happy path too. A trigger aborts
        // the insert half way; the document's ORIGINAL vectors must still be there afterwards.
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = s.insert("a.md", "T", "body", "test").unwrap();
        put_chunk(&s, id, 0, &[7u8; 8]);
        put_chunk(&s, id, 1, &[7u8; 8]);
        s.conn
            .execute(
                "UPDATE documents SET embedded_sha256 = ?1 WHERE id = ?2",
                params![sha256_hex(b"body"), id],
            )
            .unwrap();

        s.conn
            .execute_batch(
                "CREATE TRIGGER boom BEFORE INSERT ON bge_chunk_embeddings
                 WHEN NEW.heading_path = 'BOOM'
                 BEGIN SELECT RAISE(ABORT, 'boom'); END;",
            )
            .unwrap();

        let good = ChunkPlacement { heading_path: "", char_start: 0, char_end: 4 };
        let bad = ChunkPlacement { heading_path: "BOOM", char_start: 0, char_end: 4 };
        let err = s.replace_chunk_embeddings(
            id,
            2,
            &[(vec![9u8; 8], good), (vec![9u8; 8], bad)],
            &sha256_hex(b"body"),
        );
        assert!(err.is_err(), "the aborting insert must surface as an error");

        s.conn.execute_batch("DROP TRIGGER boom;").unwrap();

        let rows: Vec<Vec<u8>> = {
            let mut st = s
                .conn
                .prepare("SELECT vec FROM bge_chunk_embeddings WHERE doc_id = ?1 ORDER BY chunk_index")
                .unwrap();
            st.query_map(params![id], |r| r.get(0))
                .unwrap()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap()
        };
        assert_eq!(rows.len(), 2, "the DELETE must have rolled back with the failed INSERT");
        assert!(
            rows.iter().all(|v| v == &vec![7u8; 8]),
            "the ORIGINAL vectors must survive, not the half-written new ones"
        );
    }

    #[test]
    fn chunk_rows_without_a_stamp_do_not_drain_the_queue() {
        // The converse of the test above, stated as its own assertion so the two cannot be
        // confused: chunk rows ALONE are no longer enough. This is the exact state every
        // document in the live corpus is in right after the hash columns are added, and it is
        // the state a crash between "vectors written" and "stamp written" would produce.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = store.insert("a.md", "A", "body", "cli").unwrap();
        store.backfill_content_hashes(10).unwrap();
        put_chunk(&store, id, 0, &[0u8; 8]); // vectors, but no stamp

        let work = store.docs_missing_chunk_embeddings(10).unwrap();
        assert_eq!(
            work.len(),
            1,
            "vectors with no embedded_sha256 must still be queued"
        );
    }

    #[test]
    fn chunk_embeddings_cascade_away_with_their_document() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = store.insert("a.md", "A", "body", "cli").unwrap();
        put_chunk(&store, id, 0, &[0u8; 8]);
        store
            .conn
            .execute("DELETE FROM documents WHERE id = ?1", params![id])
            .unwrap();
        let rows: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM bge_chunk_embeddings", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(rows, 0, "FK ON DELETE CASCADE must reach the chunk table");
    }

    #[test]
    fn load_all_chunk_embeddings_excludes_mirror_copy_paths() {
        // Same query-time exclusion the pooled loader applies. Without it the top hits were
        // three copies of one mirrored SKILL.md — and a chunk table multiplies each
        // duplicate by its chunk count.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let keep = store
            .insert("projects/keep.md", "Keep", "body", "vault")
            .unwrap();
        let mirror = store
            .insert(
                "tools/state/skill-radar/disabled-loader-roots-2026-05-09-active-finder-only/skill.md",
                "Mirror",
                "body",
                "vault",
            )
            .unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        put_chunk(&store, keep, 0, &dummy);
        put_chunk(&store, keep, 1, &dummy);
        put_chunk(&store, mirror, 0, &dummy);

        let loaded = store.load_all_chunk_embeddings().unwrap();
        assert_eq!(
            loaded.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![keep, keep],
            "BGE chunk matrix must skip mirror-copy vectors and keep every live chunk"
        );
    }

    // ── GC unit tests ─────────────────────────────────────────────────────────

    #[test]
    fn gc_count_orphaned_embeddings_zero_when_all_live() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id1 = store.insert("a.md", "A", "body a", "cli").unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(id1, &dummy, 2).unwrap();

        // No tombstones yet — count must be 0.
        assert_eq!(store.count_orphaned_embeddings().unwrap(), 0);
    }

    #[test]
    fn hard_delete_cascades_embedding_row_away() {
        // Regression: embeddings.doc_id has ON DELETE CASCADE, but SQLite
        // defaults FK enforcement OFF. init_schema must enable it, otherwise a
        // hard DELETE of the parent document (ingest_dir re-index by path, and
        // reset()) leaves a parentless ghost embedding that bloats the db on
        // every re-index. count_embeddings() is JOIN-based and HIDES such
        // ghosts, so this test counts raw embedding rows to see the leak.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id1 = store.insert("a.md", "A", "body a", "cli").unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(id1, &dummy, 2).unwrap();

        let raw_before: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(raw_before, 1);

        // Hard-delete the parent document (same DELETE statement ingest_dir
        // and reset() issue).
        store.reset().unwrap();

        let raw_after: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            raw_after, 0,
            "ON DELETE CASCADE must physically remove the orphaned embedding"
        );
    }

    #[test]
    fn gc_count_orphaned_embeddings_detects_tombstone() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id1 = store.insert("a.md", "A", "body a", "cli").unwrap();
        let id2 = store.insert("b.md", "B", "body b", "cli").unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(id1, &dummy, 2).unwrap();
        store.upsert_embedding(id2, &dummy, 2).unwrap();

        store.supersede_id(id1).unwrap(); // tombstone doc 1

        assert_eq!(store.count_orphaned_embeddings().unwrap(), 1);
    }

    #[test]
    fn gc_apply_removes_only_orphaned_embeddings() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id1 = store.insert("a.md", "A", "body a", "cli").unwrap();
        let id2 = store.insert("b.md", "B", "body b", "cli").unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(id1, &dummy, 2).unwrap();
        store.upsert_embedding(id2, &dummy, 2).unwrap();

        store.supersede_id(id1).unwrap(); // tombstone doc 1

        let deleted = store.delete_orphaned_embeddings().unwrap();
        assert_eq!(deleted, 1, "exactly one orphaned embedding removed");

        // Live embedding for id2 must still exist.
        assert_eq!(store.count_embeddings().unwrap(), 1);
        // Live doc count unchanged.
        assert_eq!(store.count_valid().unwrap(), 1);
    }

    #[test]
    fn gc_apply_never_deletes_live_doc_embeddings() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id1 = store.insert("a.md", "A", "body a", "cli").unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(id1, &dummy, 2).unwrap();

        // No tombstones — apply must delete nothing.
        let deleted = store.delete_orphaned_embeddings().unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(store.count_embeddings().unwrap(), 1);
    }

    #[test]
    fn gc_apply_is_idempotent() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id1 = store.insert("a.md", "A", "body a", "cli").unwrap();
        let id2 = store.insert("b.md", "B", "body b", "cli").unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(id1, &dummy, 2).unwrap();
        store.upsert_embedding(id2, &dummy, 2).unwrap();
        store.supersede_id(id1).unwrap();

        let first = store.delete_orphaned_embeddings().unwrap();
        assert_eq!(first, 1);

        // Second run: nothing left to delete.
        let second = store.delete_orphaned_embeddings().unwrap();
        assert_eq!(second, 0, "idempotent: second gc run deletes 0 rows");
    }

    #[test]
    fn gc_dry_run_count_does_not_mutate() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id1 = store.insert("a.md", "A", "body a", "cli").unwrap();
        let id2 = store.insert("b.md", "B", "body b", "cli").unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(id1, &dummy, 2).unwrap();
        store.upsert_embedding(id2, &dummy, 2).unwrap();
        store.supersede_id(id1).unwrap();

        // Dry-run: only count, do not delete.
        let orphaned = store.count_orphaned_embeddings().unwrap();
        assert_eq!(orphaned, 1);

        // Total embeddings table still has 2 rows (live + orphaned).
        let total: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(total, 2, "dry-run must not delete any rows");
    }

    #[test]
    fn count_embeddings_excludes_superseded_docs() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();

        // Insert two docs and give each an embedding.
        let id1 = store.insert("a.md", "A", "body a", "cli").unwrap();
        let id2 = store.insert("b.md", "B", "body b", "cli").unwrap();
        let dummy_vec: Vec<u8> = vec![0u8; 8]; // 2 × f32, dim doesn't matter for counting
        store.upsert_embedding(id1, &dummy_vec, 2).unwrap();
        store.upsert_embedding(id2, &dummy_vec, 2).unwrap();

        // Both docs valid → count must be 2.
        assert_eq!(store.count_embeddings().unwrap(), 2);

        // Supersede doc 1 (sets valid_to; row is kept — bi-temporal).
        store.supersede_id(id1).unwrap();

        // Now only one valid doc remains; count_embeddings must reflect that.
        assert_eq!(store.count_embeddings().unwrap(), 1);
    }

    // ── 4 additive edge-case tests ────────────────────────────────────────────

    /// The keyword channel must honour the same path exclusions as the vector channels.
    ///
    /// Regression: it did not, so the exclusion silently depended on a vector channel being
    /// live. With Gemini switched off, an excluded scratch copy of a profile for a DIFFERENT,
    /// closed company with the same name returned to rank 1. A filter that only works while an
    /// optional channel is up is not a filter.
    #[test]
    fn fts_search_excludes_the_same_paths_as_the_vector_channels() {
        // SAFETY: single-threaded test; sets the documented runtime exclude knob.
        std::env::set_var("RECALL_EXCLUDE_EXTRA", "scratch,zip-digest");
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store
            .insert(
                "scratch/acme-zip-digest/research_notes.md",
                "Acme Research Notes",
                "Acme company profile: Location London England, Permanently Closed",
                "vault",
            )
            .unwrap();
        store
            .insert(
                "research/acme-inc-overview.md",
                "Acme Incorporated",
                "Acme Inc is a Delaware corporation",
                "vault",
            )
            .unwrap();

        let hits = store.fts_search("Acme", 10).unwrap();
        assert!(
            !hits.iter().any(|h| h.path.contains("zip-digest")),
            "scratch/tool-dump paths must never reach keyword results: {:?}",
            hits.iter().map(|h| &h.path).collect::<Vec<_>>()
        );
        assert!(
            hits.iter().any(|h| h.path.starts_with("research/")),
            "the real document must still be returned"
        );
        std::env::remove_var("RECALL_EXCLUDE_EXTRA");
    }

    // ── tier gate ─────────────────────────────────────────────────────────────

    /// The agent skill library must not compete with the owner's own notes in the default index.
    ///
    /// Regression, measured 2026-08-04: `skills/db-backup/SKILL.md` came back at **rank 1**
    /// for "backup and recovery", ahead of every note he had written, and
    /// `skills/memory-search/SKILL.md` at rank 2 for "what did we decide about the
    /// reranker". 9,681 densely-keyworded how-to files against 25,229 notes will win any
    /// generic phrasing on keyword score alone. Skills moved to their own index; this test
    /// pins the gate that keeps them out of this one.
    #[test]
    fn fts_search_default_tier_gate_keeps_skills_out_of_the_brain_index() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let skill = store
            .insert(
                "skills/db-backup/SKILL.md",
                "db-backup",
                "How to run backup and recovery for a database",
                "vault",
            )
            .unwrap();
        let note = store
            .insert(
                "projects/notes/backups.md",
                "Backup decisions",
                "Our backup and recovery policy: B2 over the S3 endpoint",
                "vault",
            )
            .unwrap();
        set_tier(&store, skill, "SKILLS");
        set_tier(&store, note, "BRAIN");

        let hits = store
            .fts_search_with_tiers("backup recovery", 10, &TierFilter::load())
            .unwrap();
        assert!(
            !hits.iter().any(|h| h.path.starts_with("skills/")),
            "skill docs must not reach the default index: {:?}",
            hits.iter().map(|h| &h.path).collect::<Vec<_>>()
        );
        assert!(
            hits.iter().any(|h| h.id == note),
            "the user's own note must still be returned"
        );

        // ...and asking for them explicitly must still work, or the skills index would be
        // unreachable through recall entirely.
        let all = store
            .fts_search_with_tiers("backup recovery", 10, &TierFilter::from_spec("all"))
            .unwrap();
        assert!(all.iter().any(|h| h.id == skill));
    }

    /// A document with no tier yet must stay searchable. `tier_for_path` classifies most real
    /// vault paths automatically now, but a path matching no rule at all still lands NULL —
    /// and a gate that hid NULL rows would make an unclassifiable note silently unfindable,
    /// which is far worse than a skill ranking too high.
    #[test]
    fn tier_gate_fails_open_on_untagged_documents() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let fresh = store
            .insert(
                "some-new-top-level-dir/just-written.md",
                "Just written",
                "quokka telemetry decision",
                "vault",
            )
            .unwrap();
        // A pre-tagger document (or one a private rule list left unmatched) has tier NULL.
        store.conn.execute("UPDATE documents SET tier = NULL WHERE id = ?1", params![fresh]).unwrap();

        let hits = store
            .fts_search_with_tiers("quokka", 10, &TierFilter::load())
            .unwrap();
        assert!(
            hits.iter().any(|h| h.id == fresh),
            "an untagged document must remain searchable"
        );
    }

    // ── tier_for_path: automatic tiering at ingest (mirrors apply_tiers.py::tier_of) ─────

    #[test]
    fn tier_for_path_empty_is_brain() {
        // provenance note 6: `recall save` rows have no vault path.
        assert_eq!(tier_for_path(""), Some("BRAIN"));
        assert_eq!(tier_for_path("   "), Some("BRAIN"));
    }

    #[test]
    fn public_default_tiers_are_layout_independent() {
        assert_eq!(tier_for_path("skills/optimization-suite/ln-810/SKILL.md"), Some("SKILLS"));
        assert_eq!(tier_for_path(".claude/skills/foo/SKILL.md"), Some("SKILLS"));
        assert_eq!(tier_for_path(".agents/skills/bar/SKILL.md"), Some("SKILLS"));
        assert_eq!(tier_for_path("some/pkg/node_modules/left-pad/index.js"), Some("VENDOR"));
        assert_eq!(tier_for_path("some/proj/target/debug/build.log"), Some("VENDOR"));
        assert_eq!(tier_for_path("tools/state/disabled-loader-roots-2026-05-09/skill.md"), Some("VENDOR"));
        assert_eq!(tier_for_path("some/repo/worktrees/branch-x/file.md"), Some("VENDOR"));
        // case-insensitive, like the rules always were
        assert_eq!(tier_for_path("SKILLS/Foo/SKILL.MD"), Some("SKILLS"));
        // no vault layout is assumed: everything else is the user's own writing
        assert_eq!(tier_for_path("projects/a/plan.md"), Some("BRAIN"));
        assert_eq!(tier_for_path("loose-note-at-root.md"), Some("BRAIN"));
    }

    /// Every match kind of the private rules file, on a synthetic layout, in file order.
    #[test]
    fn private_tier_rules_cover_every_match_kind_first_match_wins() {
        let text = r#"{"tiers": [
            {"tier": "SKILLS", "match": "prefix", "value": "Agents/"},
            {"tier": "VENDOR", "match": "subdir_after", "value": "lib/repos/"},
            {"tier": "VENDOR", "match": "raw_subdir_after", "value": "inbox/drops/"},
            {"tier": "VENDOR", "match": "contains", "value": "/mirror/"},
            {"tier": "BRAIN", "match": "numbered_folder"},
            {"tier": "BRAIN", "match": "numbered_root_file"},
            {"tier": "BRAIN", "match": "prefix", "value": "lib/"},
            {"tier": "VENDOR", "match": "root_note_file"}
        ]}"#;
        let rules = parse_tier_rules(text).unwrap().unwrap();
        let t = |p: &str| tier_by_rules(&p.to_lowercase(), &rules);
        assert_eq!(t("agents/x/SKILL.md"), Some("SKILLS"), "values are lower-cased");
        assert_eq!(t("lib/repos/someclone/README.md"), Some("VENDOR"), "a subdirectory under repos/");
        assert_eq!(t("lib/repos/notes-about-repos.md"), Some("BRAIN"), "a file directly in repos/ is not a clone");
        assert_eq!(t("inbox/drops/one/raw/file.txt"), Some("VENDOR"));
        assert_eq!(t("inbox/drops/one/digest.md"), None, "the digest beside raw/ is not the drop");
        assert_eq!(t("a/b/mirror/c.md"), Some("VENDOR"));
        assert_eq!(t("12-area/note.md"), Some("BRAIN"));
        assert_eq!(t("12-area.md"), Some("BRAIN"));
        assert_eq!(t("loose.qmd"), Some("VENDOR"), "root_note_file, told apart from numbered_root_file");
        assert_eq!(t("other/dir/note.md"), None);
        // first match wins: a mirror copy inside a numbered folder is VENDOR, not BRAIN
        assert_eq!(t("12-area/mirror/x.md"), Some("VENDOR"), "contains is listed before numbered_folder");
    }

    #[test]
    fn private_paths_are_optional_vault_relative_and_validated() {
        let p = parse_private_settings(r#"{"recall_dev_dir": "dev/recall/", "bench_fixtures": "data\\fx.jsonl"}"#).unwrap();
        assert_eq!(p.recall_dev_dir.as_deref(), Some("dev/recall"));
        assert_eq!(p.bench_fixtures.as_deref(), Some("data/fx.jsonl"));
        let n = parse_private_settings(r#"{"recall_dev_dir": "./dev//recall/."}"#).unwrap();
        assert_eq!(n.recall_dev_dir.as_deref(), Some("dev/recall"), "normalised, so exclusion prefixes match");
        assert_eq!(parse_private_settings("{}").unwrap(), PrivateSettings::default());
        let q = parse_private_settings(r#"{"eval_queries": ["one topic", "two"]}"#).unwrap();
        assert_eq!(q.eval_queries, vec!["one topic".to_string(), "two".to_string()]);
        for bad in [r#"{"eval_queries": ["ok", ""]}"#, r#"{"eval_queries": "x"}"#, r#"{"recall_dev_dir": "../x"}"#, r#"{"recall_dev_dir": "C:/x"}"#, r#"{"bench_fixtures": 3}"#, r#"{"recall_dev_dir": ""}"#] {
            assert!(parse_private_settings(bad).is_err(), "{bad}");
        }
        // public default: no dev dir, so only the generic eval/results/ prefix is kept out of results
        let ex = PathExcludeList::with_extra(&[]);
        assert!(is_query_excluded_path(&ex, "eval/results/run.json"));
        assert!(!is_query_excluded_path(&ex, "dev/recall/eval/results/run.json"));
    }

    #[test]
    fn a_bad_private_rules_file_is_an_error_never_a_skipped_rule() {
        assert!(parse_tier_rules(r#"{"tiers": [{"tier": "MINE", "match": "prefix", "value": "a/"}]}"#).is_err());
        assert!(parse_tier_rules(r#"{"tiers": [{"tier": "BRAIN", "match": "glob", "value": "a/"}]}"#).is_err());
        assert!(parse_tier_rules(r#"{"tiers": [{"tier": "BRAIN", "match": "prefix"}]}"#).is_err());
        assert!(parse_tier_rules("not json").is_err());
        assert_eq!(parse_tier_rules(r#"{"other": 1}"#).unwrap(), None, "no tiers key = public defaults");
    }

    /// A path matching no rule stays `None` — never a made-up "UNCLASSIFIED" literal, which
    /// would fall outside `KNOWN_TIERS` and be deleted from every search result by the gate.
    #[test]
    fn tier_for_path_unmatched_is_none_not_a_guess() {
        let rules = parse_tier_rules(r#"{"tiers": [{"tier": "SKILLS", "match": "prefix", "value": "skills/"}]}"#)
            .unwrap()
            .unwrap();
        let computed = tier_by_rules("some-unknown-top-level-dir/notes.md", &rules);
        assert_eq!(computed, None, "an unmatched path must be None, never a guessed tier");
    }

    // ── automatic tagging at the real INSERT sites ──────────────────────────────

    #[test]
    fn store_insert_tags_tier_automatically() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let id = store
            .insert("skills/foo/SKILL.md", "Foo", "body", "vault")
            .unwrap();
        let tier: Option<String> = store
            .conn
            .query_row("SELECT tier FROM documents WHERE id = ?1", params![id], |r| r.get(0))
            .unwrap();
        assert_eq!(tier.as_deref(), Some("SKILLS"));
    }

    #[test]
    fn ingest_dir_tags_new_documents_automatically() {
        let dir = std::env::temp_dir().join(format!(
            "recall-autotier-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("skills/foo")).unwrap();
        std::fs::write(dir.join("skills/foo/SKILL.md"), "# Note\nbody").unwrap();

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store.ingest_dir(&dir).unwrap();

        let tier: Option<String> = store
            .conn
            .query_row(
                "SELECT tier FROM documents WHERE path = 'skills/foo/SKILL.md'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            tier.as_deref(),
            Some("SKILLS"),
            "a document ingested for the first time must be tagged immediately, not left NULL"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 🔴 The one rule that matters most: a tier the owner corrected by hand must survive a
    /// re-index. `ingest_dir`'s UPDATE branch (existing path, changed body) must never touch
    /// `tier` — only the INSERT branch (brand-new row) is allowed to set it.
    #[test]
    fn ingest_dir_never_overwrites_an_existing_tier_on_update() {
        let dir = std::env::temp_dir().join(format!(
            "recall-tierpreserve-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // The auto-computed tier is BRAIN (the default catch-all), so the hand correction to
        // VENDOR below is unambiguous: VENDOR was never the computed value.
        std::fs::create_dir_all(dir.join("some-unknown-dir")).unwrap();
        std::fs::write(dir.join("some-unknown-dir/file.md"), "# F\noriginal body").unwrap();

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store.ingest_dir(&dir).unwrap();
        let id: i64 = store
            .conn
            .query_row(
                "SELECT id FROM documents WHERE path = 'some-unknown-dir/file.md'",
                [],
                |r| r.get(0),
            )
            .unwrap();

        // Simulate the owner correcting the tier by hand.
        set_tier(&store, id, "VENDOR");

        // Change the body and re-index — this hits the UPDATE branch, not INSERT.
        std::fs::write(dir.join("some-unknown-dir/file.md"), "# F\nedited body").unwrap();
        store.ingest_dir(&dir).unwrap();

        let (tier, body): (Option<String>, String) = store
            .conn
            .query_row(
                "SELECT tier, body FROM documents WHERE id = ?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(body, "# F\nedited body", "the edit must still land");
        assert_eq!(
            tier.as_deref(),
            Some("VENDOR"),
            "a hand-corrected tier must survive a re-index of the same document"
        );
    }

    #[test]
    fn tier_filter_default_is_brain_and_vendor() {
        let clause = TierFilter::load().sql_clause("d");
        assert!(clause.contains("'BRAIN'"), "{clause}");
        assert!(clause.contains("'VENDOR'"), "{clause}");
        assert!(!clause.contains("'SKILLS'"), "{clause}");
        assert!(
            clause.contains("d.tier IS NULL"),
            "must fail open on untagged rows: {clause}"
        );
    }

    #[test]
    fn tier_filter_all_emits_no_clause() {
        assert_eq!(TierFilter::from_spec("all").sql_clause("d"), "");
        assert_eq!(TierFilter::from_spec("  ").sql_clause("d"), "");
    }

    /// A typo must not silently empty the index. `RECALL_TIERS=brian` should behave like the
    /// default, not like "match nothing".
    #[test]
    fn tier_filter_unknown_names_fall_back_to_the_default() {
        let clause = TierFilter::from_spec("brian,nonsense").sql_clause("d");
        assert!(clause.contains("'BRAIN'") && clause.contains("'VENDOR'"), "{clause}");
    }

    /// Caller text is matched against a closed constant list and never interpolated, so a
    /// hostile `RECALL_TIERS` cannot reach SQL.
    #[test]
    fn tier_filter_never_interpolates_caller_text() {
        let clause = TierFilter::from_spec("brain','x') OR 1=1 --").sql_clause("d");
        assert!(!clause.contains("1=1"), "{clause}");
        assert!(!clause.contains("--"), "{clause}");
        // Nothing in that string matches a known tier ("brain'" is not "BRAIN"), so it
        // falls back to the default rather than emitting a match-nothing clause.
        assert!(clause.contains("'BRAIN'"), "{clause}");
    }

    /// Opening a database written before `tier` existed must migrate it, not crash.
    ///
    /// This is the restored-B2-snapshot case. `CREATE TABLE IF NOT EXISTS` does nothing to an
    /// existing file, so without the ALTER every query against a restored backup would die on
    /// `no such column: d.tier` — the store would look totally destroyed when it is fine.
    #[test]
    fn opening_a_pre_tier_database_migrates_it_instead_of_failing() {
        let path = std::env::temp_dir().join(format!(
            "recall-pretier-{}-{:?}.db",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        // Write the OLD shape: documents without a tier column.
        let old = rusqlite::Connection::open(&path).unwrap();
        old.execute_batch(
            "CREATE TABLE documents (
                 id INTEGER PRIMARY KEY, path TEXT NOT NULL DEFAULT '',
                 title TEXT NOT NULL DEFAULT '', body TEXT NOT NULL,
                 source TEXT NOT NULL DEFAULT 'cli', created_at TEXT NOT NULL,
                 valid_from TEXT NOT NULL, valid_to TEXT, superseded_by INTEGER);",
        )
        .unwrap();
        drop(old);

        let store = Store::open_at(&path).unwrap();
        let id = store
            .insert("notes/x.md", "X", "wombat provisioning", "vault")
            .unwrap();
        let hits = store
            .fts_search_with_tiers("wombat", 5, &TierFilter::load())
            .unwrap();
        assert!(
            hits.iter().any(|h| h.id == id),
            "a migrated pre-tier database must still be searchable"
        );
        drop(store);
        let _ = std::fs::remove_file(&path);
    }

    /// Same restored-B2-snapshot case as the `tier` test above, for `authored_on`.
    ///
    /// A database written before `authored_on` existed — which includes every snapshot in
    /// the restored backups — must migrate on open, stay searchable, and end up
    /// with a NULL-valued column that the backfill can write into. A pre-`tier` database is
    /// used as the starting shape so this also proves the two migrations compose: opening one
    /// old file adds BOTH columns in a single `init_schema` pass.
    #[test]
    fn opening_a_pre_authored_on_database_migrates_it_instead_of_failing() {
        let path = std::env::temp_dir().join(format!(
            "recall-preauthored-{}-{:?}.db",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        // The OLD shape: no tier column and no authored_on column.
        let old = rusqlite::Connection::open(&path).unwrap();
        old.execute_batch(
            "CREATE TABLE documents (
                 id INTEGER PRIMARY KEY, path TEXT NOT NULL DEFAULT '',
                 title TEXT NOT NULL DEFAULT '', body TEXT NOT NULL,
                 source TEXT NOT NULL DEFAULT 'cli', created_at TEXT NOT NULL,
                 valid_from TEXT NOT NULL, valid_to TEXT, superseded_by INTEGER);",
        )
        .unwrap();
        drop(old);

        let store = Store::open_at(&path).unwrap();
        let id = store
            .insert("notes/x.md", "X", "wombat provisioning", "vault")
            .unwrap();

        // The column exists and defaults to NULL rather than to a fabricated date.
        let authored: Option<String> = store
            .conn
            .query_row(
                "SELECT authored_on FROM documents WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            authored.is_none(),
            "a freshly inserted document must have no authored_on, got {authored:?}"
        );

        // And the backfill's exact write shape works against the migrated file.
        store
            .conn
            .execute(
                "UPDATE documents SET authored_on = ?1 WHERE id = ?2",
                params!["2026-05-08", id],
            )
            .unwrap();
        let authored: Option<String> = store
            .conn
            .query_row(
                "SELECT authored_on FROM documents WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(authored.as_deref(), Some("2026-05-08"));

        // Migrating must not cost searchability — this is the whole point of the ALTER.
        let hits = store
            .fts_search_with_tiers("wombat", 5, &TierFilter::load())
            .unwrap();
        assert!(
            hits.iter().any(|h| h.id == id),
            "a migrated pre-authored_on database must still be searchable"
        );

        // Opening the SAME file again must be a no-op, not a duplicate-column error.
        drop(store);
        let reopened = Store::open_at(&path).unwrap();
        drop(reopened);

        let _ = std::fs::remove_file(&path);
    }

    /// The exclude list must be readable from a file, not only from the environment, so the
    /// search policy is visible in the repository instead of buried in a launcher script.
    #[test]
    fn exclude_patterns_load_from_the_config_file_not_only_the_environment() {
        let dir = std::env::temp_dir().join(format!("recall-excl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("exclude.txt");
        std::fs::write(&f, "# scratch dumps
scratch

raw-intake   # raw intake
").unwrap();

        let pats = PathExcludeList::patterns_from_file(&f);
        assert!(pats.contains(&"scratch".to_string()), "{pats:?}");
        assert!(pats.contains(&"raw-intake".to_string()), "{pats:?}");
        assert_eq!(pats.len(), 2, "comments and blanks must be stripped: {pats:?}");

        let list = PathExcludeList::with_extra(&["scratch", "raw-intake"]);
        assert!(list.is_excluded("scratch/acme-zip-digest/notes.md"));
        assert!(list.is_excluded("tasks/raw-intake-2026-01-01/x.md"));
        assert!(!list.is_excluded("tasks/acme-audit.md"));

        let _ = std::fs::remove_file(&f);
        let _ = std::fs::remove_dir(&dir);
    }

    fn set_tier(store: &Store, id: i64, tier: &str) {
        store
            .conn
            .execute("UPDATE documents SET tier = ?1 WHERE id = ?2", params![tier, id])
            .unwrap();
    }

    #[test]
    fn load_all_embeddings_excludes_mirror_copy_paths() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let keep = store
            .insert("projects/keep.md", "Keep", "body", "vault")
            .unwrap();
        let mirror = store
            .insert(
                "tools/state/skill-radar/disabled-loader-roots-2026-05-09-active-finder-only/skill.md",
                "Mirror",
                "body",
                "vault",
            )
            .unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(keep, &dummy, 2).unwrap();
        store.upsert_embedding(mirror, &dummy, 2).unwrap();

        assert_eq!(
            store.count_embeddings().unwrap(),
            2,
            "coverage accounting still reports both valid embedded docs"
        );
        let loaded = store.load_all_embeddings().unwrap();
        assert_eq!(
            loaded.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![keep],
            "BGE query matrix must skip mirror-copy vectors without deleting rows"
        );
    }

    #[test]
    fn load_all_embeddings_excludes_generated_eval_result_paths_only_from_query_matrix() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let keep = store
            .insert("projects/keep.md", "Keep", "body", "vault")
            .unwrap();
        let generated = store
            .insert(
                "eval/results/secret_surrogates/surrogate.md",
                "Generated Eval Artifact",
                "body",
                "vault",
            )
            .unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(keep, &dummy, 2).unwrap();
        store.upsert_embedding(generated, &dummy, 2).unwrap();

        assert_eq!(
            store.count_embeddings().unwrap(),
            2,
            "coverage accounting still reports generated artifacts as embedded docs"
        );
        assert_eq!(
            store.count_exclude_policy_embeddings().unwrap(),
            1,
            "query-exclude dry-run count should detect generated eval artifacts"
        );
        let loaded = store.load_all_embeddings().unwrap();
        assert_eq!(
            loaded.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![keep],
            "BGE query matrix must skip generated eval artifacts without deleting rows"
        );
    }

    #[test]
    fn gc_exclude_policy_count_detects_live_excluded_embeddings() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let keep = store
            .insert("projects/keep.md", "Keep", "body", "vault")
            .unwrap();
        let mirror = store
            .insert(
                "tools/state/skill-radar/disabled-loader-roots-2026-05-09-active-finder-only/skill.md",
                "Mirror",
                "body",
                "vault",
            )
            .unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(keep, &dummy, 2).unwrap();
        store.upsert_embedding(mirror, &dummy, 2).unwrap();

        assert_eq!(store.count_embeddings().unwrap(), 2);
        assert_eq!(store.count_exclude_policy_embeddings().unwrap(), 1);
    }

    #[test]
    fn gc_exclude_policy_count_is_dry_run_only() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let mirror = store
            .insert(
                "tools/state/skill-radar/disabled-loader-roots-2026-05-09-active-finder-only/skill.md",
                "Mirror",
                "body",
                "vault",
            )
            .unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(mirror, &dummy, 2).unwrap();

        let before: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(store.count_exclude_policy_embeddings().unwrap(), 1);
        let after: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            after, before,
            "exclude-aware dry-run count must not delete rows"
        );
    }

    #[test]
    fn gc_empty_store_is_noop() {
        // GC on a store with no documents must report 0 orphans, delete 0 rows,
        // and never panic.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        assert_eq!(store.count_orphaned_embeddings().unwrap(), 0);
        let deleted = store.delete_orphaned_embeddings().unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(store.count_embeddings().unwrap(), 0);
        assert_eq!(store.count_valid().unwrap(), 0);
    }

    #[test]
    fn gc_preserves_live_doc_sharing_body_with_superseded() {
        // Two docs with identical body content. Superseding one must orphan
        // exactly one embedding; the surviving doc's embedding must remain.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let shared_body = "identical body content for both docs";
        let id1 = store
            .insert("first.md", "First", shared_body, "cli")
            .unwrap();
        let id2 = store
            .insert("second.md", "Second", shared_body, "cli")
            .unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(id1, &dummy, 2).unwrap();
        store.upsert_embedding(id2, &dummy, 2).unwrap();

        // Tombstone only the first doc.
        store.supersede_id(id1).unwrap();
        assert_eq!(store.count_orphaned_embeddings().unwrap(), 1);

        let deleted = store.delete_orphaned_embeddings().unwrap();
        assert_eq!(deleted, 1, "exactly one embedding must be removed");

        // The live doc (id2) must still have its embedding.
        assert_eq!(
            store.count_embeddings().unwrap(),
            1,
            "surviving doc embedding must be intact"
        );
        // Verify it's specifically id2's embedding that persists.
        let live_emb: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM embeddings WHERE doc_id = ?1",
                params![id2],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(live_emb, 1, "id2 embedding must survive gc");
    }

    #[test]
    fn gc_collects_embedding_of_forget_tombstoned_doc() {
        // `forget` tombstones docs by pattern match (bi-temporal). GC must collect
        // their embeddings just as it does for supersede_id tombstones.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        // source must NOT be 'vault' for forget() to fire (see forget() impl).
        let id_forgotten = store
            .insert("obs.md", "Obs", "some observation to forget", "cli")
            .unwrap();
        let id_live = store
            .insert("kept.md", "Kept", "this one stays live", "cli")
            .unwrap();
        let dummy: Vec<u8> = vec![0u8; 8];
        store.upsert_embedding(id_forgotten, &dummy, 2).unwrap();
        store.upsert_embedding(id_live, &dummy, 2).unwrap();

        // forget() matches body substring — it sets valid_to on matching rows.
        let retired = store.forget("observation to forget").unwrap();
        assert_eq!(retired, 1, "forget must retire exactly one doc");

        assert_eq!(
            store.count_orphaned_embeddings().unwrap(),
            1,
            "forget-tombstoned doc's embedding must show as orphaned"
        );

        let deleted = store.delete_orphaned_embeddings().unwrap();
        assert_eq!(
            deleted, 1,
            "gc must collect the forget-tombstoned embedding"
        );

        // The kept doc's embedding must remain.
        assert_eq!(store.count_embeddings().unwrap(), 1);
    }

    #[test]
    fn gc_dry_run_multiple_orphans_does_not_mutate_embeddings_table() {
        // With multiple orphaned embeddings, count_orphaned_embeddings() must NOT
        // remove any rows — raw COUNT(*) on the embeddings table must be unchanged.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let ids: Vec<i64> = (0..4)
            .map(|i| {
                store
                    .insert(
                        &format!("doc{i}.md"),
                        &format!("Doc {i}"),
                        &format!("body {i}"),
                        "cli",
                    )
                    .unwrap()
            })
            .collect();

        let dummy: Vec<u8> = vec![0u8; 8];
        for &id in &ids {
            store.upsert_embedding(id, &dummy, 2).unwrap();
        }

        // Tombstone the first 3 docs, leaving id[3] live.
        for &id in &ids[..3] {
            store.supersede_id(id).unwrap();
        }

        // Dry-run: count without deleting.
        let orphaned = store.count_orphaned_embeddings().unwrap();
        assert_eq!(orphaned, 3, "three orphaned embeddings reported");

        // Raw total in the embeddings table must still be 4.
        let total: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            total, 4,
            "dry-run count must not mutate the embeddings table"
        );
    }

    #[test]
    fn coverage_pct_is_100_when_all_docs_have_embeddings() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let dummy: Vec<u8> = vec![0u8; 8]; // 2x f32 LE placeholder

        let id1 = store.insert("a.md", "Title A", "body a", "vault").unwrap();
        let id2 = store.insert("b.md", "Title B", "body b", "vault").unwrap();
        store.upsert_embedding(id1, &dummy, 2).unwrap();
        store.upsert_embedding(id2, &dummy, 2).unwrap();

        let valid = store.count_valid().unwrap();
        let embedded = store.count_embeddings().unwrap();
        assert!(valid > 0, "expected at least one valid doc");
        let coverage_pct = (embedded as f64 / valid as f64) * 100.0;
        assert_eq!(
            coverage_pct, 100.0,
            "expected 100% coverage, got {embedded}/{valid}"
        );
    }

    // ── collect_docs (rg-absent fallback walk) ────────────────────────────────

    #[test]
    fn collect_docs_walks_nested_and_skips_noise_dirs() {
        use std::collections::HashSet;
        use std::fs;
        // Unique temp root so concurrent cargo-test threads don't collide.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "recall_collect_docs_{}_{stamp}",
            std::process::id()
        ));
        let deep = root.join("a").join("b").join("c");
        fs::create_dir_all(&deep).unwrap();
        fs::create_dir_all(root.join("node_modules")).unwrap();
        fs::create_dir_all(root.join(".git")).unwrap();

        fs::write(root.join("top.md"), "x").unwrap();
        fs::write(root.join("a").join("mid.qmd"), "x").unwrap();
        fs::write(deep.join("leaf.md"), "x").unwrap(); // deeply nested → exercises walk
        fs::write(root.join("note.txt"), "x").unwrap(); // wrong ext → ignored
        fs::write(root.join("node_modules").join("dep.md"), "x").unwrap(); // skip dir
        fs::write(root.join(".git").join("hook.md"), "x").unwrap(); // dotdir → skip

        let ex = PathExcludeList::load();
        let mut out = Vec::new();
        collect_docs(&root, &mut out, &ex).unwrap();

        let names: HashSet<String> = out
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        assert!(names.contains("top.md"), "top-level md found");
        assert!(names.contains("mid.qmd"), "nested qmd found");
        assert!(
            names.contains("leaf.md"),
            "deeply nested md found via iterative walk"
        );
        assert!(!names.contains("note.txt"), "non-md/qmd ignored");
        assert!(!names.contains("dep.md"), "node_modules contents skipped");
        assert!(!names.contains("hook.md"), ".git contents skipped");
        assert_eq!(out.len(), 3, "exactly the 3 knowledge files collected");

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn ingest_paths_dry_run_counts_without_mutating() {
        use std::fs;

        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "recall_ingest_paths_{}_{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("notes")).unwrap();
        fs::write(root.join("notes").join("one.md"), "# One\nbody").unwrap();
        fs::write(root.join("notes").join("two.qmd"), "# Two\nbody").unwrap();
        fs::write(root.join("notes").join("skip.txt"), "body").unwrap();

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let paths = vec![
            PathBuf::from("notes/one.md"),
            PathBuf::from("notes/two.qmd"),
            PathBuf::from("notes/skip.txt"),
        ];

        let dry = store.ingest_paths(&root, &paths, true).unwrap();
        assert_eq!(dry, 2, "dry-run counts only md/qmd candidates");
        assert_eq!(
            store.count_valid().unwrap(),
            0,
            "dry-run must not insert documents"
        );

        let inserted = store.ingest_paths(&root, &paths, false).unwrap();
        assert_eq!(inserted, 2, "apply indexes the same two candidates");
        assert_eq!(store.count_valid().unwrap(), 2);

        let unchanged = store.ingest_paths(&root, &paths, true).unwrap();
        assert_eq!(unchanged, 0, "dry-run skips unchanged indexed docs");

        fs::write(root.join("notes").join("one.md"), "# One\nupdated body").unwrap();
        let changed = store.ingest_paths(&root, &paths, true).unwrap();
        assert_eq!(changed, 1, "dry-run counts changed indexed docs");

        let refreshed = store.ingest_paths(&root, &paths, false).unwrap();
        assert_eq!(refreshed, 1, "apply refreshes changed indexed docs");
        let stats = store.stats().unwrap();
        assert_eq!(stats.valid, 2, "refresh keeps one valid row per path");
        assert_eq!(stats.superseded, 1, "refresh supersedes without deleting");

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn ingest_paths_refreshes_title_only_drift() {
        use std::fs;

        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "recall_ingest_paths_title_drift_{}_{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("notes")).unwrap();
        let body = "# Current Title\nbody";
        fs::write(root.join("notes").join("item.md"), body).unwrap();

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store
            .insert("notes/item.md", "Stale Title", body, "vault")
            .unwrap();
        let paths = vec![PathBuf::from("notes/item.md")];

        assert_eq!(store.ingest_paths(&root, &paths, true).unwrap(), 1);
        assert_eq!(
            store.stats().unwrap().superseded,
            0,
            "dry-run must not mutate the stale row"
        );
        assert_eq!(store.ingest_paths(&root, &paths, false).unwrap(), 1);

        let current_title: String = store
            .conn
            .query_row(
                "SELECT title FROM documents WHERE path = 'notes/item.md' AND valid_to IS NULL",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(current_title, "Current Title");
        let stats = store.stats().unwrap();
        assert_eq!(stats.valid, 1);
        assert_eq!(stats.superseded, 1);
        assert_eq!(store.ingest_paths(&root, &paths, true).unwrap(), 0);

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn ingest_paths_reuses_and_refreshes_case_equivalent_valid_path() {
        use std::fs;

        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "recall_ingest_paths_case_equivalent_{}_{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("notes")).unwrap();
        fs::write(root.join("notes").join("item.md"), "# Item\nbody").unwrap();

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store
            .insert("NOTES/ITEM.md", "Item", "# Item\nbody", "vault")
            .unwrap();
        let paths = vec![PathBuf::from("notes/item.md")];

        assert_eq!(
            store.ingest_paths(&root, &paths, true).unwrap(),
            0,
            "case-equivalent unchanged path must not be counted as missing"
        );
        assert_eq!(store.ingest_paths(&root, &paths, false).unwrap(), 0);
        assert_eq!(store.count_valid().unwrap(), 1);

        fs::write(root.join("notes").join("item.md"), "# Item\nupdated").unwrap();
        assert_eq!(store.ingest_paths(&root, &paths, true).unwrap(), 1);
        assert_eq!(store.ingest_paths(&root, &paths, false).unwrap(), 1);
        let stats = store.stats().unwrap();
        assert_eq!(stats.valid, 1, "refresh keeps one logical path valid");
        assert_eq!(
            stats.superseded, 1,
            "case-equivalent stale row is superseded"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn ingest_paths_rejects_divergent_case_equivalent_valid_paths() {
        use std::fs;

        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "recall_ingest_paths_divergent_case_{}_{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("notes")).unwrap();
        fs::write(root.join("notes").join("item.md"), "# Item\ncurrent").unwrap();

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store
            .insert("NOTES/ITEM.md", "Item", "# Item\nold-a", "vault")
            .unwrap();
        store
            .insert("notes/item.md", "Item", "# Item\nold-b", "vault")
            .unwrap();

        let err = store
            .ingest_paths(&root, &[PathBuf::from("notes/item.md")], true)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("divergent case-equivalent valid paths"),
            "unexpected error: {err}"
        );
        assert_eq!(
            store.count_valid().unwrap(),
            2,
            "dry-run must not mutate rows"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn ingest_paths_indexes_non_utf8_markdown_lossily() {
        use std::fs;

        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "recall_ingest_paths_non_utf8_{}_{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("notes")).unwrap();
        fs::write(
            root.join("notes").join("win1252.md"),
            b"# Reviewer \x97 Notes\nbody",
        )
        .unwrap();

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let paths = vec![PathBuf::from("notes/win1252.md")];

        let dry = store.ingest_paths(&root, &paths, true).unwrap();
        assert_eq!(dry, 1, "dry-run counts the markdown candidate");

        let inserted = store.ingest_paths(&root, &paths, false).unwrap();
        assert_eq!(inserted, 1, "apply indexes lossy-decoded markdown");
        assert_eq!(store.count_valid().unwrap(), 1);

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn ingest_paths_rejects_parent_dir_escape() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let err = store
            .ingest_paths(
                std::path::Path::new("C:/vault"),
                &[PathBuf::from("../outside.md")],
                true,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("escapes vault root"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn resolve_ingest_abs_path_uses_platform_separators_for_slash_paths() {
        let resolved = resolve_ingest_abs_path(
            std::path::Path::new("/tmp/recall-vault"),
            "notes/deep/item.md",
        )
        .unwrap();
        let expected = std::path::Path::new("/tmp/recall-vault")
            .join("notes")
            .join("deep")
            .join("item.md");

        assert_eq!(resolved, expected);
    }

    #[test]
    fn ingest_paths_resolves_canonical_claude_skill_paths_from_home() {
        use std::fs;

        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let home = std::env::temp_dir().join(format!(
            "recall_claude_skill_home_{}_{stamp}",
            std::process::id()
        ));
        let vault = std::env::temp_dir().join(format!(
            "recall_claude_skill_vault_{}_{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(home.join(".claude").join("skills").join("example")).unwrap();
        fs::create_dir_all(&vault).unwrap();
        fs::write(
            home.join(".claude")
                .join("skills")
                .join("example")
                .join("SKILL.md"),
            "# Example Skill\nsafe body",
        )
        .unwrap();

        let old_userprofile = std::env::var_os("USERPROFILE");
        let old_home = std::env::var_os("HOME");
        std::env::set_var("USERPROFILE", &home);
        std::env::remove_var("HOME");

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let paths = vec![PathBuf::from(".claude/skills/example/SKILL.md")];
        let dry = store.ingest_paths(&vault, &paths, true).unwrap();
        let inserted = store.ingest_paths(&vault, &paths, false).unwrap();

        if let Some(value) = old_userprofile {
            std::env::set_var("USERPROFILE", value);
        } else {
            std::env::remove_var("USERPROFILE");
        }
        if let Some(value) = old_home {
            std::env::set_var("HOME", value);
        } else {
            std::env::remove_var("HOME");
        }
        fs::remove_dir_all(&home).ok();
        fs::remove_dir_all(&vault).ok();

        assert_eq!(dry, 1, "dry-run sees canonical external skill file");
        assert_eq!(inserted, 1, "apply indexes canonical external skill file");
        assert_eq!(store.count_valid().unwrap(), 1);
    }

    #[test]
    fn docs_missing_embeddings_for_paths_is_targeted_and_deduped() {
        use std::fs;

        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "recall_missing_embeddings_paths_{}_{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("notes")).unwrap();
        fs::write(root.join("notes").join("one.md"), "# One\nbody").unwrap();
        fs::write(root.join("notes").join("two.md"), "# Two\nbody").unwrap();
        fs::write(root.join("notes").join("three.md"), "# Three\nbody").unwrap();

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let paths = vec![
            PathBuf::from("notes/one.md"),
            PathBuf::from("notes/two.md"),
            PathBuf::from("notes/three.md"),
        ];
        assert_eq!(store.ingest_paths(&root, &paths, false).unwrap(), 3);

        let selected = store
            .docs_missing_embeddings_for_paths(
                &root,
                &[
                    PathBuf::from("notes/one.md"),
                    PathBuf::from("notes/one.md"),
                    PathBuf::from("notes/two.md"),
                    PathBuf::from("notes/not-indexed.md"),
                ],
            )
            .unwrap();
        assert_eq!(
            selected.len(),
            2,
            "duplicates and non-indexed paths are skipped"
        );

        let dummy = vec![0_u8; 8];
        store.upsert_embedding(selected[0].0, &dummy, 2).unwrap();
        let selected_after = store
            .docs_missing_embeddings_for_paths(
                &root,
                &[PathBuf::from("notes/one.md"), PathBuf::from("notes/two.md")],
            )
            .unwrap();
        assert_eq!(
            selected_after.len(),
            1,
            "already embedded docs are skipped from targeted work queue"
        );

        let selected_case_variant = store
            .docs_missing_embeddings_for_paths(&root, &[PathBuf::from("NOTES/TWO.md")])
            .unwrap();
        assert_eq!(
            selected_case_variant.len(),
            1,
            "path-list matching is case-insensitive for Windows vault paths"
        );

        fs::remove_dir_all(&root).ok();
    }

    // ── FTS token-cap unit tests ──────────────────────────────────────────────

    #[test]
    fn snippet_is_never_computed_during_the_ranking_scan() {
        // 🔴 PERFORMANCE REGRESSION GUARD, and it guards a measured 570 ms.
        //
        // `snippet()` inside the ranking SELECT is evaluated for every matching row, and
        // `fts_match_expr` ORs every query term so a typical question matches ~94% of the corpus.
        // Measured 2026-08-21: 729 ms to rank, 1,544 ms to rank WITH snippet, 245 ms to fetch
        // snippets separately for the 72 survivors.
        //
        // This asserts the SHAPE rather than the timing, because a latency assertion on a shared
        // machine is a coin flip. Moving `snippet(` back into the rank query fails this test.
        let rank = Store::fts_rank_sql("", true);
        assert!(
            !rank.contains("snippet("),
            "snippet() must NOT appear in the ranking query - it would be computed for every \
             matching row, roughly doubling the query. Fetch snippets for survivors instead.\n{rank}"
        );
        assert!(rank.contains("bm25("), "the ranking query must still rank by bm25");
        assert!(rank.contains("ORDER BY score"), "ranking must still be ordered");

        let snip = Store::fts_snippet_sql(3);
        assert!(snip.contains("snippet("), "the snippet query must compute snippets");
        assert!(
            snip.contains("MATCH ?1"),
            "the snippet query MUST repeat the MATCH - snippet() is undefined outside a full-text \
             match context and raises a SQL error without it"
        );
        assert_eq!(
            snip.matches('?').count(),
            4,
            "one placeholder for MATCH plus one per rowid"
        );
    }

    #[test]
    fn fts_search_still_returns_snippets_after_the_split() {
        // The split must not silently return empty snippets - that would be a "faster and useless"
        // outcome, and every caller renders this text.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store
            .insert(
                "notes/alpha.md",
                "Alpha Title",
                "the quick brown fox jumps over the lazy dog near the riverbank",
                "vault",
            )
            .unwrap();
        let hits = store
            .fts_search_with_tiers("riverbank", 5, &TierFilter::from_spec("all"))
            .unwrap();
        assert_eq!(hits.len(), 1, "the indexed doc must be found");
        assert!(
            !hits[0].snippet.is_empty(),
            "snippet must be populated by the second query, not left blank"
        );
        assert!(
            hits[0].snippet.contains('[') && hits[0].snippet.contains(']'),
            "snippet must carry the highlight markers, got {:?}",
            hits[0].snippet
        );
    }

    #[test]
    fn chunk_path_queue_asks_the_chunk_table_and_keeps_title_and_body_apart() {
        // The two failures this guards, both of which look like success at the call site:
        //
        // 1. Asking `embeddings` whether a document is done while WRITING `bge_chunk_embeddings`.
        //    Every document stays "missing" forever and the loop re-embeds the same batch until
        //    it is killed — a pooled row must NOT satisfy this queue.
        // 2. Returning `title || ' ' || body` pre-joined, the way the pooled sibling does. The
        //    chunker needs the title separately to build its heading prefix, and body offsets
        //    that line up with the stored `char_start`/`char_end`.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store
            .insert("notes/one.md", "One", "body one", "vault")
            .unwrap();
        store
            .insert("notes/two.md", "Two", "body two", "vault")
            .unwrap();

        let root = std::path::Path::new("C:/vault");
        let paths = vec![
            PathBuf::from("notes/one.md"),
            PathBuf::from("notes/one.md"),
            PathBuf::from("notes/two.md"),
            PathBuf::from("notes/not-indexed.md"),
        ];
        let selected = store
            .docs_missing_chunk_embeddings_for_paths(root, &paths)
            .unwrap();
        assert_eq!(
            selected.len(),
            2,
            "duplicates and non-indexed paths are skipped"
        );
        assert_eq!(
            (selected[0].1.as_str(), selected[0].2.as_str()),
            ("One", "body one"),
            "title and body must arrive SEPARATE, not pre-joined"
        );

        // A POOLED vector must not count as done for the chunk queue.
        store
            .upsert_embedding(selected[0].0, &vec![0_u8; 8], 2)
            .unwrap();
        assert_eq!(
            store
                .docs_missing_chunk_embeddings_for_paths(root, &paths)
                .unwrap()
                .len(),
            2,
            "a pooled `embeddings` row must NOT satisfy the per-chunk queue"
        );

        // A CHUNK vector must.
        let place = ChunkPlacement {
            heading_path: "One",
            char_start: 0,
            char_end: 8,
        };
        store
            .upsert_chunk_embedding(selected[0].0, 0, &vec![0_u8; 8], 2, &place)
            .unwrap();
        let after = store
            .docs_missing_chunk_embeddings_for_paths(root, &paths)
            .unwrap();
        assert_eq!(after.len(), 1, "a chunk row DOES satisfy the per-chunk queue");
        assert_eq!(after[0].3, "notes/two.md", "the remaining doc is the other one");

        // Case-folding matches the pooled sibling's behaviour.
        assert_eq!(
            store
                .docs_missing_chunk_embeddings_for_paths(root, &[PathBuf::from("NOTES/TWO.md")])
                .unwrap()
                .len(),
            1,
            "path matching is case-insensitive"
        );
    }

    #[test]
    fn chunk_path_queue_batches_and_preserves_request_order() {
        // Same 512-wide `IN (...)` batching as the pooled sibling, and the same guarantee that
        // results come back in REQUEST order rather than doc-id order — 513 rows so the batch
        // boundary is actually crossed.
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let mut paths = Vec::new();
        for index in 0..513 {
            let path = format!("notes/batch-{index:04}.md");
            store
                .insert(&path, &format!("Title {index}"), &format!("Body {index}"), "vault")
                .unwrap();
            paths.push(PathBuf::from(path));
        }
        paths.reverse();

        let selected = store
            .docs_missing_chunk_embeddings_for_paths(std::path::Path::new("C:/vault"), &paths)
            .unwrap();

        assert_eq!(selected.len(), paths.len());
        assert!(selected
            .iter()
            .zip(paths.iter())
            .all(|((_, _, _, selected_path), requested)| selected_path
                == &requested.to_string_lossy()));
    }

    #[test]
    fn docs_missing_embeddings_for_paths_batches_and_preserves_request_order() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let mut paths = Vec::new();
        for index in 0..513 {
            let path = format!("notes/batch-{index:04}.md");
            store
                .insert(
                    &path,
                    &format!("Title {index}"),
                    &format!("Body {index}"),
                    "vault",
                )
                .unwrap();
            paths.push(PathBuf::from(path));
        }
        paths.reverse();

        let selected = store
            .docs_missing_embeddings_for_paths(std::path::Path::new("C:/vault"), &paths)
            .unwrap();

        assert_eq!(selected.len(), paths.len());
        assert!(selected
            .iter()
            .zip(paths.iter())
            .all(|((_, _, selected_path), requested_path)| selected_path
                == &requested_path.to_string_lossy()));
    }

    #[test]
    fn fts_match_expr_caps_large_query_at_max_match_tokens() {
        // A query with far more than MAX_MATCH_TOKENS alphanumeric tokens must yield exactly
        // MAX_MATCH_TOKENS OR-clauses — no panic, non-empty result.
        let many_words: String = (0..1000)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let expr = fts_match_expr(&many_words);
        assert!(!expr.is_empty(), "expression must not be empty");
        // Each OR-clause is quoted: `"wordN"`. Count occurrences of `" OR "`.
        let clause_count = expr.split(" OR ").count();
        assert_eq!(
            clause_count,
            MAX_MATCH_TOKENS,
            "large query must be capped at exactly MAX_MATCH_TOKENS ({MAX_MATCH_TOKENS}) clauses, got {clause_count}"
        );
    }

    /// T12: the judge's tokens are the first 64 of the searched tokens.
    #[test]
    fn judged_tokens_are_prefix_of_searched_tokens() {
        let q: String = (0..300).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        assert_eq!(fts_query_tokens(&q), fts_tokens_capped(&q, 256)[..64].to_vec());
        let at = |n: usize| (0..n).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        assert!(!exceeds_judge_cap(&at(64)));
        assert!(exceeds_judge_cap(&at(65)));
    }

    #[test]
    fn small_stores_take_the_original_query_path_and_big_ones_the_pushdown() {
        let legacy = Store::fts_rank_sql("", false);
        assert!(!legacy.contains("LEFT JOIN") && legacy.contains("JOIN documents d ON d.id = documents_fts.rowid"));
        assert!(legacy.contains("ORDER BY score, d.path, d.id"), "the original ordering, unchanged");
        assert!(!legacy.contains("snippet("));
        assert!(!legacy.contains("eligible"), "the original form has the original four columns");
        let big_sql = Store::fts_rank_sql("", true);
        assert!(big_sql.contains("LEFT JOIN") && big_sql.contains("LIMIT ?2) f"));

        let store = Store::open_at(Path::new(":memory:")).unwrap();
        for i in 0..FTS_PRUNE_MIN_DF {
            store.insert(&format!("n{i}.md"), "T", &format!("common word{i}"), "vault").unwrap();
        }
        assert!(!store.fts_over_floor().unwrap(), "exactly the floor is still a small store");
        let small = store.fts_search("common word7", 3).unwrap();
        store.insert("one-more.md", "T", "common extra", "vault").unwrap();
        assert!(store.fts_over_floor().unwrap(), "one row past the floor takes the big-store path");
        // Same answer on both sides of the line for a query both paths can serve.
        let big = store.fts_search("common word7", 3).unwrap();
        assert_eq!(small.iter().map(|h| h.path.clone()).collect::<Vec<_>>(), big.iter().map(|h| h.path.clone()).collect::<Vec<_>>());
    }

    #[test]
    fn pushdown_breaks_score_ties_by_path_then_id_like_the_original() {
        // Codex repro: more than the floor of identically scored documents, k=1. The pushdown's
        // inner LIMIT cuts by rowid, so without the tie widening it returned the lowest id, not the
        // lowest path. Paths run opposite to ids so the two orders disagree.
        let store = Store::open_at(Path::new(":memory:")).unwrap();
        let n = FTS_PRUNE_MIN_DF + 1;
        for i in 0..n {
            store.insert(&format!("p{:05}.md", n - i), "T", "tie token", "vault").unwrap();
        }
        assert!(store.fts_over_floor().unwrap());
        let one = store.fts_search("tie token", 1).unwrap();
        assert_eq!(one.iter().map(|h| h.path.as_str()).collect::<Vec<_>>(), vec!["p00001.md"]);
        let three = store.fts_search("tie token", 3).unwrap();
        assert_eq!(three.iter().map(|h| h.path.as_str()).collect::<Vec<_>>(), vec!["p00001.md", "p00002.md", "p00003.md"]);
    }

    /// T13: real FTS5 accepts a query that hits the ranking cap (2,000 distinct words).
    #[test]
    fn fts_search_accepts_a_max_match_tokens_query() {
        let store = Store::open_at(Path::new(":memory:")).unwrap();
        store.insert("a.md", "A", "w7 w300 alpha", "vault").unwrap();
        let q: String = (0..2000).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        assert_eq!(store.fts_search(&q, 5).unwrap().len(), 1);
    }

    /// T14: a term past position 64 is searched for ranking, not by the judged search.
    #[test]
    fn fts_search_finds_a_term_past_position_64() {
        let store = Store::open_at(Path::new(":memory:")).unwrap();
        store.insert("a.md", "A", "zz100", "vault").unwrap();
        let q: String = (0..150).map(|i| format!("zz{i}")).collect::<Vec<_>>().join(" ");
        assert_eq!(store.fts_search(&q, 5).unwrap().len(), 1);
        assert!(store.fts_search_judged(&q, 5).unwrap().is_empty());
        assert!(store.fts_judge_ids(&q, 5).unwrap().is_empty());
        let judged = store.fts_search_judged("zz100", 5).unwrap();
        assert!(!judged[0].snippet.is_empty(), "the judged search keeps snippets");
    }

    #[test]
    fn fts_match_expr_short_query_unchanged() {
        // A normal short query must produce the same number of clauses as tokens,
        // unaffected by the cap.
        let expr = fts_match_expr("rust memory search");
        let clauses: Vec<&str> = expr.split(" OR ").collect();
        assert_eq!(
            clauses.len(),
            3,
            "3-token query must yield exactly 3 OR-clauses, got {}",
            clauses.len()
        );
        assert!(clauses.contains(&"\"rust\""), "clause for 'rust' present");
        assert!(
            clauses.contains(&"\"memory\""),
            "clause for 'memory' present"
        );
        assert!(
            clauses.contains(&"\"search\""),
            "clause for 'search' present"
        );
    }

    // ── ingest_dir doc-id stability ───────────────────────────────────────────

    /// THE regression test for the 2026-07-31 incident: `recall index` used to
    /// DELETE + INSERT every file, so one re-index renumbered all ~136k docs and
    /// orphaned every vector table keyed on doc_id (recall@5 collapsed).
    /// If this test ever fails, the semantic index is being silently destroyed.
    #[test]
    fn reindex_preserves_doc_ids_and_embeddings() {
        let dir = std::env::temp_dir().join(format!(
            "recall-idstab-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.md"), "# A
alpha body").unwrap();
        std::fs::write(dir.join("b.md"), "# B
beta body").unwrap();

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store.ingest_dir(&dir).unwrap();

        let ids_before: Vec<(i64, String)> = store
            .conn
            .prepare("SELECT id, path FROM documents WHERE source='vault' ORDER BY path")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(ids_before.len(), 2);

        // Attach a vector to each doc, exactly as `recall embed` would.
        let dummy: Vec<u8> = vec![0u8; 8];
        for (id, _) in &ids_before {
            store.upsert_embedding(*id, &dummy, 2).unwrap();
        }

        // Change ONE file, leave the other byte-identical, then re-index.
        std::fs::write(dir.join("a.md"), "# A
alpha body EDITED").unwrap();
        store.ingest_dir(&dir).unwrap();

        let ids_after: Vec<(i64, String)> = store
            .conn
            .prepare("SELECT id, path FROM documents WHERE source='vault' ORDER BY path")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(
            ids_before, ids_after,
            "re-index renumbered documents — every vector table keyed on doc_id              just became orphaned rows"
        );

        // Both edited and unedited docs keep their vectors (ON DELETE CASCADE
        // only fires on a real DELETE, which must no longer happen).
        let raw: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(raw, 2, "embeddings were cascaded away by the re-index");

        // The edit really did land (we updated in place, not skipped).
        let body: String = store
            .conn
            .query_row(
                "SELECT body FROM documents WHERE path='a.md'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(body.contains("EDITED"), "in-place update did not write the new body");

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── tombstone-on-index (--prune-missing) ──────────────────────────────────

    fn prune_tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "recall-{}-{}",
            tag,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn valid_to_of(store: &Store, path: &str) -> Option<String> {
        store
            .conn
            .query_row(
                "SELECT valid_to FROM documents WHERE path = ?1",
                params![path],
                |r| r.get(0),
            )
            .unwrap()
    }

    /// THE test for the bug this feature exists to fix: a file deleted from the vault leaves a
    /// live document behind forever. Measured on the live store 2026-08-18 — 10,637 of 69,596
    /// live docs (15.3%) pointed at a file that no longer existed, and a full `recall index`
    /// run right after the deletion did not change that number by one.
    #[test]
    fn prune_missing_tombstones_a_document_whose_file_was_deleted() {
        let dir = prune_tmp_dir("prune-deleted");
        for i in 0..10 {
            std::fs::write(dir.join(format!("note{i:02}.md")), format!("# N{i}\nbody {i}")).unwrap();
        }

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store.ingest_dir(&dir).unwrap();
        assert_eq!(store.count_valid().unwrap(), 10);

        // Delete exactly one of ten. That is a 10% shrink, which sits ON the gate boundary
        // (floor(10 * 0.90) = 9, and 9 < 9 is false) — the boundary is deliberately inclusive
        // so a vault that loses a tenth of itself is still prunable without an override.
        std::fs::remove_file(dir.join("note07.md")).unwrap();

        let report = store
            .prune_missing(&dir, PruneMode::Apply, &dir.join("prune-lists"))
            .unwrap();

        assert_eq!(report.tombstoned, 1, "the deleted file was not tombstoned");
        assert_eq!(report.missing, vec!["note07.md".to_string()]);
        assert!(
            valid_to_of(&store, "note07.md").is_some(),
            "valid_to must be set on the document whose file is gone"
        );
        assert!(
            valid_to_of(&store, "note06.md").is_none(),
            "a document whose file still exists must stay live"
        );
        assert_eq!(
            store.count_valid().unwrap(),
            9,
            "exactly one document should have left the live set"
        );
        // Reversible by construction: the id is untouched, so every vector keyed to it still
        // resolves and `valid_to = NULL` puts the row straight back.
        let still_there: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM documents WHERE path = 'note07.md'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(still_there, 1, "the row must be tombstoned, never deleted");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A tombstoned file that comes BACK must reclaim its original row, not get a fresh one.
    /// Tombstoning is what made `ingest_dir`'s insert branch reachable for a path that already
    /// has a row, and a new id there would orphan every vector keyed to the old one — the
    /// 2026-07-31 renumbering failure, re-introduced one document at a time.
    #[test]
    fn a_restored_file_reclaims_its_doc_id_instead_of_being_renumbered() {
        let dir = prune_tmp_dir("prune-revive");
        for i in 0..10 {
            std::fs::write(dir.join(format!("r{i:02}.md")), format!("# R{i}\nbody {i}")).unwrap();
        }
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store.ingest_dir(&dir).unwrap();

        let id_before: i64 = store
            .conn
            .query_row("SELECT id FROM documents WHERE path = 'r04.md'", [], |r| {
                r.get(0)
            })
            .unwrap();
        store.upsert_embedding(id_before, &[0u8; 8], 2).unwrap();

        std::fs::remove_file(dir.join("r04.md")).unwrap();
        let report = store
            .prune_missing(&dir, PruneMode::Apply, &dir.join("prune-lists"))
            .unwrap();
        assert_eq!(report.tombstoned, 1);

        // The note comes back — restored from a backup, or the deletion was a mistake.
        std::fs::write(dir.join("r04.md"), "# R4\nbody 4 restored").unwrap();
        store.ingest_dir(&dir).unwrap();

        let rows: Vec<(i64, Option<String>)> = store
            .conn
            .prepare("SELECT id, valid_to FROM documents WHERE path = 'r04.md'")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(rows.len(), 1, "a second row was inserted instead of reviving");
        assert_eq!(rows[0].0, id_before, "the doc id was renumbered on restore");
        assert!(rows[0].1.is_none(), "the revived row is still tombstoned");

        let vec_rows: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM embeddings WHERE doc_id = ?1",
                params![id_before],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(vec_rows, 1, "the revived document lost its vector");

        let body: String = store
            .conn
            .query_row("SELECT body FROM documents WHERE path = 'r04.md'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(body.contains("restored"), "revive did not write the new body");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// 🔴 The gate must FAIL CLOSED. A truncated enumeration — rg losing its 5-second race, an
    /// AV intercept, a stalled path — looks exactly like a mass deletion, and tombstoning off
    /// one would mark thousands of live documents dead in a single pass.
    #[test]
    fn prune_missing_refuses_when_enumeration_is_short_and_changes_nothing() {
        let dir = prune_tmp_dir("prune-truncated");
        std::fs::write(dir.join("a.md"), "# A\nalpha").unwrap();
        std::fs::write(dir.join("b.md"), "# B\nbeta").unwrap();

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store.ingest_dir(&dir).unwrap();

        // Seed the history with a healthy 5,000-doc enumeration of this same vault. This run
        // will see 2 — the shape of an enumeration that came back truncated.
        store
            .conn
            .execute(
                "INSERT INTO index_runs (ran_at, enumerated, used_fallback, tombstoned)
                 VALUES ('0', 5000, 0, 0)",
                [],
            )
            .unwrap();

        let err = store
            .prune_missing(&dir, PruneMode::Apply, &dir.join("prune-lists"))
            .expect_err("a 2-doc enumeration against a 5,000-doc baseline must be refused");
        assert!(
            err.to_string().contains("prune gate refused"),
            "unexpected error: {err}"
        );

        assert_eq!(
            store.count_valid().unwrap(),
            2,
            "a refused prune must change nothing at all"
        );
        assert!(valid_to_of(&store, "a.md").is_none());
        assert!(valid_to_of(&store, "b.md").is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// 🔴 The single most likely way to get this wrong. `enumerate_docs` ALREADY drops every
    /// `is_excluded` path, so diffing live documents straight against its output would
    /// tombstone every excluded document in the vault — ~29% of the corpus — even though every
    /// one of those files is sitting on disk. Only filesystem absence may tombstone.
    #[test]
    fn prune_missing_never_tombstones_an_excluded_file_that_still_exists() {
        let dir = prune_tmp_dir("prune-excluded");
        std::fs::create_dir_all(dir.join("disabled-loader-roots-snapshot")).unwrap();
        std::fs::write(dir.join("kept.md"), "# Kept\nbody").unwrap();
        // Present on disk, and excluded by BUILTIN_EXCLUDE_PREFIXES ("disabled-").
        std::fs::write(
            dir.join("disabled-loader-roots-snapshot/mirror.md"),
            "# Mirror\nbody",
        )
        .unwrap();

        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store.ingest_dir(&dir).unwrap();
        // ingest_dir skips excluded paths, so the row is inserted directly — which is exactly
        // how the live store came to hold them: they were indexed before the rule existed.
        let excluded_path = "disabled-loader-roots-snapshot/mirror.md";
        store
            .insert(excluded_path, "Mirror", "body", "vault")
            .unwrap();
        // A non-vault observation has no file at all and must never be swept up either.
        store.insert("", "Observation", "remembered thing", "cli").unwrap();

        let report = store
            .prune_missing(&dir, PruneMode::Apply, &dir.join("prune-lists"))
            .unwrap();

        assert!(
            report.not_enumerated >= 1,
            "the excluded doc must reach the candidate set, or this test proves nothing"
        );
        assert_eq!(
            report.tombstoned, 0,
            "an excluded-but-present file was tombstoned: {:?}",
            report.missing
        );
        assert!(valid_to_of(&store, excluded_path).is_none());
        assert!(
            valid_to_of(&store, "").is_none(),
            "a source='cli' observation has no file and must never be tombstoned"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Dry mode is what the owner reads before approving anything: it must report the real count,
    /// write the real list, and touch no row.
    /// Preflight or refuse: without rg the enumeration is the fallback walk, so prune refuses in
    /// both modes, says why, and changes nothing.
    #[test]
    fn prune_refuses_when_rg_is_missing_and_changes_nothing() {
        let dir = prune_tmp_dir("prune-no-rg");
        for i in 0..3 {
            std::fs::write(dir.join(format!("n{i}.md")), format!("# N{i}\nbody {i}")).unwrap();
        }
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store.ingest_dir(&dir).unwrap();
        std::fs::remove_file(dir.join("n1.md")).unwrap();
        for mode in [PruneMode::Dry, PruneMode::Apply] {
            let err = store.prune_missing_inner(&dir, mode, &dir.join("lists"), false).unwrap_err();
            assert!(err.to_string().contains("ripgrep (rg) not found"), "{err}");
        }
        assert_eq!(store.count_valid().unwrap(), 3, "nothing tombstoned");
        assert!(!dir.join("lists").exists(), "no list written");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Regression 2026-11-06: the hourly capture's `index --path-list` of
    /// `<capture dir>/<day>/<id>.md` (a dot-folder) printed "Indexed 0 docs" and exited 0. The
    /// path list itself is fine: a relative path inside a dot-directory indexes against the root.
    #[test]
    fn a_path_list_entry_inside_a_dot_dir_indexes() {
        let dir = prune_tmp_dir("dotdir-pathlist");
        let rel = "notes/.captures/2026-11-06/session.md";
        std::fs::create_dir_all(dir.join("notes/.captures/2026-11-06")).unwrap();
        std::fs::write(dir.join(rel), "# Session\n\ncaptured text").unwrap();
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        assert_eq!(store.ingest_paths(&dir, &[PathBuf::from(rel)], false).unwrap(), 1);
        assert!(valid_to_of(&store, rel).is_none() && live_id_like(&store, rel).is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// ...and the real cause: the caller did not set RECALL_VAULT, so the root was the default
    /// `~/recall-notes`, which did not exist. Writers now refuse a missing root instead.
    #[test]
    fn a_missing_vault_root_is_refused_with_the_fix() {
        let dir = prune_tmp_dir("vault-root-check");
        assert_eq!(check_vault_root(dir.clone()).unwrap(), dir);
        let missing = dir.join("no-such-vault");
        let err = check_vault_root(missing).unwrap_err().to_string();
        assert!(err.contains("vault root not found") && err.contains("RECALL_VAULT"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rg_available_is_false_for_a_binary_that_does_not_exist() {
        assert!(!rg_available_at("recall-no-such-rg-binary-7f3a"));
    }

    #[test]
    fn prune_missing_dry_writes_the_list_and_mutates_nothing() {
        let dir = prune_tmp_dir("prune-dry");
        for i in 0..10 {
            std::fs::write(dir.join(format!("n{i:02}.md")), format!("# N{i}\nbody {i}")).unwrap();
        }
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store.ingest_dir(&dir).unwrap();
        std::fs::remove_file(dir.join("n03.md")).unwrap();

        let list_dir = dir.join("prune-lists");
        let report = store
            .prune_missing(&dir, PruneMode::Dry, &list_dir)
            .unwrap();

        assert_eq!(report.missing, vec!["n03.md".to_string()]);
        assert_eq!(report.tombstoned, 0, "dry mode must not mutate");
        assert_eq!(store.count_valid().unwrap(), 10, "dry mode must not mutate");
        assert!(valid_to_of(&store, "n03.md").is_none());

        // Not one row, anywhere — including the run history. `ingest_dir` above wrote exactly
        // one `index_runs` row; the dry prune must not have added a second.
        let runs: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM index_runs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(runs, 1, "dry mode wrote to index_runs — that is a mutation");

        let text = std::fs::read_to_string(&report.list_path).unwrap();
        assert!(
            text.contains("n03.md"),
            "the would-tombstone list must name the path: {text}"
        );
        assert!(
            text.contains("valid_to = NULL"),
            "the list must carry its own undo instruction: {text}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // The gate as a pure decision — every branch, no filesystem.

    #[test]
    fn prune_gate_allows_a_healthy_enumeration() {
        assert!(prune_gate_verdict(69_596, Some(69_600), false).is_ok());
        // Exactly at the floor is allowed: floor(1000 * 0.90) = 900, and 900 < 900 is false.
        assert!(prune_gate_verdict(900, Some(1000), false).is_ok());
    }

    #[test]
    fn prune_gate_refuses_a_count_below_the_shrink_floor() {
        let err = prune_gate_verdict(899, Some(1000), false)
            .expect_err("899 is below the 900 floor and must be refused");
        assert!(err.contains("below the baseline"), "unexpected: {err}");
    }

    #[test]
    fn prune_gate_refuses_any_drop_at_all_on_the_fallback_walk() {
        // Only 1 short of the baseline — well inside the 10% band — but the fallback walk
        // enumerates a genuinely different set (no .gitignore, skips node_modules/target/
        // dotdirs), so its shortfall carries no information about the disk.
        let err = prune_gate_verdict(999, Some(1000), true)
            .expect_err("a fallback enumeration that dropped at all must be refused");
        assert!(err.contains("fallback walk"), "unexpected: {err}");
        // The same count from rg is fine.
        assert!(prune_gate_verdict(999, Some(1000), false).is_ok());
        // A fallback run that did NOT drop is fine.
        assert!(prune_gate_verdict(1000, Some(1000), true).is_ok());
    }

    #[test]
    fn prune_gate_allows_when_no_baseline_has_ever_been_recorded() {
        // Nothing to compare against. Allowed, because the per-file filesystem check is an
        // independent authority — refusing forever would make the feature unreachable on a
        // fresh store. The caller prints "Baseline: NONE recorded" prominently.
        assert!(prune_gate_verdict(2, None, false).is_ok());
        assert!(prune_gate_verdict(2, None, true).is_ok());
    }

    #[test]
    fn path_is_absent_resolves_dot_claude_the_way_ingest_does() {
        // 🔴 `resolve_ingest_abs_path` sends `.claude/...` to USERPROFILE, not to the vault root.
        // If `path_is_absent` joins against the root instead, every `.claude` document reads as
        // "its file is gone" forever - which makes it a permanent move candidate whose id and
        // vectors the next byte-identical vault file inherits, and makes prune_missing tombstone
        // it. The two resolvers must agree.
        let root = prune_tmp_dir("claude-absent-root");
        let home = prune_tmp_dir("claude-absent-home");
        let claude = home.join(".claude").join("skills").join("foo");
        std::fs::create_dir_all(&claude).unwrap();
        std::fs::write(claude.join("SKILL.md"), "# S\n\nbody").unwrap();

        let prev = std::env::var_os("USERPROFILE");
        // SAFETY: single-threaded test; restored below.
        unsafe { std::env::set_var("USERPROFILE", &home) };

        let rel = ".claude/skills/foo/SKILL.md";
        let present = !path_is_absent(&root, rel);
        let missing_is_absent = path_is_absent(&root, ".claude/skills/foo/GONE.md");

        match prev {
            Some(v) => unsafe { std::env::set_var("USERPROFILE", v) },
            None => unsafe { std::env::remove_var("USERPROFILE") },
        }

        assert!(
            present,
            "a .claude file that EXISTS under the home must not read as absent"
        );
        assert!(
            missing_is_absent,
            "a .claude file that genuinely does not exist must still read as absent"
        );

        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn path_is_absent_only_answers_yes_to_a_real_not_found() {
        let dir = prune_tmp_dir("prune-absent");
        std::fs::write(dir.join("here.md"), "x").unwrap();

        assert!(!path_is_absent(&dir, "here.md"), "present file");
        assert!(path_is_absent(&dir, "gone.md"), "genuinely deleted file");
        // A path that cannot be resolved against the root safely is never "absent".
        assert!(!path_is_absent(&dir, "../escape.md"));
        // An absolute path on THIS platform (on Linux "C:/Windows/..." is just a relative name).
        let abs = if cfg!(windows) { "C:/Windows/nope.md" } else { "/etc/nope.md" };
        assert!(!path_is_absent(&dir, abs));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The baseline must be the MAX over the window, not the last value — otherwise one
    /// truncated enumeration recorded as "previous" would authorise the next equally
    /// truncated run and the gate would ratchet itself open one bad run at a time.
    /// Every `recall index` stores how long its enumeration took, and an index_runs table from
    /// before the column existed gets it on open.
    #[test]
    fn index_runs_records_the_enumeration_time_and_old_tables_gain_the_column() {
        let dir = prune_tmp_dir("enum-ms");
        std::fs::write(dir.join("a.md"), "# A\nbody").unwrap();
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store.ingest_dir(&dir).unwrap();
        let ms: Option<i64> = store
            .conn
            .query_row("SELECT enum_ms FROM index_runs ORDER BY id DESC LIMIT 1", [], |r| r.get(0))
            .unwrap();
        assert!(ms.is_some_and(|m| m >= 0), "{ms:?}");
        std::fs::remove_dir_all(&dir).ok();

        let db = std::env::temp_dir().join(format!("recall-enum-ms-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&db);
        {
            let c = Connection::open(&db).unwrap();
            c.execute_batch(
                "CREATE TABLE index_runs (id INTEGER PRIMARY KEY, ran_at TEXT NOT NULL,
                 enumerated INTEGER NOT NULL, used_fallback INTEGER NOT NULL,
                 tombstoned INTEGER NOT NULL DEFAULT 0);",
            )
            .unwrap();
        }
        let old = Store::open_at(&db).unwrap();
        old.record_enumeration(5, false, 0, Some(1234)).unwrap();
        let got: i64 = old.conn.query_row("SELECT enum_ms FROM index_runs", [], |r| r.get(0)).unwrap();
        assert_eq!(got, 1234);
        drop(old);
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn prune_baseline_is_the_max_over_the_window_not_the_last_run() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store.record_enumeration(70_000, false, 0, None).unwrap();
        store.record_enumeration(3, false, 0, None).unwrap(); // a truncated run
        assert_eq!(
            store.prune_enumeration_baseline(false).unwrap(),
            Some(70_000),
            "a low reading must not be able to lower the bar"
        );
    }

    /// Baselines are per enumeration METHOD. rg and the fallback walk return different sets
    /// over the same tree, so mixing them is comparing two different measurements — and
    /// keying on the method is also what keeps the gate alive on a machine where rg never
    /// runs, instead of leaving it permanently baseline-less and therefore permissive.
    #[test]
    fn prune_baseline_is_kept_per_enumeration_method() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store.record_enumeration(70_000, false, 0, None).unwrap();
        assert_eq!(store.prune_enumeration_baseline(false).unwrap(), Some(70_000));
        assert_eq!(
            store.prune_enumeration_baseline(true).unwrap(),
            None,
            "an rg count is not a baseline for the walk"
        );

        store.record_enumeration(41_000, true, 0, None).unwrap();
        assert_eq!(store.prune_enumeration_baseline(true).unwrap(), Some(41_000));
        assert_eq!(
            store.prune_enumeration_baseline(false).unwrap(),
            Some(70_000),
            "a walk count must not contaminate the rg baseline"
        );
    }


    #[test]
    fn stats_reports_chunk_coverage_separately_from_legacy() {
        let s = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let a = s.insert("a.md", "A", "one", "test").unwrap();
        let b = s.insert("b.md", "B", "two", "test").unwrap();
        for id in [a, b] {
            s.conn
                .execute(
                    "INSERT INTO bge_chunk_embeddings (doc_id, chunk_index, vec, dim) \
                     VALUES (?1, 0, x'00000000', 1)",
                    params![id],
                )
                .unwrap();
        }
        s.upsert_embedding(a, &[0, 0, 0, 0], 1).unwrap();
        let st = s.stats().unwrap();
        assert_eq!(st.chunk_embedded, 2, "chunk coverage is what the query path reads");
        assert_eq!(st.embeddings, 1, "legacy pooled table is counted separately");
    }

    #[test]
    fn keep_rare_drops_common_tokens_but_keeps_the_rarest_three() {
        let t = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        // df: what 900, did 800, kestra 3, password 40, the 950; max_df 100
        let got = keep_rare(t(&["what", "did", "kestra", "password", "the"]), &[900, 800, 3, 40, 950], 100.0);
        assert_eq!(got, t(&["did", "kestra", "password"]), "the 3 rarest stay even when common, order kept");
        // all common: still the 3 rarest, never empty
        let got = keep_rare(t(&["a1", "b1", "c1", "d1"]), &[10, 20, 30, 40], 0.0);
        assert_eq!(got, t(&["a1", "b1", "c1"]));
        // nothing common: unchanged
        let got = keep_rare(t(&["x1", "y1", "z1", "w1"]), &[1, 2, 3, 4], 100.0);
        assert_eq!(got, t(&["x1", "y1", "z1", "w1"]));
    }
}
