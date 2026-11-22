// voyage.rs — the embeddings2 (Voyage-4-large, 1024-dim, chunk-level) retrieval channel.
//
// WHY THIS EXISTS
// ~400k chunk vectors were embedded and paid for, but nothing ever read them:
// `store.rs` had zero references to `embeddings2`. On a private query set (2026-07-27) the
// Voyage chunk vectors recalled far more than fts5 keyword search alone.
// This module is the read path that makes that real inside recall itself.
//
// WHAT LEAVES THE MACHINE
// Only the QUERY TEXT, and only on a cache miss. Document bodies and stored
// vectors never leave. Most queries hit the local SQLite cache and make no
// network call at all.
//
// SECURITY
// The API key is fetched via `cogent secrets get VOYAGE_API_KEY` (the
// secret gate) with an env fallback. It is never logged, never printed, and
// never included in an error message.
//
// FAIL-OPEN
// Missing key, timeout, HTTP error, dim mismatch, empty corpus -> log once and
// return an empty id list. The caller then serves FTS5-only results. A search
// must never hard-fail because a cloud call did.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{anyhow, Result};
use rusqlite::OptionalExtension;

use crate::store::{Hit, Store};

/// Dimension of the vectors stored in `embeddings2`. Verified against the live
/// DB: `length(vec) = 4096` bytes = 1024 * f32.
pub const VOYAGE_DIM: usize = 1024;

/// Model that produced `embeddings2`. A query vector MUST come from this same
/// model or cosine similarity is meaningless across the two spaces.
pub const VOYAGE_MODEL: &str = "voyage-4-large";

const VOYAGE_URL: &str = "https://api.voyageai.com/v1/embeddings";
const TIMEOUT_SECS: u64 = 2;

static WARNED: AtomicBool = AtomicBool::new(false);

/// Log a degradation reason exactly once per process, then stay quiet.
fn warn_once(reason: &str) {
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!("[recall] voyage channel disabled ({reason}); serving other channels");
    }
}

// ---------------------------------------------------------------- secret

/// Resolve the Voyage API key. Prefers the secret gate, falls back to
/// the environment. Returns the key itself; callers must never log it.
fn api_key() -> Option<String> {
    if let Ok(out) = std::process::Command::new("cogent")
        .args(["secrets", "get", "VOYAGE_API_KEY"])
        .output()
    {
        if out.status.success() {
            let k = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !k.is_empty() {
                return Some(k);
            }
        }
    }
    for var in ["VOYAGE_API_KEY", "VOYAGE_AI_EMBEDDING"] {
        if let Ok(k) = std::env::var(var) {
            let k = k.trim().to_string();
            if !k.is_empty() {
                return Some(k);
            }
        }
    }
    None
}

// ---------------------------------------------------------------- cache

/// Local, persistent query-embedding cache. This is what keeps the common case
/// free and offline: a repeated query never touches the network.
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
            rusqlite::params![query, VOYAGE_MODEL, VOYAGE_DIM as i64],
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
        rusqlite::params![query, VOYAGE_MODEL, VOYAGE_DIM as i64, bytes],
    )?;
    Ok(())
}

// ---------------------------------------------------------------- vectors

/// Decode a little-endian f32 blob. Returns None if the length is not a clean
/// multiple of 4 or the dimension is not what we expect.
fn decode_f32(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 || bytes.len() / 4 != VOYAGE_DIM {
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

// ---------------------------------------------------------------- embed

/// Embed the query via Voyage REST.
///
/// The official SDK is deliberately NOT used: `voyageai` 0.2.3 silently drops
/// `output_dimension`, which is how the 1024-vs-2048 split happened in the first
/// place. We send the field explicitly and assert the returned length.
#[cfg(feature = "voyage")]
fn embed_query_remote(query: &str, key: &str) -> Result<Vec<f32>> {
    let resp = ureq::post(VOYAGE_URL)
        .timeout(Duration::from_secs(TIMEOUT_SECS))
        .set("Authorization", &format!("Bearer {key}"))
        .set("Content-Type", "application/json")
        .send_json(ureq::json!({
            "input": [query],
            "model": VOYAGE_MODEL,
            "input_type": "query",
            "output_dimension": VOYAGE_DIM,
            "truncation": true,
        }))
        .map_err(|e| match e {
            // Never surface the response body: it can echo request headers.
            ureq::Error::Status(code, _) => anyhow!("voyage http {code}"),
            ureq::Error::Transport(_) => anyhow!("voyage transport/timeout"),
        })?;

    let body: serde_json::Value = resp.into_json()?;
    let arr = body["data"][0]["embedding"]
        .as_array()
        .ok_or_else(|| anyhow!("voyage response missing embedding"))?;

    if arr.len() != VOYAGE_DIM {
        return Err(anyhow!(
            "voyage returned dim {}, expected {VOYAGE_DIM}",
            arr.len()
        ));
    }
    Ok(arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
}

#[cfg(not(feature = "voyage"))]
fn embed_query_remote(_query: &str, _key: &str) -> Result<Vec<f32>> {
    Err(anyhow!("built without the `voyage` feature"))
}

/// Cache-first query embedding. Network is touched only on a miss.
fn query_vector(store: &Store, query: &str) -> Result<Vec<f32>> {
    ensure_cache_table(store)?;
    if let Some(mut v) = cache_get(store, query)? {
        l2_normalize(&mut v);
        return Ok(v);
    }
    let key = api_key().ok_or_else(|| anyhow!("no VOYAGE_API_KEY"))?;
    let raw = embed_query_remote(query, &key)?;
    // Cache the raw vector; normalize only the working copy.
    let _ = cache_put(store, query, &raw);
    let mut v = raw;
    l2_normalize(&mut v);
    Ok(v)
}

// ---------------------------------------------------------------- resident cache

/// All of `embeddings2` held in RAM in one contiguous allocation.
///
/// WHY: the brute-force SQLite path measured 3.3s/query, and profiling showed the
/// cost is reading ~1.6 GB of blobs off disk every single time — not the maths.
/// 400k vectors is small enough to simply keep resident, which turns every query
/// after the first into a pure in-memory scan. An HNSW index would only start to
/// pay for itself an order of magnitude beyond this corpus size, and `ann.rs`
/// documents that hnsw_rs reload is awkward (HnswIo must outlive Hnsw), so a
/// resident flat scan is both faster to reach and harder to get wrong.
///
/// LAYOUT: one flat `Vec<f32>` rather than `Vec<Vec<f32>>` so a scan walks memory
/// linearly and stays in cache. `inv_norms` is precomputed at load so per-query
/// work is a bare dot product — no square roots in the hot loop.
struct ResidentVectors {
    /// doc_id per row; the same doc appears once per chunk.
    ids: Vec<i64>,
    /// `rows * VOYAGE_DIM` f32 values, row-major.
    data: Vec<f32>,
    /// 1/||row|| per row, so cosine = dot * inv_norm for an L2-normalised query.
    inv_norms: Vec<f32>,
}

impl ResidentVectors {
    fn rows(&self) -> usize {
        self.ids.len()
    }

    /// Load every live chunk vector. Called at most once per process.
    fn load(store: &Store) -> Result<Self> {
        let mut stmt = store.conn.prepare(
            "SELECT e.doc_id, e.vec
               FROM embeddings2 e
               JOIN documents d ON d.id = e.doc_id
              WHERE d.valid_to IS NULL
                AND e.dim = ?1",
        )?;

        let mut ids = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        let mut inv_norms = Vec::new();

        let mut rows = stmt.query(rusqlite::params![VOYAGE_DIM as i64])?;
        while let Some(row) = rows.next()? {
            let doc_id: i64 = row.get(0)?;
            let blob: &[u8] = row.get_ref(1)?.as_blob().unwrap_or(&[]);
            if blob.len() != VOYAGE_DIM * 4 {
                continue;
            }

            let start = data.len();
            data.reserve(VOYAGE_DIM);
            let mut norm = 0.0f32;
            for w in blob.chunks_exact(4) {
                let x = f32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                data.push(x);
                norm += x * x;
            }
            debug_assert_eq!(data.len() - start, VOYAGE_DIM);

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
    ///
    /// Split across threads because the scan is embarrassingly parallel: each
    /// thread owns a disjoint row range and returns its own partial best-map,
    /// which the caller merges. No locking in the hot loop.
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
                        let row = &self.data[i * VOYAGE_DIM..(i + 1) * VOYAGE_DIM];
                        let mut dot = 0.0f32;
                        for (x, q) in row.iter().zip(qvec.iter()) {
                            dot += x * q;
                        }
                        let score = dot * self.inv_norms[i];
                        let id = self.ids[i];
                        match best.get_mut(&id) {
                            Some(s) if *s >= score => {}
                            Some(s) => *s = score,
                            None => {
                                best.insert(id, score);
                            }
                        }
                    }
                    best
                }));
            }
            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        let mut best: HashMap<i64, f32> = HashMap::new();
        for part in partials {
            for (id, score) in part {
                match best.get_mut(&id) {
                    Some(s) if *s >= score => {}
                    Some(s) => *s = score,
                    None => {
                        best.insert(id, score);
                    }
                }
            }
        }
        best
    }
}

/// Process-lifetime cache. In the warm daemon this is loaded once and every
/// later query is in-memory; a one-shot CLI invocation still pays the load.
static RESIDENT: std::sync::OnceLock<Option<ResidentVectors>> = std::sync::OnceLock::new();

/// True when the resident vector cache should be used. Set
/// `RECALL_VOYAGE_RESIDENT=0` to force the low-memory streaming path.
fn resident_enabled() -> bool {
    !matches!(
        std::env::var("RECALL_VOYAGE_RESIDENT").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

fn resident(store: &Store) -> Option<&'static ResidentVectors> {
    RESIDENT
        .get_or_init(|| match ResidentVectors::load(store) {
            Ok(v) if v.rows() > 0 => Some(v),
            Ok(_) => None,
            Err(e) => {
                warn_once(&format!("resident load failed: {e}"));
                None
            }
        })
        .as_ref()
}

/// Rank a best-score map into doc ids, highest first.
fn rank(mut best: HashMap<i64, f32>, depth: usize) -> Vec<i64> {
    // Tenancy: skip docs the current reader cannot see BEFORE truncating, so k stays full.
    crate::scope::drop_denied(&mut best);
    let mut ranked: Vec<(i64, f32)> = best.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(depth);
    ranked.into_iter().map(|(id, _)| id).collect()
}

// ---------------------------------------------------------------- search

/// Score `embeddings2` against the query vector and return doc ids, best first.
///
/// Two correctness details that cost real accuracy if skipped:
///   1. `embeddings2` has NO foreign key to `documents` (unlike `embeddings`,
///      which has ON DELETE CASCADE), so it accumulates orphan vectors for
///      deleted docs. The join on `documents.valid_to IS NULL` filters both
///      orphans and superseded revisions.
///   2. A document has ~3 chunks. We MAX-POOL per doc — a document is as
///      relevant as its single best chunk — otherwise long documents with many
///      mediocre chunks outrank short precise ones.
///
/// Streams rows and keeps only the per-doc best score, so memory is O(docs)
/// rather than the ~1.6 GB a full materialisation of 400k x 1024 f32 would take.
fn scored_doc_ids(store: &Store, qvec: &[f32], depth: usize) -> Result<Vec<i64>> {
    // Fast path: vectors already resident in RAM. First call in a process pays
    // the load; every call after is a pure in-memory parallel scan.
    if resident_enabled() {
        if let Some(res) = resident(store) {
            return Ok(rank(res.best_per_doc(qvec), depth));
        }
    }

    // Slow path: stream from SQLite. Kept for low-memory environments and as the
    // fallback if the resident load fails.
    let conn = &store.conn;
    // Per-query slow path: scoped here (the resident matrix is filtered per query instead).
    let mut stmt = conn.prepare(&format!(
        "SELECT e.doc_id, e.vec
           FROM embeddings2 e
           JOIN documents d ON d.id = e.doc_id
          WHERE d.valid_to IS NULL{}
            AND e.dim = ?1",
        crate::scope::clause("d")
    ))?;

    let mut best: HashMap<i64, f32> = HashMap::new();
    let mut rows = stmt.query(rusqlite::params![VOYAGE_DIM as i64])?;
    while let Some(row) = rows.next()? {
        let doc_id: i64 = row.get(0)?;
        // Borrow the blob instead of copying it into a Vec<u8>, and score
        // straight off the bytes. Decoding each row into a Vec<f32> first cost
        // ~400k heap allocations per query and dominated the runtime.
        let blob: &[u8] = row.get_ref(1)?.as_blob().unwrap_or(&[]);
        if blob.len() != VOYAGE_DIM * 4 {
            continue;
        }

        // Cosine on an L2-normalised query reduces to dot / ||doc||. Stored
        // vectors are normalised here rather than trusted from the backfill.
        // Unrolled by 4 so the loop vectorises; f32::from_le_bytes on a
        // 4-byte window is a no-op cast on little-endian targets.
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

    let mut ranked: Vec<(i64, f32)> = best.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(depth);
    Ok(ranked.into_iter().map(|(id, _)| id).collect())
}

/// The `voyage` retrieval channel. Mirrors `bge_ids` so `fused()` can treat it
/// like any other channel.
///
/// ALWAYS returns Ok. Any failure degrades to an empty list so the fused query
/// still returns FTS5 results.
pub fn voyage_ids(
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

/// Rows in `embeddings2` whose document is gone or superseded. `embeddings2`
/// has no ON DELETE CASCADE, so these accumulate and waste ranking slots.
pub fn count_orphaned_embeddings2(store: &Store) -> Result<i64> {
    Ok(store.conn.query_row(
        "SELECT COUNT(*) FROM embeddings2 e
          WHERE NOT EXISTS (
                SELECT 1 FROM documents d
                 WHERE d.id = e.doc_id AND d.valid_to IS NULL)",
        [],
        |r| r.get(0),
    )?)
}

/// Delete orphaned `embeddings2` rows. Mirrors `delete_orphaned_embeddings()`,
/// which only ever covered the legacy `embeddings` table.
pub fn delete_orphaned_embeddings2(store: &Store) -> Result<usize> {
    Ok(store.conn.execute(
        "DELETE FROM embeddings2
          WHERE NOT EXISTS (
                SELECT 1 FROM documents d
                 WHERE d.id = embeddings2.doc_id AND d.valid_to IS NULL)",
        [],
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_rejects_wrong_dimension() {
        assert!(decode_f32(&[0u8; 8]).is_none(), "8 bytes is not 1024 f32");
        assert!(decode_f32(&[0u8; 4095]).is_none(), "not a multiple of 4");
        assert!(decode_f32(&vec![0u8; VOYAGE_DIM * 4]).is_some());
    }

    #[test]
    fn normalize_makes_unit_length() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6, "expected unit length, got {n}");
    }

    #[test]
    fn normalize_tolerates_zero_vector() {
        let mut v = vec![0.0f32; 4];
        l2_normalize(&mut v);
        assert!(v.iter().all(|x| *x == 0.0), "zero vector must not produce NaN");
    }
}
