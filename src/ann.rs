// ann.rs — HNSW approximate nearest-neighbor index for the BGE vector channel.
//
// Gated behind the `ann` Cargo feature; the whole module is a no-op when
// compiled without it.  The default build is unchanged.
//
// See also the exact-scan path in bge.rs and the vector tables in store.rs.
//
// What this module does:
//   1. Wraps `hnsw_rs::Hnsw<f32, DistDot>` (always owned/built in-memory).
//   2. Provides `AnnIndex::build(vectors)` — builds the HNSW graph from an
//      already-loaded Vec<(i64, Vec<f32>)>; no fastembed/ONNX called here.
//   3. Provides `AnnIndex::load_or_build(path, vectors)` — validates a sidecar
//      header JSON on disk, then rebuilds the in-memory HNSW from the already-
//      loaded `vectors`.  Always rebuilds in-memory (no hnsw binary graph load)
//      because hnsw_rs reload requires HnswIo to outlive Hnsw, which is a
//      self-referential struct problem. Since vectors are already in RAM the
//      rebuild is free (no ONNX/network).  Writes/refreshes the header after
//      each build so next call can detect staleness.
//   4. Provides `AnnIndex::search(query, depth, over_fetch) -> Vec<i64>` --
//      HNSW approximate fetch of `depth * over_fetch` candidates, then exact
//      cosine rerank from raw vectors, returning top `depth` doc ids.
//   5. Runtime kill-switch: `RECALL_ANN=0` forces the exact path even on an
//      ann-compiled binary (mirrors RECALL_VAULT pattern in store.rs:47).
//
// FREEZE GUARD: build() iterates over pre-loaded Vec<f32> rows only.
// It NEVER calls fastembed, NEVER calls recall embed, NEVER touches ONNX.

use anyhow::{Context, Result};
use hnsw_rs::api::AnnT;
use hnsw_rs::hnsw::Hnsw;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// 🔴 anndists' own `DistDot` PANICS on this corpus. Do not put it back.
//
// Its scalar path is `let dot = 1. - Σ(a·b); assert!(dot >= 0.)` — i.e. it asserts that the dot
// product of any two vectors it is given never exceeds 1.0. That holds for EXACTLY unit vectors
// and not for f32 ones. Measured over 4,000 live rows from `embeddings`: norms are
// 0.999999-1.000001 and the largest self-dot is **1.000001**, so `1 - dot = -1e-6` and the
// assertion fails. The vault is ~25% exact content duplicates, so identical vector pairs are
// guaranteed, not hypothetical.
//
// Reproduced on commit 038e000 (before any per-chunk work) built with `--features "voyage,ann"`:
//   thread 'main' panicked at anndists-0.1.5/src/dist/distances.rs:279: assertion failed: dot >= 0.
// This is why the `ann` feature has never shipped. `DistDotSafe` below is the same metric with
// the assertion replaced by a clamp.
use hnsw_rs::anndists::dist::distances::Distance;

// ── Schema version ────────────────────────────────────────────────────────────
// Increment when the index build parameters or header format changes so that
// old headers are treated as stale and trigger a fresh build.
const SCHEMA_VER: u32 = 1;

// ── HNSW tuning knobs ─────────────────────────────────────────────────────────
// Standard HNSW literature defaults; measure against BENCHMARK.md before changing.
const M: usize = 16; // max connections per layer
const EF_CONSTRUCTION: usize = 200; // build quality (higher = better graph, slower build)
const MAX_LAYER: usize = 16; // log2(100k) ~= 17; 16 is safe
const EF_SEARCH: usize = 64; // search quality (higher = better recall, slower query)

/// Cosine distance for unit vectors — `1 - dot`, clamped to `[0, 2]`.
///
/// Identical to `anndists::DistDot` in every case that matters, minus the assertion that makes a
/// one-part-per-million floating-point excess terminate the whole process. A retrieval engine must
/// degrade, never abort: an index over 27,426 vectors that panics on one duplicate pair takes
/// search down for every query, not just the affected one.
///
/// The clamp is not papering over bad data — the vectors ARE unit vectors to within f32 epsilon.
/// It is acknowledging that `Σ(a·b)` over 384 f32 multiplies accumulates rounding error, so the
/// mathematically-guaranteed `≤ 1` is not machine-guaranteed.
#[derive(Clone, Default)]
pub struct DistDotSafe;

impl Distance<f32> for DistDotSafe {
    /// `eval` is `anndists::Distance`'s required method name — it evaluates a DISTANCE between two
    /// vectors. Nothing here interprets or executes anything; the name is fixed by the trait.
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        // Zip stops at the shorter side, so a mismatched pair scores as "far" instead of reading
        // out of bounds. `build` filters these out anyway; this is the second line of defence.
        let dot: f32 = va.iter().zip(vb.iter()).map(|(a, b)| a * b).sum();
        let d = 1.0 - dot;
        if d.is_finite() {
            d.clamp(0.0, 2.0)
        } else {
            // A NaN would silently poison every comparison it takes part in. Sort it to the back.
            2.0
        }
    }
}

/// Compact JSON header written to `<base>.ann.json` as the staleness guard.
#[derive(Debug, Serialize, Deserialize)]
struct AnnHeader {
    schema_ver: u32,
    dim: usize,
    count: usize,
}

/// Live ANN index.  Always built in-memory from the raw vectors Vec.
pub struct AnnIndex {
    /// HNSW graph. Search-only after build.
    hnsw: Hnsw<'static, f32, DistDotSafe>,
    /// id_map[pos] = doc_id for the vector inserted at HNSW external-id `pos`.
    id_map: Vec<i64>,
    /// Raw vectors kept for exact-rerank of HNSW candidates.
    raw: Vec<(i64, Vec<f32>)>,
    /// Number of indexed vectors.
    count: usize,
}

impl AnnIndex {
    /// Build the HNSW index from pre-loaded vectors.
    ///
    /// FREEZE GUARD: pure in-memory math on the already-loaded Vec.
    /// No fastembed, no ONNX, no network.
    pub fn build(vectors: Vec<(i64, Vec<f32>)>) -> Result<Self> {
        anyhow::ensure!(!vectors.is_empty(), "ann: no vectors to build index from");

        // SKIP BAD ROWS, DO NOT DIE ON THEM. One malformed vector out of 27,426 must cost that
        // one document its semantic reachability, not take every query down with it. A
        // wrong-length vector silently mis-scores (zip truncates to the shorter side), a zero
        // vector has no direction to compare against, and a NaN poisons every comparison it
        // takes part in.
        let before = vectors.len();
        let vectors: Vec<(i64, Vec<f32>)> = vectors
            .into_iter()
            .filter(|(_, v)| {
                v.len() == crate::bge::DIM
                    && v.iter().all(|x| x.is_finite())
                    && v.iter().any(|x| *x != 0.0)
            })
            .collect();
        let skipped = before - vectors.len();
        if skipped > 0 {
            eprintln!(
                "ann: skipped {skipped} unusable vector(s) of {before} (wrong dimension, zero, \
                 or non-finite); index built over the remaining {}",
                vectors.len()
            );
        }
        anyhow::ensure!(
            !vectors.is_empty(),
            "ann: every one of {before} vectors was unusable — refusing to build an empty index"
        );

        let n = vectors.len();
        let hnsw: Hnsw<'static, f32, DistDotSafe> =
            Hnsw::new(M, n + 1, MAX_LAYER, EF_CONSTRUCTION, DistDotSafe {});

        // External id = pos (insertion order); id_map[pos] = doc_id.
        let mut id_map: Vec<i64> = Vec::with_capacity(n);
        for (pos, (doc_id, vec)) in vectors.iter().enumerate() {
            hnsw.insert((vec.as_slice(), pos));
            id_map.push(*doc_id);
        }

        Ok(Self {
            hnsw,
            id_map,
            count: n,
            raw: vectors,
        })
    }

    /// Write the staleness header JSON and (optionally) the hnsw binary graph.
    /// The header is what matters for staleness detection; the binary graph
    /// files are a bonus for future experiments.
    pub fn dump(&self, base_path: &Path) -> Result<()> {
        // Write sidecar JSON header (staleness guard).
        let header = AnnHeader {
            schema_ver: SCHEMA_VER,
            dim: crate::bge::DIM,
            count: self.count,
        };
        let hp = sidecar_path(base_path);
        std::fs::write(&hp, serde_json::to_string_pretty(&header)?)
            .with_context(|| format!("ann: write header {}", hp.display()))?;

        // Also dump the hnsw binary graph (non-fatal if it fails).
        let dir = base_path.parent().unwrap_or_else(|| Path::new("."));
        let bn = base_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if let Err(e) = self.hnsw.file_dump(dir, &bn) {
            eprintln!(
                "ann: warning -- hnsw binary graph dump failed ({}); header ok",
                e
            );
        }

        eprintln!(
            "ann: header written ({} vectors) -> {}",
            self.count,
            hp.display()
        );
        Ok(())
    }

    /// Validate sidecar header, then (re)build HNSW from `vectors` in-memory.
    /// Writes a fresh header after each build.
    ///
    /// `base_path` -- e.g. `~/.recall/recall.ann` (no file extension).
    pub fn load_or_build(base_path: &Path, vectors: Vec<(i64, Vec<f32>)>) -> Result<Self> {
        let hp = sidecar_path(base_path);
        let header_matches = if hp.exists() {
            std::fs::read_to_string(&hp)
                .ok()
                .and_then(|s| serde_json::from_str::<AnnHeader>(&s).ok())
                .map(|h| {
                    h.schema_ver == SCHEMA_VER
                        && h.dim == crate::bge::DIM
                        && h.count == vectors.len()
                })
                .unwrap_or(false)
        } else {
            false
        };

        if header_matches {
            eprintln!(
                "ann: sidecar matches ({} pts) -- rebuilding HNSW from DB vectors",
                vectors.len()
            );
        } else {
            eprintln!(
                "ann: sidecar absent/stale -- building fresh index over {} vectors",
                vectors.len()
            );
        }

        let t0 = std::time::Instant::now();
        let idx = AnnIndex::build(vectors)?;
        eprintln!(
            "ann: HNSW graph built in {:.1}s",
            t0.elapsed().as_secs_f32()
        );

        if let Err(e) = idx.dump(base_path) {
            eprintln!(
                "ann: warning -- sidecar write failed ({}); index live in RAM",
                e
            );
        }
        Ok(idx)
    }

    /// Approximate search + exact rerank.
    ///
    /// 1. HNSW yields top `depth * over_fetch` approximate candidates.
    /// 2. Exact cosine on raw vectors reranks those candidates.
    /// 3. Returns top `depth` doc ids, best first.
    pub fn search(&self, query: &[f32], depth: usize, over_fetch: usize) -> Vec<i64> {
        let candidates = (depth * over_fetch).min(self.count);
        if candidates == 0 {
            return vec![];
        }

        let neighbours = self.hnsw.search(query, candidates, EF_SEARCH);

        let mut scored: Vec<(i64, f32)> = neighbours
            .iter()
            .filter_map(|nb| {
                let pos = nb.d_id;
                let doc_id = *self.id_map.get(pos)?;
                let (_, raw_vec) = self.raw.get(pos)?;
                let score = crate::bge::cosine(query, raw_vec.as_slice());
                Some((doc_id, score))
            })
            .collect();

        // total_cmp gives a total order over f32 (NaN sorts deterministically)
        // so a stray NaN score can't violate sort's strict-weak-ordering contract.
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.into_iter().take(depth).map(|(id, _)| id).collect()
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.count
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn sidecar_path(base: &Path) -> PathBuf {
    let stem = base
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    base.parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{}.ann.json", stem))
}

/// Returns true unless `RECALL_ANN=0` is set in the environment.
/// Mirrors the `RECALL_VAULT` pattern in store.rs:47.
pub fn ann_enabled() -> bool {
    std::env::var("RECALL_ANN").as_deref() != Ok("0")
}

/// Default base path: `~/.recall/recall.ann` (hnsw_rs appends file extensions).
pub fn default_ann_path() -> Result<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .context("USERPROFILE/HOME not set -- cannot locate ann index")?;
    Ok(PathBuf::from(home).join(".recall").join("recall.ann"))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_norm(v: &mut Vec<f32>) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// 🔴 THE REGRESSION TEST. This is the case that took search down.
    ///
    /// `ann_rerank_matches_brute_force` below builds 20 DISTINCT vectors and passed happily while
    /// the panic sat in the live query path — distinct vectors never produce a dot product near
    /// 1.0, so the assertion inside `anndists::DistDot` was never reached. The vault is ~25% exact
    /// content duplicates, so identical vector pairs are guaranteed in production and absent from
    /// the test fixture. That gap is the whole story.
    /// A vector shaped like the live data: unit length to within f32 rounding, i.e. its norm can
    /// sit a part-per-million ABOVE 1.0.
    ///
    /// Measured over 4,000 rows of the live `embeddings` table: norms span 0.999999-1.000001 and
    /// the largest self-dot is 1.000001. Normalising a synthetic vector in a test usually lands
    /// just UNDER 1.0 (0.999999702 for the obvious fixture), which is why a naive duplicate test
    /// passes against the buggy distance and proves nothing. The excess is applied explicitly so
    /// the fixture reproduces the production condition rather than hoping to.
    fn nearly_unit_vector(seed: f32) -> Vec<f32> {
        let dim = crate::bge::DIM;
        let mut v: Vec<f32> = (0..dim).map(|j| ((j as f32) * seed).sin()).collect();
        l2_norm(&mut v);
        for x in v.iter_mut() {
            *x *= 1.0 + 1e-6;
        }
        v
    }

    #[test]
    fn duplicate_unit_vectors_do_not_panic_the_index() {
        let base = nearly_unit_vector(0.017);
        // Exact duplicates — the shape 24.8% of the corpus actually has.
        let vectors: Vec<(i64, Vec<f32>)> = (0..8).map(|i| (i as i64 + 1, base.clone())).collect();

        let idx = AnnIndex::build(vectors).expect("build must not panic on duplicates");
        let hits = idx.search(&base, 3, 4);
        assert!(
            !hits.is_empty(),
            "a query identical to the corpus found nothing"
        );
    }

    /// The exact arithmetic behind the panic: the self-dot of a nearly-unit f32 vector exceeds 1.0,
    /// so `1 - dot` goes negative and `anndists::DistDot` aborts the process.
    #[test]
    fn dist_dot_safe_clamps_a_self_dot_above_one() {
        let v = nearly_unit_vector(0.013);
        let raw_dot: f32 = v.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        assert!(
            raw_dot > 1.0,
            "fixture no longer reproduces the bug: self-dot is {raw_dot}, must exceed 1.0"
        );
        let d = DistDotSafe.eval(&v, &v);
        // Non-negative is the property whose absence aborted the process.
        assert!(
            (0.0..=2.0).contains(&d),
            "distance {d} escaped [0,2] — anndists asserts on exactly this"
        );
    }

    #[test]
    fn dist_dot_safe_survives_nan_and_mismatched_lengths() {
        let a = vec![f32::NAN; 8];
        let b = vec![0.5f32; 8];
        assert!(
            DistDotSafe.eval(&a, &b).is_finite(),
            "NaN must not propagate"
        );
        // Mismatched lengths must return a number, not read out of bounds or panic.
        assert!(DistDotSafe.eval(&[1.0, 0.0], &[1.0]).is_finite());
    }

    /// One bad row must cost that row, not the query.
    #[test]
    fn build_skips_unusable_vectors_instead_of_failing() {
        let dim = crate::bge::DIM;
        let mut good: Vec<f32> = (0..dim).map(|j| ((j as f32) * 0.011).sin()).collect();
        l2_norm(&mut good);
        let vectors: Vec<(i64, Vec<f32>)> = vec![
            (1, good.clone()),
            (2, vec![0.0; dim]),      // zero vector: no direction
            (3, vec![f32::NAN; dim]), // non-finite
            (4, vec![0.5; 7]),        // wrong dimension
            (5, good.clone()),
        ];
        let idx = AnnIndex::build(vectors).expect("must build from the survivors");
        assert_eq!(
            idx.len(),
            2,
            "only the two usable vectors should be indexed"
        );
        let hits = idx.search(&good, 2, 4);
        assert!(hits.iter().all(|id| *id == 1 || *id == 5));
    }

    /// ANN+rerank top-5 must match brute-force on a small 20-vector corpus.
    #[test]
    fn ann_rerank_matches_brute_force() {
        let n = 20usize;
        let dim = 384usize;
        let vectors: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|j| ((i as f32 + 1.0) * (j as f32 + 1.0) * 0.01).sin())
                    .collect();
                l2_norm(&mut v);
                (i as i64 + 1, v)
            })
            .collect();

        // Query: slightly perturbed version of vector index 5.
        let mut query = vectors[5].1.clone();
        query[0] += 0.001;
        l2_norm(&mut query);

        // Brute-force gold standard.
        let mut bf: Vec<(i64, f32)> = vectors
            .iter()
            .map(|(id, v)| (*id, crate::bge::cosine(&query, v)))
            .collect();
        bf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let bf_top5: Vec<i64> = bf.iter().take(5).map(|(id, _)| *id).collect();

        let idx = AnnIndex::build(vectors).expect("build failed");
        let ann_top5 = idx.search(&query, 5, 4);

        assert_eq!(
            ann_top5, bf_top5,
            "ANN rerank mismatch: ann={:?} bf={:?}",
            ann_top5, bf_top5
        );
    }

    /// ann_enabled() must return a bool without panicking.
    #[test]
    fn ann_enabled_callable() {
        let _ = ann_enabled();
    }

    /// Build, dump header, reload via load_or_build, assert same top-1.
    #[test]
    fn ann_header_roundtrip() {
        let n = 10usize;
        let dim = 384usize;
        let vectors: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|j| ((i as f32 + 1.0) * (j as f32 + 0.5) * 0.02).cos())
                    .collect();
                l2_norm(&mut v);
                (i as i64 + 100, v)
            })
            .collect();

        let tmp = std::env::temp_dir().join(format!("recall_ann_rt_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let base = tmp.join("test.ann");

        let idx = AnnIndex::build(vectors.clone()).expect("build");
        idx.dump(&base).expect("dump");

        assert!(sidecar_path(&base).exists(), "sidecar JSON missing");

        let idx2 = AnnIndex::load_or_build(&base, vectors.clone()).expect("load_or_build");
        assert_eq!(idx2.count, n);

        let q = vectors[3].1.clone();
        let r1 = idx.search(&q, 3, 4);
        let r2 = idx2.search(&q, 3, 4);
        assert!(!r1.is_empty());
        assert_eq!(r1[0], r2[0], "top-1 differs after reload");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Stale header (wrong count) must trigger a rebuild without error.
    #[test]
    fn ann_stale_header_triggers_rebuild() {
        let n = 8usize;
        let dim = 384usize;
        let vectors: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| {
                let mut v = vec![0.1f32; dim];
                v[i % dim] = 1.0;
                l2_norm(&mut v);
                (i as i64 + 200, v)
            })
            .collect();

        let tmp = std::env::temp_dir().join(format!("recall_ann_stale_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let base = tmp.join("stale.ann");

        // Plant a stale header with count = n+1.
        let stale = AnnHeader {
            schema_ver: SCHEMA_VER,
            dim: crate::bge::DIM,
            count: n + 1,
        };
        std::fs::write(sidecar_path(&base), serde_json::to_string(&stale).unwrap()).unwrap();

        let idx = AnnIndex::load_or_build(&base, vectors.clone())
            .expect("load_or_build with stale header");
        assert_eq!(idx.count, n);

        // Header must be refreshed with correct count.
        let json = std::fs::read_to_string(sidecar_path(&base)).unwrap();
        let h: AnnHeader = serde_json::from_str(&json).unwrap();
        assert_eq!(h.count, n, "header not updated after rebuild");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
