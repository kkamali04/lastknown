// bge.rs — in-process dense embeddings (BAAI/bge-small-en-v1.5, 384-dim).
//
// This is the "meaning-based" recall channel: it turns text into a vector so we
// can find notes that mean the same thing even when they share no keywords —
// exactly the misses that keyword (FTS5) search produces.
//
// Fully offline. fastembed reads model bytes directly from the local hf-hub
// snapshot cache (~/.recall/models) via try_new_from_user_defined — hf-hub,
// ureq, native-tls, and reqwest are NOT compiled in. The binary is incapable
// of any outbound TLS connection for model download (ZERO-CLOUD).
//
// CPU-only — no GPU, no cloud API (hard constraint: NO Voyage, NO OpenAI).
//
// WP2 (ann feature): when compiled with --features ann AND recall.ann exists
// on disk AND its header matches the current vector count, search_ids routes
// through ANN+exact-rerank instead of the O(N) brute-force loop.  The
// `vectors` field is kept regardless — it is the exact-rerank truth source.

use anyhow::{Context, Result};
use fastembed::{
    InitOptionsUserDefined, Pooling, QuantizationMode, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use ort::ep::{ExecutionProvider, ExecutionProviderDispatch};
use std::sync::Mutex;
use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};

/// Which ONNX Runtime execution provider to ask for, from `RECALL_BACKEND`.
///
/// `cpu` (the default) · `dml` · `auto`.
///
/// 🔴 The trap this exists to close: `ExecutionProviderDispatch` defaults to
/// `error_on_failure = false`, and ort documents that as *"silently fail and fall back to the next
/// execution provider, or the CPU provider if no registrations succeed"*. So the naive wiring —
/// hand it a GPU provider and hope — produces a run that succeeds, exits 0, is ~10x slower, and
/// says nothing. That is the same fail-open shape as the Vertex 403 and the keyword fallback.
///
/// Two guards, and BOTH are needed:
///   * `is_available()` asks the LOADED onnxruntime.dll (via `GetAvailableProviders`) whether the
///     provider is actually compiled into it. A CPU-only DLL answers false — which is the only
///     reliable way to tell, since a CPU-only build still contains the provider NAME strings.
///   * `.error_on_failure()` makes registration failure an error instead of a log line, for the
///     case where the provider exists but cannot initialise (no compatible device, driver, …).
///
/// `dml` REFUSES if unavailable. Only `auto` may fall back, and it announces the choice.
pub fn resolve_execution_providers() -> Result<(Vec<ExecutionProviderDispatch>, String)> {
    let dll = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| "<unset>".into());
    let want = std::env::var("RECALL_BACKEND")
        .unwrap_or_else(|_| "cpu".into())
        .to_ascii_lowercase();

    // Asking the runtime is the only honest answer. Errors here mean the DLL could not be probed
    // at all, which is itself a "no".
    let dml_available = ort::ep::DirectML::default().is_available().unwrap_or(false);

    match want.as_str() {
        "cpu" => Ok((vec![cpu_ep()], format!("cpu (requested, arena=off; dll={dll})"))),
        "dml" | "directml" => {
            if !dml_available {
                anyhow::bail!(
                    "REFUSING: RECALL_BACKEND=dml but the loaded ONNX Runtime has no DirectML \
                     execution provider.\n  dll = {dll}\n\
                     That DLL is a CPU-only build. Point ORT_DYLIB_PATH at a DirectML build, or \
                     drop it beside recall.exe, then re-run.\n\
                     Refusing rather than falling back: a silent CPU run here is ~10x slower and \
                     indistinguishable from success."
                );
            }
            Ok((
                vec![ort::ep::DirectML::default().build().error_on_failure()],
                format!("directml (requested, available, error_on_failure=true; dll={dll})"),
            ))
        }
        "auto" => {
            if dml_available {
                Ok((
                    vec![ort::ep::DirectML::default().build().error_on_failure()],
                    format!("directml (auto: available; dll={dll})"),
                ))
            } else {
                Ok((
                    vec![cpu_ep()],
                    format!("cpu (auto: DirectML not in this runtime, arena=off; dll={dll})"),
                ))
            }
        }
        other => anyhow::bail!(
            "REFUSING: unknown RECALL_BACKEND={other:?}. Use cpu | dml | auto."
        ),
    }
}

/// The CPU provider with ONNX Runtime's CPU memory arena OFF. With no provider passed, ORT keeps
/// its default arena, which grows to the largest batch's attention scratch and never shrinks:
/// measured 4,678 MB commit for `embed --batch 64` on 500 SciFact docs, 18.6 GB at 192 rows
/// Off, memory returns to the ~0.65 GB model
/// baseline after every batch. An allocator setting: the vectors are unchanged.
pub(crate) fn cpu_ep() -> ExecutionProviderDispatch {
    ort::ep::CPU::default().with_arena_allocator(false).build()
}

/// Embedding dimensionality for bge-small-en-v1.5.
pub const DIM: usize = 384;
const MIN_ORT_MAJOR: u64 = 1;
const MIN_ORT_MINOR: u64 = 24;

/// Holds the loaded model. Construct once (load is expensive) and reuse.
///
/// fastembed's `embed` needs `&mut self`; we wrap it in a `Mutex` so callers can keep a shared
/// `&Embedder`, from several daemon request threads at once (one embed call at a time).
/// Most chunk rows in one ONNX call, for every caller of [`Embedder::embed`] (both embed queues
/// cap their batches at this too). 64 since 2026-10-31: with the ORT arena off ([`cpu_ep`]) a
/// 64-row worst-case call peaks near 2.5 GB vs 6.3 GB at 192 (measured).
pub const MAX_ROWS_PER_ONNX_CALL: usize = 64;

pub struct Embedder {
    model: Mutex<TextEmbedding>,
}

impl Embedder {
    /// Load the model from the local hf-hub snapshot cache, purely offline.
    ///
    /// Uses `try_new_from_user_defined` to read tokenizer + ONNX bytes directly
    /// from disk. hf-hub, ureq, native-tls, and reqwest are NOT compiled in —
    /// this is enforced at the Cargo.toml level, not just suppressed at runtime.
    ///
    /// The snapshot directory is located by reading refs/main (the commit hash
    /// written by the one-time model fetch) and joining it under the snapshots/
    /// subdirectory of the local hf-hub cache.
    pub fn new() -> Result<Self> {
        let cache = cache_dir()?;
        let snap = cache
            .join("models--Xenova--bge-small-en-v1.5")
            .join("snapshots")
            .join(snapshot_hash(&cache)?);

        let read = |name: &str| -> Result<Vec<u8>> {
            let bytes = std::fs::read(snap.join(name))
                .with_context(|| format!("model file missing from local cache: {name}"))?;
            if name.ends_with(".onnx") && bytes.len() > 200 * 1024 * 1024 {
                eprintln!(
                    "bge: model file '{}' is unusually large ({:.1} MB); proceeding",
                    name,
                    bytes.len() as f64 / (1024.0 * 1024.0)
                );
            }
            Ok(bytes)
        };

        let model = UserDefinedEmbeddingModel::new(
            read("onnx/model.onnx")?,
            TokenizerFiles {
                tokenizer_file: read("tokenizer.json")?,
                config_file: read("config.json")?,
                special_tokens_map_file: read("special_tokens_map.json")?,
                tokenizer_config_file: read("tokenizer_config.json")?,
            },
        )
        .with_pooling(Pooling::Cls)
        .with_quantization(QuantizationMode::None);

        let (providers, resolved) = resolve_execution_providers()?;
        chatter(&format!("backend={resolved}"));
        let te = TextEmbedding::try_new_from_user_defined(
            model,
            InitOptionsUserDefined::new().with_execution_providers(providers),
        )
        .context("load bge-small-en-v1.5 from local snapshot (user-defined)")?;

        Ok(Self {
            model: Mutex::new(te),
        })
    }

    /// Embed a batch of documents. Returns one 384-dim vector per input.
    ///
    /// fastembed runs `texts` in ONNX calls of at most [`MAX_ROWS_PER_ONNX_CALL`] rows. v3 r3
    /// (Codex r2 MEDIUM #4): `None` here meant fastembed's default 256-row call, so one document
    /// of 200 chunks (the embed loops always take the first document whole) bypassed the 64-row
    /// memory ceiling. Any `texts` of <= 64 rows is still exactly one call, so vectors are unchanged.
    ///
    /// v3 r4 (Codex r3 MEDIUM #3): fastembed keeps EVERY sub-batch's raw ONNX output
    /// (`last_hidden_state[batch, seq, 384]`) until it pools, so one call over a 2,000-chunk document
    /// still held all of them at once. Each 64-row slice is now its own fastembed call and only its
    /// pooled 384-float vectors are kept. Same 64-row batches as before, so the vectors are identical.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for slice in texts.chunks(MAX_ROWS_PER_ONNX_CALL) {
            let refs: Vec<&str> = slice.iter().map(|s| s.as_str()).collect();
            let pooled = self
                .model
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .embed(refs, Some(MAX_ROWS_PER_ONNX_CALL))
                .context("bge embed batch")?;
            out.extend(pooled);
        }
        Ok(out)
    }

    /// Embed a single query string.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.embed(std::slice::from_ref(&text.to_string()))?;
        Ok(v.pop().unwrap_or_default())
    }
}

/// A ready-to-query meaning-search channel: the loaded model plus every stored
/// document vector in RAM. Build once per process (model load + matrix load are
/// both expensive) and reuse across all queries — this is what keeps `bench`
/// from reloading the model 45 times.
///
/// WP2: when compiled with `--features ann`, also holds an optional HNSW index
/// that makes `search_ids` O(log N) instead of O(N).  The `vectors` field is
/// kept regardless — it is the exact-rerank truth source and fallback path.
pub struct BgeSearch {
    pub(crate) embedder: Embedder,
    /// One entry per stored vector. Since the per-chunk table landed a doc_id appears
    /// ONCE PER CHUNK, so this is not a per-document list and must not be treated as
    /// one — `best_per_doc` collapses it to one score per document.
    pub(crate) vectors: Vec<(i64, Vec<f32>)>,
    /// ANN index — present only when compiled with `ann` feature AND the index
    /// was successfully loaded/built at `BgeSearch::load` time.
    #[cfg(feature = "ann")]
    pub(crate) ann: Option<crate::ann::AnnIndex>,
}

impl BgeSearch {
    /// Load the model and pull all stored vectors out of the store. Returns
    /// `Ok(None)` when no vectors exist yet (so callers can skip the channel
    /// without treating an un-embedded store as an error).
    ///
    /// WP2: when `ann` feature is on, attempts to load/build the HNSW index
    /// from `~/.recall/recall.ann`.  Failures are non-fatal: the brute-force
    /// path remains available.
    pub fn load(store: &crate::store::Store) -> Result<Option<Self>> {
        // Freeze-safe guard, and it REFUSES: ort is built load-dynamic, so it needs
        // onnxruntime.dll at the pinned ORT_DYLIB_PATH (set in main() to the beside-exe
        // location). Without it ort would fall back to the OS DLL search and load the stale
        // System32 onnxruntime.dll, which WEDGES at session init.
        //
        // 🔴 This used to print a warning and return Ok(None) — AFTER a ~17 s vector load — and
        // the fts5 rescue tier then served keyword hits under the bge name with exit 0 (E2E 2c,
        // 2026-10-29). Checked first now, and an error: callers that want the old degrade pass
        // `--allow-fallback` and say so on stderr.
        if let Err(e) = ort_runtime_preflight() {
            anyhow::bail!(
                "bge: ONNX Runtime DLL not usable (ORT_DYLIB_PATH / beside recall.exe) — semantic \
                 channel unavailable: {e:#}"
            );
        }
        chatter("loading vectors from DB…");
        // UNION the two tables: per-chunk vectors where they exist, legacy pooled vectors for
        // every document that has not been re-embedded yet.
        //
        // 🔴 An earlier version preferred the chunk table whenever it was non-EMPTY and fell back
        // to pooled only when it had zero rows. That is only correct if the re-embed is atomic,
        // and it is not — it runs for hours. Observed live mid-run: only a few hundred documents had
        // chunk rows, so this matrix silently shrank from over 100k vectors to a few thousand and the
        // semantic channel covered 0.7% of the vault. Nothing looked wrong: the fts5 rescue tier
        // kept returning plausible hits (skill `failopen-fallback-masks-the-real-channel`).
        //
        // Unioning makes the rollout monotonic — coverage never drops, and each document upgrades
        // from pooled to per-chunk the moment it is re-embedded.
        let mut vectors = store.load_all_chunk_embeddings()?;
        let chunk_rows = vectors.len();
        let pooled = store.load_pooled_embeddings_without_chunks()?;
        let pooled_rows = pooled.len();
        vectors.extend(pooled);
        // The matrix holds every scope; its size is the operator's business only (Codex tenancy b1).
        if crate::scope::current().is_operator() {
            chatter(&format!(
                "loaded {} vectors ({chunk_rows} chunk + {pooled_rows} pooled-not-yet-rechunked)",
                vectors.len()
            ));
        }
        if vectors.is_empty() {
            eprintln!("bge: no vectors stored — run `recall embed`");
            return Ok(None);
        }
        chatter("initializing ONNX model…");
        let embedder = Embedder::new()?;
        chatter("model ready");

        // Build the optional ANN index when the `ann` feature is compiled in.
        // The `ann` cfg block and the `not(ann)` block are mutually exclusive at
        // compile time, so exactly one branch contributes the final Ok(Some(…)).
        #[cfg(feature = "ann")]
        let ann: Option<crate::ann::AnnIndex> = if crate::ann::ann_enabled() {
            match crate::ann::default_ann_path() {
                Ok(ann_path) => {
                    match crate::ann::AnnIndex::load_or_build(&ann_path, vectors.clone()) {
                        Ok(idx) => {
                            chatter(&format!("ANN index ready ({} pts)", idx.len()));
                            Some(idx)
                        }
                        Err(e) => {
                            eprintln!(
                                "bge: ANN index unavailable ({}); falling back to exact scan",
                                e
                            );
                            None
                        }
                    }
                }
                Err(e) => {
                    eprintln!("bge: ANN path error ({}); falling back to exact scan", e);
                    None
                }
            }
        } else {
            eprintln!("bge: RECALL_ANN=0 — exact scan forced");
            None
        };

        #[cfg(feature = "ann")]
        {
            Ok(Some(Self {
                embedder,
                vectors,
                ann,
            }))
        }

        #[cfg(not(feature = "ann"))]
        Ok(Some(Self { embedder, vectors }))
    }

    /// Swap these documents' vectors in the in-RAM matrix for what the database holds now.
    ///
    /// The warm daemon loads the matrix once; without this, a refresh that re-embeds an edited
    /// note would be invisible to semantic search until the daemon restarts. Documents that lost
    /// their vectors (tombstoned, excluded) simply drop out. Order inside `vectors` is not
    /// significant to `best_per_doc`.
    ///
    /// ⚠️ With `--features ann` the HNSW index is NOT patched; that feature is gated off.
    pub fn patch_docs(&mut self, ids: &[i64], fresh: Vec<(i64, Vec<f32>)>) {
        let set: std::collections::HashSet<i64> = ids.iter().copied().collect();
        self.vectors.retain(|(id, _)| !set.contains(id));
        self.vectors.extend(fresh);
    }

    /// Top-k doc ids by cosine similarity to `query`, best first.
    ///
    /// Dispatch logic:
    /// - `ann` feature ON + index loaded + RECALL_ANN != "0" → HNSW approximate
    ///   search + exact rerank (O(log N))
    /// - otherwise (feature off, index absent, env kill-switch) → brute-force
    ///   cosine scan (O(N)) — unchanged from before WP2
    pub fn search_ids(&self, query: &str, k: usize) -> Result<Vec<i64>> {
        Ok(self
            .search_scored(query, k)?
            .into_iter()
            .map(|(id, _)| id)
            .collect())
    }

    /// The vector a search query is ranked by: the query with the BGE query instruction
    /// ([`bge_query_text`]). The one place a query is embedded, for the active path and the oracle.
    fn query_vector(&self, query: &str) -> Result<Vec<f32>> {
        self.embedder.embed_one(&bge_query_text(query))
    }

    /// [`Self::search_ids`] with each document's best-chunk cosine, best first. The cosine is
    /// what the abstain floor (`query::BGE_MIN_TOP_COSINE`) is checked against.
    pub fn search_scored(&self, query: &str, k: usize) -> Result<Vec<(i64, f32)>> {
        let q = self.query_vector(query)?;
        if q.is_empty() {
            return Ok(vec![]);
        }

        #[cfg(feature = "ann")]
        if let Some(ref idx) = self.ann {
            if crate::ann::ann_enabled() {
                // over_fetch = 4: fetch 4× more candidates than needed so the
                // exact rerank can recover from HNSW approximate misses.
                //
                // ANN_CHUNK_HEADROOM is the second over-fetch, and it exists because
                // the index is built over CHUNK vectors: several returned ids can be
                // different chunks of the SAME document, and without headroom one long
                // note fills the whole top-k with itself. Ask for headroom×k ranked
                // ids, collapse duplicates keeping each document's best-ranked chunk,
                // then truncate — the same max-pool-per-doc rule `best_per_doc` applies
                // on the exact path.
                const ANN_CHUNK_HEADROOM: usize = 8;
                let mut seen = std::collections::HashSet::new();
                let ids: Vec<i64> = idx
                    .search(&q, k.saturating_mul(ANN_CHUNK_HEADROOM), 4)
                    .into_iter()
                    // Denied docs out BEFORE the cut, as on the exact path (CodeRabbit, tenancy PR).
                    .filter(|id| !crate::scope::is_denied(*id) && seen.insert(*id))
                    .take(k)
                    .collect();
                // Exact cosine for just the returned docs (dot only on their chunks).
                let keep: std::collections::HashSet<i64> = ids.iter().copied().collect();
                let mut best: HashMap<i64, f32> = HashMap::new();
                for (id, v) in self.vectors.iter().filter(|(id, _)| keep.contains(id)) {
                    let c = v.iter().zip(&q).map(|(a, b)| a * b).sum::<f32>();
                    let e = best.entry(*id).or_insert(f32::MIN);
                    *e = e.max(c);
                }
                return Ok(ids
                    .into_iter()
                    .map(|id| (id, best.get(&id).copied().unwrap_or(0.0)))
                    .collect());
            }
        }

        // Brute-force fallback (default when feature is off or ann is None).
        let mut per_doc = self.best_per_doc(&q);
        crate::scope::drop_denied(&mut per_doc);
        Ok(exact_scored(per_doc, k))
    }

    /// Exact top-k by brute-force cosine over every stored vector — the
    /// ground-truth ranking. O(N), single-threaded. Because it compares the
    /// query against every embedded doc, it cannot miss anything that is
    /// embedded: this is the 100%-recall reference the ANN path is graded
    /// against (see `crate::eval::recall_at_k`).
    ///
    /// It embeds the query exactly as the active path does ([`Self::query_vector`]). It used to
    /// embed the raw text with no query instruction, so on a store with more than k documents
    /// `recall eval` compared rankings for two different query vectors and scored below 1.0 with
    /// nothing wrong.
    pub fn exact_search_ids(&self, query: &str, k: usize) -> Result<Vec<i64>> {
        let q = self.query_vector(query)?;
        if q.is_empty() {
            return Ok(vec![]);
        }
        Ok(self.exact_search_vec(&q, k))
    }

    /// Brute-force cosine top-k for an already-embedded query vector.
    ///
    /// MAX-POOL PER DOC: a document is as relevant as its single best chunk. Ranking the
    /// raw chunk list instead would let one long note occupy every slot in the top-k with
    /// eight near-identical chunks of itself.
    pub(crate) fn exact_search_vec(&self, q: &[f32], k: usize) -> Vec<i64> {
        exact_scored(self.best_per_doc(q), k)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    /// Best (highest cosine) score per doc_id, over every chunk vector held in RAM.
    ///
    /// Mirrors `gemini::ResidentVectors::best_per_doc` / `voyage::ResidentVectors::best_per_doc`
    /// — the same max-pool-per-doc aggregation those channels have always used — adapted to
    /// this channel's `Vec<(i64, Vec<f32>)>` layout instead of a flat contiguous matrix.
    pub(crate) fn best_per_doc(&self, q: &[f32]) -> HashMap<i64, f32> {
        best_per_doc(&self.vectors, q)
    }
}

/// Per-doc scores sorted best first, top `k`.
pub(crate) fn exact_scored(per_doc: HashMap<i64, f32>, k: usize) -> Vec<(i64, f32)> {
    let mut scored: Vec<(i64, f32)> = per_doc.into_iter().collect();
    // Descending by score, then ASCENDING BY doc_id. The id tie-break is not
    // decoration: `best_per_doc` returns a HashMap, whose iteration order is
    // randomised per process, so without it two runs over the same store could rank
    // equal-scoring documents differently — and `load_all_chunk_embeddings` orders its
    // rows precisely so the eval oracle is reproducible. `total_cmp` rather than
    // `partial_cmp` so a NaN score cannot break sort's strict-weak-ordering contract.
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.truncate(k);
    scored
}

/// Max-pool per doc over a chunk-vector list — the free function behind
/// [`BgeSearch::best_per_doc`], split out so it is testable without loading the ONNX model.
///
/// Only the vector slice is passed to the worker threads, never `&BgeSearch`.
pub(crate) fn best_per_doc(vectors: &[(i64, Vec<f32>)], q: &[f32]) -> HashMap<i64, f32> {
    let rows = vectors.len();
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
                for (id, v) in &vectors[lo..hi] {
                    let score = cosine(q, v);
                    best.entry(*id)
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

fn cache_dir() -> Result<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .context("USERPROFILE/HOME not set — cannot locate model cache")?;
    Ok(PathBuf::from(home).join(".recall").join("models"))
}

/// Read refs/main (trimmed) to resolve the snapshot directory hash.
///
/// The refs/main file is written by the one-time model fetch and contains the
/// HuggingFace commit hash identifying which snapshot to load.
fn snapshot_hash(cache: &Path) -> Result<String> {
    let refs = cache
        .join("models--Xenova--bge-small-en-v1.5")
        .join("refs")
        .join("main");
    let hash = std::fs::read_to_string(&refs).with_context(|| {
        format!(
            "model not installed (refs/main missing) — run scripts/install.sh or scripts/install.ps1: {}",
            refs.display()
        )
    })?;
    Ok(hash.trim().to_string())
}

#[derive(Debug, Clone)]
pub(crate) struct OrtRuntimePreflight {
    pub version: String,
}

#[repr(C)]
struct OrtApiBase {
    get_api: unsafe extern "system" fn(u32) -> *const c_void,
    get_version_string: unsafe extern "system" fn() -> *const c_char,
}

/// Validate the ONNX Runtime DLL that ort will load.
///
/// main() pins `ORT_DYLIB_PATH` to the beside-exe DLL path (whether or not it
/// exists), so the first branch is the normal case; the exe-dir fallback covers
/// callers that bypass main() (e.g. unit tests).
pub(crate) fn ort_runtime_preflight() -> Result<OrtRuntimePreflight> {
    let path = ort_runtime_path().context("ONNX Runtime DLL not found beside recall.exe")?;
    let version = ort_runtime_version(&path).context("read ONNX Runtime DLL version")?;
    if !ort_version_compatible(&version) {
        anyhow::bail!(
            "ONNX Runtime {} is below required 1.24.0; known-bad 1.20.x runtimes can hang at session init. Install ONNX Runtime 1.24.4 beside the recall binary (scripts/install.sh or install.ps1 does this).",
            version
        );
    }
    Ok(OrtRuntimePreflight { version })
}

/// File name of the ONNX Runtime shared library on this OS, as shipped in the official
/// Microsoft release archives. Must sit beside the recall binary (the installer puts it there).
pub(crate) const fn ort_lib_filename() -> &'static str {
    if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

fn ort_runtime_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("ORT_DYLIB_PATH") {
        let path = PathBuf::from(p);
        return path.exists().then_some(path);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let path = dir.join(ort_lib_filename());
            return path.exists().then_some(path);
        }
    }
    None
}

fn ort_runtime_version(path: &Path) -> Result<String> {
    unsafe {
        let lib = libloading::Library::new(path).context("load ONNX Runtime DLL")?;
        let get_api_base: libloading::Symbol<unsafe extern "system" fn() -> *const OrtApiBase> =
            lib.get(b"OrtGetApiBase\0")
                .context("find OrtGetApiBase export")?;
        let base = get_api_base();
        if base.is_null() {
            anyhow::bail!("OrtGetApiBase returned null");
        }
        let version_ptr = ((*base).get_version_string)();
        if version_ptr.is_null() {
            anyhow::bail!("GetVersionString returned null");
        }
        Ok(CStr::from_ptr(version_ptr).to_string_lossy().into_owned())
    }
}

fn ort_version_compatible(version: &str) -> bool {
    let Some((major, minor)) = parse_ort_major_minor(version) else {
        return false;
    };
    major > MIN_ORT_MAJOR || (major == MIN_ORT_MAJOR && minor >= MIN_ORT_MINOR)
}

fn parse_ort_major_minor(version: &str) -> Option<(u64, u64)> {
    let mut parts = version.split('.');
    let major = parse_u64_prefix(parts.next()?)?;
    let minor = parse_u64_prefix(parts.next()?)?;
    Some((major, minor))
}

fn parse_u64_prefix(value: &str) -> Option<u64> {
    let digits: String = value.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// BAAI's documented query instruction for bge-*-en-v1.5 short-query -> passage retrieval
/// (FlagEmbedding model card). Queries only; stored passages are embedded without it.
const BGE_QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";
const BGE_QUERY_INSTRUCTION_ENV: &str = "RECALL_BGE_QUERY_INSTRUCTION";

/// The text the bge channel embeds for `query`. Default: the instruction + query, EXCEPT for a
/// statement (a non-question ending in '.'), which is embedded raw. v3 r3 (Codex r2 HIGH #3): the
/// instruction (2026-10-30) lifted a private question set but cut BEIR SciFact bge nDCG@10 0.7126 -> 0.6989 and
/// fused 0.7312 -> 0.7226. SciFact queries are claims (287 of 300 end in '.'), statements shaped
/// like the passages they match; no question in the private tuning sets ends in '.'. Instruction only for
/// questions was measured first: SciFact fused 0.7311, but keyword titles lost bge answers (tuning
/// pos_title fused 58 -> 57), so keyword queries keep the instruction.
/// `RECALL_BGE_QUERY_INSTRUCTION=off` never adds it, `=always` always does.
pub(crate) fn bge_query_text(query: &str) -> String {
    let statement = query.trim_end().ends_with('.') && !crate::query::is_question_form(query);
    match std::env::var(BGE_QUERY_INSTRUCTION_ENV).ok().as_deref().map(str::trim) {
        Some("off") | Some("0") | Some("false") => query.to_string(),
        Some("always") => format!("{BGE_QUERY_INSTRUCTION}{query}"),
        _ if statement => query.to_string(),
        _ => format!("{BGE_QUERY_INSTRUCTION}{query}"),
    }
}

/// Cosine similarity between two equal-length vectors. fastembed returns
/// L2-normalised vectors, so this is effectively a dot product, but we compute
/// the full form to stay correct if that ever changes.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Encode a vector as little-endian f32 bytes for BLOB storage.
/// BGE-small reads ~512 tokens and silently drops the rest. Measured 2026-08-07 over 63,281
/// live documents: **58.1% exceed that limit, and 76.5% of all characters had never been
/// embedded at all.** `docs_missing_embeddings` hands `title || ' ' || body` to the model as a
/// single string, so for a median 2,685-character note the tail is simply discarded, and the
/// longest document in the vault (513,318 chars) contributed roughly its first page.
///
/// That is not a tuning problem, it is a missing step: the Python/Gemini path chunks correctly
/// (chonkie `RecursiveChunker` with an explicit `tokenizer=`, ~3.2 chunks/doc), and the Rust
/// path never chunked at all.
///
/// EVERY CHUNK CARRIES ITS TITLE. A bare middle-of-document fragment is close to useless in a
/// result list - you cannot tell what it is or where it came from. The Gemini path already
/// prepends `title | path` for exactly this reason, at a measured 4.9% token cost. Same idea
/// here, and it is why the budget below reserves room for the prefix.
///
/// 🔴 THIS IS A BYTE CAP, NOT THE LIMIT THAT MATTERS. It used to be the only bound, and that was
/// the same bug one level down: 1800 is BYTES and the model's limit is 512 TOKENS, and the ratio
/// is not a constant. Measured 2026-08-11 with the real BGE tokenizer over 1,200 sampled vault
/// docs (7,515 chunks): **25.74% of chunks exceeded 512 tokens and were truncated by the
/// tokenizer** (`fastembed-5.14.0/src/common.rs:108` sets `with_truncation(max_length: 512)`).
///
/// | content | 1800 bytes as tokens | bytes/token |
/// |---|---|---|
/// | english prose | 370 | 4.86 |
/// | vault markdown bullets | 512 | 3.52 |
/// | CJK | 602 | 2.99 |
/// | markdown table | 938 | 1.92 |
/// | base64 | 988 | 1.82 |
/// | minified JSON | 1377 | 1.31 |
///
/// The real bound is now [`CHUNK_MAX_TOKENS`], enforced by counting. This constant survives only
/// as a secondary guard so one pathological token (a 100 KB unbroken run is a SINGLE token) cannot
/// produce an absurdly large chunk.
pub const CHUNK_BUDGET_CHARS: usize = 1800;

/// Tokens per chunk we aim for.
///
/// Short segments retrieve better than long ones — a long chunk averages several topics into one
/// vector and matches none of them sharply. 350 also leaves generous headroom under
/// [`CHUNK_MAX_TOKENS`] for the title prefix and for tokenizer boundary effects.
pub const CHUNK_TARGET_TOKENS: usize = 350;

/// Tokens each chunk repeats from the end of the previous one.
///
/// Without overlap a sentence straddling a boundary is split across two vectors and is retrievable
/// from neither — each side holds half a thought. This is the cheapest fix for the boundary loss
/// that fixed-size chunking necessarily creates.
pub const CHUNK_OVERLAP_TOKENS: usize = 64;

/// Hard ceiling on tokens in a finished chunk, INCLUDING its title prefix.
///
/// 512 is the model's window and `[CLS]` + `[SEP]` consume two of them, so 510 is the real content
/// budget. Nothing may ship above this: past it the tokenizer silently drops the tail and the
/// chunk describes text it does not contain.
pub const CHUNK_MAX_TOKENS: usize = 510;

/// Hard ceiling on the tokens a chunk's PREFIX (title + heading path) may consume.
///
/// The prefix used to be unbounded, and that was a live hole in the "nothing ships over
/// [`CHUNK_MAX_TOKENS`]" guarantee: `title` is the first `# ` line of a note, so it can be
/// arbitrarily long, and the passage budget floor (`.max(32)`) meant a 600-token title produced a
/// chunk of 600 + 1 tokens that `shrink_to_fit` could not rescue — it can only shorten the
/// PASSAGE. Adding the heading path makes the prefix longer still, so it now gets its own cap.
///
/// 128 leaves at least 382 tokens for the passage under any title, which is above
/// [`CHUNK_TARGET_TOKENS`] and so costs nothing on real documents.
pub const PREFIX_MAX_TOKENS: usize = 128;

/// Fewest tokens a summary block must have before it earns its own chunk.
///
/// A bare `## Summary` with one line under it is not a summary, it is a label. Embedding it
/// spends a vector and a ranking slot on nothing.
pub const SUMMARY_MIN_TOKENS: usize = 24;

/// Most summary chunks one document may contribute.
///
/// Some vault documents are transcripts of many sessions and carry a dozen `## Overview`
/// sections; without a cap a single note could double its own chunk count.
pub const SUMMARY_MAX_BLOCKS: usize = 8;

/// Section labels that mark a human- or LLM-written summary of the surrounding material.
///
/// 🔴 Measured on the live store 2026-08-19 (69,665 valid docs) before this was written, because
/// the premise was worth checking: `**Key takeaways:**` — the block the enrichment step writes —
/// appears in **178 documents (0.26%)**. Widening to every label below reaches **6,415 (9.21%)**,
/// dominated by `## Summary` (2,594) and `## Overview` (2,501). The narrow rule would have been a
/// re-embed of the whole corpus to change 0.26% of it.
///
/// Normalised before comparison: lowercased, trimmed, trailing `:` and `**` removed, and any
/// leading non-alphanumeric run (emoji, `🔴`, `>`) stripped.
const SUMMARY_LABELS: &[&str] = &[
    "key takeaway",
    "key takeaways",
    "takeaway",
    "takeaways",
    "tl;dr",
    "tldr",
    "summary",
    "key point",
    "key points",
    "abstract",
    "overview",
    "in short",
];

/// WordPiece gives up on a whitespace-delimited run longer than this and emits a single `[UNK]`.
///
/// Verified in the live model: `tokenizer.json` -> `model.max_input_chars_per_word = 100`, and an
/// unbroken 1800-byte run encodes to exactly `['[CLS]', '[UNK]', '[SEP]']` — the entire passage
/// reduced to one out-of-vocabulary marker, embedding NO content. The hard-split path produced
/// precisely such pieces for minified blobs and base64 payloads, so those chunks were pure cost:
/// CPU, storage and a ranking slot, for a vector that can only match other junk.
///
/// [`soften_long_runs`] is the fix — break the run at character boundaries so it tokenizes into
/// real sub-word tokens instead.
pub const MAX_WORD_CHARS: usize = 100;

/// What the chunker needs from a tokenizer.
///
/// A trait, not a concrete type, for two reasons: the chunking logic stays unit-testable without
/// loading a 130 MB ONNX model, and the fake used in tests makes the token-budget assertions
/// readable (one token per whitespace-delimited word).
pub trait ChunkTokenizer {
    /// BYTE offsets `(start, end)` of every token in `text`, in order.
    ///
    /// MUST be produced with special tokens OFF and truncation OFF. A truncating tokenizer would
    /// report 512 for every long document and the chunker would believe it.
    fn token_offsets(&self, text: &str) -> Vec<(usize, usize)>;

    /// Token count of `text`. Same contract as above.
    fn count(&self, text: &str) -> usize {
        self.token_offsets(text).len()
    }
}

/// One token per whitespace-delimited word — the tokenizer the unit tests use.
///
/// Deliberately crude: it makes every token assertion countable by eye, and it keeps the chunking
/// tests free of the 130 MB ONNX model. Lives outside `mod tests` because `store.rs` tests need it
/// too; the `real_tokenizer_*` tests are what tie the chunking logic back to the real vocabulary.
#[cfg(test)]
pub(crate) struct WordTokenizer;

#[cfg(test)]
impl ChunkTokenizer for WordTokenizer {
    fn token_offsets(&self, text: &str) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut start = None;
        for (i, ch) in text.char_indices() {
            if ch.is_whitespace() {
                if let Some(s) = start.take() {
                    out.push((s, i));
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            out.push((s, text.len()));
        }
        out
    }
}

/// The real BGE tokenizer, loaded from the same local snapshot the model comes from.
///
/// Deliberately a SECOND instance rather than `Embedder`'s: fastembed configures its copy with
/// `with_truncation(max_length: 512)`, which is right for embedding and useless for measuring —
/// it cannot tell you that a passage is 1,377 tokens, only that it is "512".
pub struct BgeTokenizer {
    inner: tokenizers::Tokenizer,
}

impl BgeTokenizer {
    pub fn load() -> Result<Self> {
        let cache = cache_dir()?;
        let snap = cache
            .join("models--Xenova--bge-small-en-v1.5")
            .join("snapshots")
            .join(snapshot_hash(&cache)?);
        let mut inner = tokenizers::Tokenizer::from_file(snap.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("load tokenizer.json for chunking: {e}"))?;
        // Both OFF on purpose. Truncation would cap every measurement at the number we are trying
        // to stay under; padding would inflate the count of every short chunk.
        inner.with_truncation(None).map_err(anyhow::Error::msg)?;
        inner.with_padding(None);
        Ok(Self { inner })
    }
}

impl ChunkTokenizer for BgeTokenizer {
    fn token_offsets(&self, text: &str) -> Vec<(usize, usize)> {
        match self.inner.encode(text, false) {
            Ok(enc) => enc.get_offsets().to_vec(),
            // A tokenizer failure must not silently produce a zero-chunk document. Fall back to
            // one span covering everything; the byte cap and the verify pass still bound it.
            Err(_) => {
                warn_once_bge("tokenizer failed on a document — falling back to a byte-bounded split");
                if text.is_empty() {
                    vec![]
                } else {
                    vec![(0, text.len())]
                }
            }
        }
    }
}

/// Start-up narration ("loading vectors", "model ready"): stderr only under `RECALL_VERBOSE=1`, so a plain
/// `recall query` prints its answer and nothing else. Warnings and errors never go through here.
fn chatter(msg: &str) {
    if crate::query::verbose_on() {
        eprintln!("bge: {msg}");
    }
}

fn warn_once_bge(msg: &str) {
    static WARNED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    let set = WARNED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    if let Ok(mut s) = set.lock() {
        if s.insert(msg.to_string()) {
            eprintln!("bge: {msg}");
        }
    }
}

/// Break whitespace-delimited runs longer than [`MAX_WORD_CHARS`] so they tokenize to real tokens.
///
/// Applied ONLY to the text that gets embedded. `char_start`/`char_end` keep pointing at the
/// original body bytes, so provenance is unaffected — `text` is "what the model saw", the offsets
/// are "where it came from", and for content containing an over-long run the two differ by the
/// inserted spaces. That is the whole trade: a base64 blob becomes a mediocre vector instead of a
/// guaranteed-useless one, and no content is dropped.
pub fn soften_long_runs(s: &str) -> String {
    // Fast path: the overwhelming majority of prose has no run this long, and allocating a new
    // String for every chunk of a 62k-document corpus is worth avoiding.
    if !has_long_run(s) {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + s.len() / MAX_WORD_CHARS + 1);
    let mut run_chars = 0usize;
    for ch in s.chars() {
        if ch.is_whitespace() {
            run_chars = 0;
        } else {
            if run_chars >= MAX_WORD_CHARS {
                out.push(' ');
                run_chars = 0;
            }
            run_chars += 1;
        }
        out.push(ch);
    }
    out
}

fn has_long_run(s: &str) -> bool {
    let mut run = 0usize;
    for ch in s.chars() {
        if ch.is_whitespace() {
            run = 0;
        } else {
            run += 1;
            if run > MAX_WORD_CHARS {
                return true;
            }
        }
    }
    false
}

/// One chunk, plus where in the document it came from.
///
/// The provenance fields exist so a hit can be shown in its surrounding context and
/// independently re-verified against the source note, rather than trusted because a vector
/// said so. They are written to `bge_chunk_embeddings`.
pub struct ChunkSpan {
    /// The text that is actually embedded: title prefix + passage.
    pub text: String,
    /// 🔴 BYTE offsets into the ORIGINAL `body`, NOT character (code point) offsets, and NOT
    /// offsets into `text` (which carries a title prefix the body does not have). `body[start..end]`
    /// slices correctly in Rust; a Python consumer must index `body.encode()` , not `body`.
    ///
    /// Named `char_*` to match the column names. That mismatch is called out here rather than
    /// renamed because this codebase has already been bitten once by a length that was counted in
    /// characters and reported as tokens (the `chonkie` bug that poisoned `embeddings3`).
    pub char_start: usize,
    pub char_end: usize,
    /// Markdown heading chain in scope where this chunk starts, joined `"H1 > H2 > H3"`, or
    /// empty above the first heading. Section boundaries carry real retrieval signal, and a
    /// mid-document fragment is far more interpretable when you can see which section it is in.
    pub heading_path: String,
    /// True for the extra, section-aligned chunk emitted over a summary block
    /// (`Key takeaways` / `TL;DR` / `Summary` …) by [`summary_chunk_spans`].
    ///
    /// These do NOT tile: they deliberately re-cover text the window chunks already cover, so any
    /// invariant about spans being contiguous or reaching the end of the body applies to
    /// `!is_summary` chunks only. Not a stored column — `bge_chunk_embeddings` is unchanged.
    ///
    /// Read only by the tests today, and that is the point of it: the tiling invariants
    /// (`starts at 0`, `no gaps`, `reaches the end`) cannot be stated without a way to exclude
    /// the non-tiling rows, and a test that silently stopped checking them is how the 25.74%
    /// truncation survived a green suite.
    #[allow(dead_code)]
    pub is_summary: bool,
}

/// The label line every chunk carries: `"Title > H1 > H2"`, capped at [`PREFIX_MAX_TOKENS`].
///
/// 🔴 THE HEADING PATH IS EMBEDDED, NOT JUST STORED. It was computed and written to
/// `bge_chunk_embeddings.heading_path` from the first per-chunk commit, but the text handed to the
/// model was the title alone — so a chunk from the middle of a long note arrived with no idea
/// which section it came from, and `## Assets` / `## Transcript` / `## Visual content` were
/// indistinguishable to the vector. That distinction is exactly what separates "which repo did I
/// save about X" from "who said X".
///
/// Redundancy is stripped, not repeated: `title` is usually the note's first `# ` line, so the
/// heading path very often BEGINS with it. `Doc` + `Doc > Setup` must render `Doc > Setup`, never
/// `Doc\nDoc > Setup` — the duplicate costs tokens and teaches the model nothing.
fn chunk_prefix(title: &str, heading_path: &str, tok: &dyn ChunkTokenizer) -> String {
    let label = if heading_path.is_empty() {
        title.to_string()
    } else if title.is_empty() || heading_path == title {
        heading_path.to_string()
    } else if heading_path.starts_with(title) && heading_path[title.len()..].starts_with(" > ") {
        heading_path.to_string()
    } else {
        format!("{title} > {heading_path}")
    };
    if label.is_empty() {
        return String::new();
    }
    // Soften BEFORE capping: an unbroken 100 KB title tokenizes to a single `[UNK]`, so a token
    // cap alone would happily pass it through and blow the byte side of the budget instead.
    let label = soften_long_runs(&label);
    if tok.count(&label) <= PREFIX_MAX_TOKENS {
        return format!("{label}\n\n");
    }

    // Over budget. Drop components from the LEFT, never the right.
    //
    // Caught by `measure_vault_chunk_tokens` on a real note whose chain is
    // `Features Research Template > Feature Research > Feature Landscape > Differentiation…`:
    // a plain token truncation kept the generic outer headings, cut the specific inner one in
    // half mid-word, and so threw away precisely the component that distinguishes this chunk
    // from every other chunk of the same document. The deepest heading is the signal.
    let parts: Vec<&str> = label.split(" > ").collect();
    for i in 1..parts.len() {
        let candidate = parts[i..].join(" > ");
        if tok.count(&candidate) <= PREFIX_MAX_TOKENS {
            return format!("{candidate}\n\n");
        }
    }
    // Even the deepest component alone is over budget — a heading longer than a paragraph. Now
    // truncation is the only option left, and it is applied to that component, not to its parents.
    let deepest = truncate_to_tokens(tok, parts.last().copied().unwrap_or(""), PREFIX_MAX_TOKENS);
    format!("{deepest}\n\n")
}

/// First `max` tokens of `s`, cut on a token boundary (and then a char boundary).
fn truncate_to_tokens(tok: &dyn ChunkTokenizer, s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    let offs = tok.token_offsets(s);
    if offs.len() <= max {
        return s.to_string();
    }
    let end = ceil_char_boundary(s, offs[max - 1].1.min(s.len()));
    s[..end].to_string()
}

/// Split a document into title-prefixed chunks that fit inside BGE's context window.
///
/// TOKEN-BOUNDED, not byte-bounded. The previous version accumulated paragraphs up to
/// [`CHUNK_BUDGET_CHARS`] bytes; measured against the real tokenizer, 25.74% of the chunks it
/// produced exceeded the model's 512-token window and were silently truncated — the same class of
/// bug as the un-chunked path it replaced, one level down. Bytes cannot bound tokens when the
/// ratio swings from 1.31 (minified JSON) to 4.86 (english prose) across one vault.
///
/// The shape now:
///
///   1. Tokenize the body ONCE with an untruncated tokenizer, giving exact token byte-offsets.
///   2. Window over tokens: [`CHUNK_TARGET_TOKENS`] per chunk, stepping back
///      [`CHUNK_OVERLAP_TOKENS`] each time so a sentence on a boundary survives in one piece on
///      one side. [`CHUNK_BUDGET_CHARS`] still caps bytes, because a single `[UNK]` token can
///      cover 100 KB.
///   3. [`soften_long_runs`] the slice so unbroken runs tokenize to real tokens instead of one
///      `[UNK]`, then VERIFY the assembled `prefix + passage` against [`CHUNK_MAX_TOKENS`] and
///      shrink until it fits. The verify step is what makes the ceiling a guarantee rather than an
///      estimate: softening ADDS tokens, and prefix+passage does not tokenize as the sum of its
///      parts.
///
/// Every byte of the body still lands in at least one chunk. Nothing is dropped.
pub fn chunk_for_embedding_spans(title: &str, body: &str, tok: &dyn ChunkTokenizer) -> Vec<ChunkSpan> {
    let title = title.trim();
    let headings = scan_headings(body);
    let offsets = tok.token_offsets(body);

    // An empty (or whitespace-only) body still deserves one row: the title alone is a real,
    // matchable signal, and a document with zero chunks would be selected by the embed work queue
    // on every pass, forever.
    if offsets.is_empty() {
        return vec![ChunkSpan {
            text: chunk_prefix(title, "", tok).trim_end().to_string(),
            char_start: 0,
            char_end: 0,
            heading_path: String::new(),
            is_summary: false,
        }];
    }

    // WHOLE-DOCUMENT FLOOR: never split a document the model can read in one pass.
    //
    // The two window constants are both far below the model's real capacity: the chunker aims at
    // CHUNK_TARGET_TOKENS (350) and CHUNK_BUDGET_CHARS caps window growth at 1800 BYTES, while
    // CHUNK_MAX_TOKENS (510) is what actually fits. A ~1.9 KB note is only ~450 tokens — it fits
    // whole — yet the byte cap forced it into two overlapping halves, so neither vector held the
    // complete document and a one-line answer (a table row, a checklist item) was embedded with
    // half its context missing.
    //
    // The threshold is deliberately NOT a byte count. "Does the finished chunk fit the model's
    // window" is the only question that matters, it is measured with the untruncated tokenizer,
    // and it self-adjusts across prose, tables and minified JSON — which range 1.31 to 4.86
    // bytes per token, so any byte threshold is wrong for most of the corpus by construction.
    //
    // Emitting one span here is also strictly more information than two overlapping ones: the
    // single vector covers 100% of the document, where each half covered ~60%.
    {
        let heading_path = heading_at(&headings, 0);
        let prefix = chunk_prefix(title, &heading_path, tok);
        let whole = format!("{prefix}{}", soften_long_runs(body));
        if tok.count(&whole) <= CHUNK_MAX_TOKENS {
            let mut chunks = vec![ChunkSpan {
                text: whole,
                char_start: 0,
                char_end: body.len(),
                heading_path,
                is_summary: false,
            }];
            // Summary blocks still get their own span. A short document can carry a
            // `Key takeaways` block, and that block is the highest-signal passage in it.
            let extra = summary_chunk_spans(title, body, &headings, &chunks, tok);
            chunks.extend(extra);
            return chunks;
        }
    }

    let mut chunks: Vec<ChunkSpan> = Vec::new();
    let mut start_b = 0usize;
    loop {
        // The prefix is rebuilt PER CHUNK, because it carries the heading path in scope where this
        // chunk starts — which is only knowable here. Everything the prefix influences (the token
        // room left for the passage, the target window, the overlap) therefore moves inside the
        // loop with it.
        let heading_path = heading_at(&headings, start_b);
        let prefix = chunk_prefix(title, &heading_path, tok);
        let prefix_tokens = tok.count(&prefix);

        // Room left for the passage after the prefix has taken its share. The floor keeps a
        // pathological prefix from driving the passage budget to zero and looping forever;
        // [`PREFIX_MAX_TOKENS`] is what keeps the floor from being reached in the first place.
        let room_tokens = CHUNK_MAX_TOKENS.saturating_sub(prefix_tokens).max(32);
        let target_tokens = CHUNK_TARGET_TOKENS.min(room_tokens);
        // Overlap must stay strictly under the target or the window never advances.
        let overlap_tokens = CHUNK_OVERLAP_TOKENS.min(target_tokens.saturating_sub(1) / 2);

        let fits = |start: usize, end: usize| -> bool {
            let text = format!("{prefix}{}", soften_long_runs(&body[start..end]));
            tok.count(&text) <= CHUNK_MAX_TOKENS
        };

        // First token starting at or after the cursor.
        let i = offsets.partition_point(|(s, _)| *s < start_b);

        let mut end_b = if i >= offsets.len() {
            body.len()
        } else {
            // Grow to `target_tokens` tokens, stopping early at the byte cap.
            let mut j = i;
            while j < offsets.len()
                && j - i < target_tokens
                && offsets[j].1.saturating_sub(start_b) <= CHUNK_BUDGET_CHARS
            {
                j += 1;
            }
            if j > i {
                offsets[j - 1].1
            } else {
                // Not even ONE token fits the byte cap — a 100 KB unbroken run is a single
                // token. Step by bytes, or the cap silently never applies and the chunk is
                // however wide that token happens to be.
                start_b + CHUNK_BUDGET_CHARS
            }
        };
        end_b = ceil_char_boundary(body, end_b.max(start_b + 1).min(body.len()));

        // Whatever follows the last token is trailing whitespace. Leaving it out stops the spans
        // one byte short of the body, which reads as a gap in coverage for no benefit.
        if body[end_b..].trim().is_empty() {
            end_b = body.len();
        }

        // VERIFY, then shrink. Softening can multiply the token count several times over (a
        // 1800-byte base64 run becomes ~18 words of real sub-word tokens), so the provisional
        // window is a starting guess, never the answer.
        if !fits(start_b, end_b) {
            end_b = shrink_to_fit(body, start_b, end_b, &fits);
        }

        chunks.push(ChunkSpan {
            text: format!("{prefix}{}", soften_long_runs(&body[start_b..end_b])),
            char_start: start_b,
            char_end: end_b,
            heading_path,
            is_summary: false,
        });

        if end_b >= body.len() {
            break;
        }
        start_b = next_start(&offsets, start_b, end_b, overlap_tokens);
    }

    if chunks.is_empty() {
        chunks.push(ChunkSpan {
            text: chunk_prefix(title, "", tok).trim_end().to_string(),
            char_start: 0,
            char_end: 0,
            heading_path: String::new(),
            is_summary: false,
        });
    }

    let extra = summary_chunk_spans(title, body, &headings, &chunks, tok);
    chunks.extend(extra);
    chunks
}

/// Give every summary block (`Key takeaways`, `TL;DR`, `Summary`, …) a chunk of its own.
///
/// The text is ALREADY embedded — the sliding window covers every byte of the body — so this is
/// not a coverage fix. It is an ISOLATION fix: a summary sentence currently shares a vector with
/// ~350 tokens of whatever happened to surround it, and the average of "dense summary" and "the
/// raw material it summarises" matches neither sharply. The retrieval literature's advice for
/// paraphrased queries (HyPE, proposition chunking, document augmentation) is to spend an LLM call
/// generating a dense summary and embed THAT; the enrichment step already spent the call, and the
/// result was being diluted rather than indexed.
///
/// 🔴 Read the measurement in [`SUMMARY_LABELS`] before believing this moves a benchmark. On the
/// private evaluation set, almost no answer span sits inside a summary block, and
/// no answer document contains the `**Key takeaways:**` block specifically.
///
/// Appended after the window chunks so `chunk_index` keeps meaning "position in the document" for
/// the tiling chunks, and duplicates are cheap: `best_per_doc` takes the max over a document's
/// chunks, so an extra near-identical vector can raise a score and can never lower one.
fn summary_chunk_spans(
    title: &str,
    body: &str,
    headings: &[(usize, String)],
    window_chunks: &[ChunkSpan],
    tok: &dyn ChunkTokenizer,
) -> Vec<ChunkSpan> {
    let mut out: Vec<ChunkSpan> = Vec::new();
    for (start, end) in summary_blocks(body) {
        if out.len() >= SUMMARY_MAX_BLOCKS {
            break;
        }
        // Too small to be a summary — a label with a line under it.
        // A `## Overview` section can be the whole document. Both the size test and the shrink
        // below only ever look at the first few hundred tokens, so clamp the BYTES first —
        // otherwise a 500 KB section costs a full tokenizer pass per binary-search probe.
        // 3× the byte cap is ~1,100 tokens even for the loosest prose ratio measured (4.86
        // bytes/token), so this can never cut below the 510-token ceiling.
        let end = ceil_char_boundary(body, end.min(start + CHUNK_BUDGET_CHARS * 3));
        if tok.count(&body[start..end]) < SUMMARY_MIN_TOKENS {
            continue;
        }
        // Already isolated. When a window chunk is barely wider than the block, a second vector
        // of the same text buys nothing and costs a ranking slot.
        let block_len = end - start;
        if window_chunks.iter().any(|c| {
            c.char_start <= start && c.char_end >= end && (c.char_end - c.char_start) * 4 <= block_len * 5
        }) {
            continue;
        }

        let heading_path = heading_at(headings, start);
        let prefix = chunk_prefix(title, &heading_path, tok);
        let fits = |s: usize, e: usize| -> bool {
            tok.count(&format!("{prefix}{}", soften_long_runs(&body[s..e]))) <= CHUNK_MAX_TOKENS
        };
        // A `## Summary` section can itself be pages long. Truncating it is right here in a way it
        // would not be for the tiling pass: the window chunks still cover the whole block, so
        // nothing is dropped — this row is an extra, sharper view of its opening.
        let end = if fits(start, end) {
            end
        } else {
            shrink_to_fit(body, start, end, &fits)
        };

        out.push(ChunkSpan {
            text: format!("{prefix}{}", soften_long_runs(&body[start..end])),
            char_start: start,
            char_end: end,
            heading_path,
            is_summary: true,
        });
    }
    out
}

/// Byte ranges of blocks introduced by a [`SUMMARY_LABELS`] line, label line included.
///
/// A block ends at the next ATX heading of the same level or shallower, at the next summary label,
/// or at the end of the document. Bold labels (`**Key takeaways:**`) are treated as level 7, so any
/// heading at all closes them — that is what the enrichment step emits, and it sits inside a
/// section rather than starting one.
fn summary_blocks(body: &str) -> Vec<(usize, usize)> {
    let mut opens: Vec<(usize, usize)> = Vec::new(); // (offset, level)
    let mut closers: Vec<(usize, usize)> = Vec::new(); // (offset, level) of every heading
    let mut offset = 0usize;
    for line in body.split('\n') {
        if let Some((level, text)) = parse_atx_heading(line) {
            closers.push((offset, level));
            if is_summary_label(&text) {
                opens.push((offset, level));
            }
        } else if let Some(text) = parse_bold_label(line) {
            if is_summary_label(&text) {
                opens.push((offset, 7));
            }
        }
        offset += line.len() + 1; // +1 for the '\n' that `split` consumed
    }

    opens
        .iter()
        .map(|&(start, level)| {
            let by_heading = closers
                .iter()
                .find(|&&(off, l)| off > start && l <= level)
                .map(|&(off, _)| off);
            let by_label = opens.iter().find(|&&(off, _)| off > start).map(|&(off, _)| off);
            let end = match (by_heading, by_label) {
                (Some(a), Some(b)) => a.min(b),
                (Some(a), None) => a,
                (None, Some(b)) => b,
                (None, None) => body.len(),
            };
            (start, end.min(body.len()))
        })
        .filter(|(s, e)| e > s)
        .collect()
}

/// `**Key takeaways:**` -> `Some("Key takeaways:")`. The whole line must be the bold run, or a
/// sentence that merely opens in bold would be mistaken for a section label.
fn parse_bold_label(line: &str) -> Option<String> {
    let l = line.trim();
    let inner = l.strip_prefix("**")?;
    let inner = inner.strip_suffix("**").or_else(|| {
        // `**Key takeaways**:` — the colon outside the bold run is just as common.
        inner.strip_suffix(":").and_then(|s| s.strip_suffix("**"))
    })?;
    if inner.is_empty() || inner.contains("**") {
        return None;
    }
    Some(inner.to_string())
}

/// Is this section label one of [`SUMMARY_LABELS`], allowing for decoration?
///
/// `🔴 Key takeaways:` and `## Summary` and `**TL;DR**` all normalise to the same thing. A label
/// with trailing words (`## Summary of the 2026-08 incident`) deliberately does NOT match: it
/// names a topic, and the block under it is ordinary prose, not a dense summary of its neighbours.
fn is_summary_label(text: &str) -> bool {
    let t = text.trim().trim_end_matches('*').trim().trim_end_matches(':');
    let t = t.trim_start_matches(|c: char| !c.is_alphanumeric()).trim();
    let t = t.to_lowercase();
    SUMMARY_LABELS.contains(&t.as_str())
}

/// Byte offset the next chunk starts at, stepped back [`CHUNK_OVERLAP_TOKENS`] whole tokens.
///
/// Two guards, both load-bearing:
///   * the result must be strictly greater than `start_b`, or a chunk that got shrunk to fewer
///     tokens than the overlap would set the next start behind its own and loop forever;
///   * when the chunk contains no interior token boundary to step back to — it is one opaque
///     token, a minified blob — there is nothing meaningful to overlap, so advance cleanly rather
///     than crawl forward a byte at a time and emit thousands of near-identical chunks.
fn next_start(
    offsets: &[(usize, usize)],
    start_b: usize,
    end_b: usize,
    overlap_tokens: usize,
) -> usize {
    let k = offsets.partition_point(|(s, _)| *s < end_b);
    let back = k.saturating_sub(overlap_tokens);
    let candidate = offsets.get(back).map(|(s, _)| *s).unwrap_or(end_b);
    if candidate > start_b {
        candidate
    } else {
        end_b
    }
}

/// Largest `end` in `(start, end]` whose softened, prefixed text fits the token ceiling.
///
/// Binary search rather than a linear back-off: one pathological token can span 100 KB, and
/// walking that back a token at a time would be O(n) tokenizer calls on exactly the documents that
/// are already the most expensive.
fn shrink_to_fit(
    body: &str,
    start: usize,
    mut end: usize,
    fits: &dyn Fn(usize, usize) -> bool,
) -> usize {
    let mut lo = start + 1;
    let mut hi = end;
    let mut best = None;
    while lo <= hi {
        let raw = lo + (hi - lo) / 2;
        let mid = ceil_char_boundary(body, raw);
        // 🔴 `ceil_char_boundary` rounds UP, so `mid` can land PAST `hi` when `raw` falls inside
        // a multi-byte character. The `hi = mid - 1` below then computes a value >= the current
        // `hi`, the window does not shrink, and the search repeats the same state forever.
        //
        // Measured 2026-08-20: this hung the full-corpus backfill dead on
        // `skills/fastmcp-server/references/features/sampling.md` — one core pegged at 100%,
        // ZERO bytes of I/O, no database write for 40 minutes, while the process looked alive.
        // The offending state was (lo=40, hi=42) with a 3-byte em dash at byte 40: raw=41 is a
        // continuation byte, ceil gives 43, and `hi = 42` is exactly what it already was.
        //
        // Shrinking by the RAW midpoint instead always makes progress: raw < hi whenever
        // lo < hi, and raw == lo == hi collapses the window and ends the loop.
        if mid > hi {
            hi = raw.saturating_sub(1);
            continue;
        }
        if mid <= start {
            lo = mid + 1;
            continue;
        }
        if fits(start, mid) {
            best = Some(mid);
            lo = mid + 1;
        } else {
            if mid == start + 1 {
                break;
            }
            hi = mid - 1;
        }
    }
    // Nothing fits, not even one character — impossible for a real tokenizer, but returning
    // `start` would emit an empty chunk and stall the loop. Take one character and move on.
    end = best.unwrap_or_else(|| ceil_char_boundary(body, start + 1));
    end.min(body.len())
}

/// Round `i` up to the next char boundary so `&body[..i]` never panics on multi-byte UTF-8.
fn ceil_char_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Byte offset -> heading chain in scope at that offset, in document order.
///
/// Computed in one pass up front rather than tracked while chunking: the two concerns are
/// independent, and windows no longer align to paragraphs so there is nothing to piggyback on.
fn scan_headings(body: &str) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = Vec::new();
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut offset = 0usize;
    for line in body.split('\n') {
        if let Some((level, text)) = parse_atx_heading(line) {
            stack.retain(|(l, _)| *l < level);
            stack.push((level, text));
            out.push((offset, heading_path_of(&stack)));
        }
        offset += line.len() + 1; // +1 for the '\n' split consumed
    }
    out
}

/// The heading chain in scope at `pos` — the last heading at or before it.
fn heading_at(headings: &[(usize, String)], pos: usize) -> String {
    match headings.partition_point(|(off, _)| *off <= pos) {
        0 => String::new(),
        n => headings[n - 1].1.clone(),
    }
}


/// `## Heading` -> `(2, "Heading")`. Only ATX headings; setext (`===` underlines) are not
/// recognised, and a `#` inside a fenced code block will be misread as one — accepted, because
/// the cost is a slightly wrong label on a chunk, never a lost or misplaced chunk.
fn parse_atx_heading(line: &str) -> Option<(usize, String)> {
    let l = line.trim_start();
    let level = l.chars().take_while(|c| *c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &l[level..];
    // A heading marker must be followed by a space — `#hashtag` is not a heading.
    if !rest.starts_with(' ') {
        return None;
    }
    let text = rest.trim().trim_end_matches('#').trim();
    if text.is_empty() {
        return None;
    }
    Some((level, text.to_string()))
}

fn heading_path_of(stack: &[(usize, String)]) -> String {
    stack
        .iter()
        .map(|(_, t)| t.as_str())
        .collect::<Vec<_>>()
        .join(" > ")
}

/// Average several chunk vectors into one document vector, then L2-normalise.
///
/// Mean-pooling was deliberate: it kept ONE row per document, so `embeddings` stayed keyed by
/// `doc_id` and no schema migration, no ANN rebuild and no change to the query path was needed.
/// It fixed the 76.5% blindness reversibly, and bought less than it looked like: the mean of a
/// long note's chunk vectors resembles none of them, so the passage in the middle stayed
/// unfindable. The per-chunk table it deferred is now live — see `bge_chunk_embeddings` and
/// [`best_per_doc`].
///
/// 🔴 NO LONGER ON THE EMBED PATH. Kept, with its tests, because it is the exact operation any
/// future A/B against per-chunk retrieval needs, and because deleting a tested function to
/// silence one dead-code warning is how a comparison baseline disappears.
///
/// Re-normalising matters: the cosine search assumes unit vectors, and the mean of unit vectors
/// is not itself unit length.
#[allow(dead_code)]
pub fn mean_pool(vecs: &[Vec<f32>]) -> Vec<f32> {
    if vecs.is_empty() {
        return vec![0.0; DIM];
    }
    if vecs.len() == 1 {
        return vecs[0].clone();
    }
    let mut out = vec![0.0f32; vecs[0].len()];
    for v in vecs {
        for (o, x) in out.iter_mut().zip(v.iter()) {
            *o += *x;
        }
    }
    let n = vecs.len() as f32;
    for o in out.iter_mut() {
        *o /= n;
    }
    let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for o in out.iter_mut() {
            *o /= norm;
        }
    }
    out
}

pub fn to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode little-endian f32 bytes back into a vector.
pub fn from_bytes(b: &[u8]) -> Vec<f32> {
    // A BLOB whose length is not a multiple of 4 means a corrupt/truncated
    // embedding; chunks_exact would silently drop the remainder and yield a
    // short vector that cosine() then rejects (len mismatch -> 0.0), making the
    // doc invisible with no error. Catch the corruption in debug/test builds.
    debug_assert_eq!(b.len() % 4, 0, "embedding BLOB length not a multiple of 4");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Codex v3 r2 MEDIUM #4: the per-call cap is the memory ceiling, so it must stay 64 and
    /// `Embedder::embed` must pass it to fastembed (None = fastembed's 256-row default). Codex v3 r3
    /// MEDIUM #3: and fastembed is called once per 64-row slice, so raw outputs never accumulate.
    #[test]
    fn every_onnx_call_is_capped_at_64_rows() {
        assert_eq!(MAX_ROWS_PER_ONNX_CALL, 64);
        let src = include_str!("bge.rs");
        let body = &src[src.find("pub fn embed(&self").unwrap()..];
        let body = &body[..body.find("pub fn embed_one").unwrap()];
        assert!(body.contains(".embed(refs, Some(MAX_ROWS_PER_ONNX_CALL))"), "embed must cap each ONNX call");
        assert!(!body.contains(".embed(refs, None)"));
        assert!(body.contains("texts.chunks(MAX_ROWS_PER_ONNX_CALL)"), "one fastembed call per 64-row slice");
    }

    /// `recall eval` grades the active path against the exact scan. Both must rank by the SAME query
    /// vector, or the score measures the query instruction, not the search path: on a store with
    /// more than k documents two different vectors give two different top-k sets.
    #[test]
    fn the_eval_oracle_embeds_the_query_as_the_active_path_does() {
        let src = include_str!("bge.rs");
        let body_of = |sig: &str| {
            let b = &src[src.find(sig).unwrap()..];
            &b[..10 + b[10..].find("    pub").unwrap()] // up to the next method
        };
        for sig in ["pub fn search_scored(&self", "pub fn exact_search_ids(&self"] {
            let body = body_of(sig);
            assert!(body.contains("self.query_vector(query)"), "{sig} must embed through query_vector");
            assert!(!body.contains("embed_one("), "{sig} must not embed the query its own way");
        }
        // Why it matters, without the model: 7 documents, k = 5, two nearby query vectors.
        let docs: Vec<(i64, Vec<f32>)> = (0..7).map(|i| (i as i64, vec![(i as f32 * 0.3).cos(), (i as f32 * 0.3).sin()])).collect();
        let top = |q: &[f32]| exact_scored(best_per_doc(&docs, q), 5).into_iter().map(|(id, _)| id).collect::<Vec<_>>();
        assert_eq!(top(&[1.0, 0.0]), top(&[1.0, 0.0]), "same vector, same top-k");
        assert_ne!(top(&[1.0, 0.0]), top(&[0.0, 1.0]), "a different query vector moves the top-k");
        assert_eq!(crate::eval::recall_at_k(&top(&[1.0, 0.0]), &top(&[1.0, 0.0]), 5), 1.0);
        assert!(crate::eval::recall_at_k(&top(&[1.0, 0.0]), &top(&[0.0, 1.0]), 5) < 1.0);
    }

    /// The embed memory fix: the CPU provider is passed explicitly with the arena OFF. Going back
    /// to `Vec::new()` (ORT's default arena) or to `with_arena_allocator(true)` fails here.
    #[test]
    fn cpu_provider_disables_the_ort_memory_arena() {
        let ep = cpu_ep();
        let cpu = ep.downcast_ref::<ort::ep::CPU>().expect("CPUExecutionProvider");
        assert_eq!(format!("{cpu:?}"), "CPU { use_arena: false }");
        assert_eq!(format!("{ep:?}"), "CPUExecutionProvider { error_on_failure: false }");
    }

    #[test]
    fn bytes_roundtrip() {
        let v = vec![0.1f32, -0.5, 1.0, 0.0];
        assert_eq!(from_bytes(&to_bytes(&v)), v);
    }

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![0.3f32, 0.4, 0.5];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-5);
    }

    // ── chunk provenance (spans + heading path) ─────────────────────────────

    /// The provenance invariant, stated exactly.
    ///
    /// `text` is prefix + SOFTENED passage, so it is not byte-identical to `body[start..end]`
    /// whenever the passage contains an over-long unbroken run. Softening only ever inserts
    /// spaces, so removing them recovers the original — that is the property worth guarding, and
    /// it is what makes a stored hit re-checkable against the note.
    fn assert_span_maps_back(body: &str, prefix: &str, span: &ChunkSpan) {
        let passage = span
            .text
            .strip_prefix(prefix)
            .unwrap_or_else(|| panic!("chunk lost its title prefix: {:?}", span.text));
        let original = &body[span.char_start..span.char_end];
        assert_eq!(
            passage.replace(' ', ""),
            original.replace(' ', ""),
            "offsets {}..{} do not map back to the chunk text",
            span.char_start,
            span.char_end
        );
    }

    #[test]
    fn span_offsets_map_back_to_the_body() {
        let body = format!(
            "# Doc\n\nshort para\n\n{}\n\ntail para",
            "z".repeat(5000) // forces both the byte cap and the softening path
        );
        for span in chunk_for_embedding_spans("Doc", &body, &WordTokenizer) {
            assert_span_maps_back(&body, "Doc\n\n", &span);
        }
    }

    #[test]
    fn span_offsets_survive_multibyte_utf8() {
        // Byte offsets into a body with multi-byte characters must land on char boundaries, or
        // the slice panics.
        let body = "# Notes\n\nカフェのテキストです\n\némoji 🎉 and accents éàü\n\ntail";
        for span in chunk_for_embedding_spans("Notes", body, &WordTokenizer) {
            assert_span_maps_back(body, "Notes\n\n", &span);
        }
    }

    /// Heading tracking, tested directly rather than through chunk-size arithmetic.
    ///
    /// Asserting on which CHUNK carries which heading couples the test to the token budget, so
    /// tuning the budget breaks a test that is not about the budget. This checks the mapping the
    /// chunker actually consumes.
    #[test]
    fn heading_path_tracks_the_nesting_and_pops_back_out() {
        let body = "# Top\n\nintro\n\n## Mid\n\nmiddle\n\n### Deep\n\ndeep\n\n## Other\n\nother";
        let headings = scan_headings(body);
        let paths: Vec<&str> = headings.iter().map(|(_, p)| p.as_str()).collect();
        assert_eq!(
            paths,
            vec!["Top", "Top > Mid", "Top > Mid > Deep", "Top > Other"],
            "heading stack did not nest and unwind correctly"
        );

        // And the offset lookup must return the chain in scope, not the nearest by any other rule.
        let at = |needle: &str| heading_at(&headings, body.find(needle).unwrap());
        assert_eq!(at("intro"), "Top");
        assert_eq!(at("middle"), "Top > Mid");
        assert_eq!(at("deep"), "Top > Mid > Deep");
        // `## Other` must POP `### Deep` and `## Mid`, not append to them.
        assert_eq!(at("other"), "Top > Other");
    }

    #[test]
    fn chunks_carry_the_heading_they_start_under() {
        let para = |tag: &str, n: usize| format!("{tag} {}", "word ".repeat(n));
        let body = format!(
            "# Top\n\n{}\n\n## Mid\n\n{}\n\n### Deep\n\n{}\n\n## Other\n\n{}",
            para("intro", 100),
            para("middle", 100),
            para("deep", 300),
            para("other", 200)
        );
        let spans = chunk_for_embedding_spans("Top", &body, &WordTokenizer);
        let paths: Vec<&str> = spans.iter().map(|s| s.heading_path.as_str()).collect();
        assert!(spans.len() > 2, "need several chunks, got {}", spans.len());
        assert!(
            paths.iter().any(|p| p.contains(" > ")),
            "no chunk landed under a nested heading: {paths:?}"
        );
        assert!(
            !paths.iter().any(|p| p.contains("Deep > Other")),
            "a sibling heading was nested under its predecessor: {paths:?}"
        );
    }

    #[test]
    fn heading_at_returns_empty_above_the_first_heading() {
        let body = "preamble text before any heading\n\n# First\n\nunder first";
        let spans = chunk_for_embedding_spans("First", body, &WordTokenizer);
        assert_eq!(spans[0].heading_path, "", "preamble is not under a heading");
    }

    #[test]
    fn a_hashtag_is_not_a_heading() {
        // `#hashtag` has no space after the marker, so it must not shift the heading path.
        let body = "# Real\n\n#nothashtag and more words\n\nbody text";
        let spans = chunk_for_embedding_spans("Real", body, &WordTokenizer);
        assert!(
            spans.iter().all(|s| s.heading_path == "Real"),
            "{:?}",
            spans.iter().map(|s| &s.heading_path).collect::<Vec<_>>()
        );
    }

    // ── the heading path is EMBEDDED, not merely stored (improvement A) ──────
    //
    // `heading_path` was computed, written to `bge_chunk_embeddings` and unit-tested from the
    // first per-chunk commit — and never reached the model, because `prefix` was built from the
    // title alone, once, outside the window loop. The column looked like the feature.

    #[test]
    fn the_embedded_text_carries_the_heading_path_not_just_the_column() {
        let para = |tag: &str, n: usize| format!("{tag} {}", "word ".repeat(n));
        let body = format!(
            "# Reel\n\n{}\n\n## Transcript\n\n{}\n\n## Visual content\n\n{}",
            para("intro", 200),
            para("spoken", 400),
            para("onscreen", 400)
        );
        let spans = chunk_for_embedding_spans("Reel", &body, &WordTokenizer);
        for s in &spans {
            if s.heading_path.is_empty() {
                continue;
            }
            assert!(
                s.text.starts_with(&format!("{}\n\n", s.heading_path)),
                "chunk under {:?} does not lead with it: {:?}",
                s.heading_path,
                &s.text[..s.text.len().min(60)]
            );
        }
        // And the two sections must be distinguishable to the model, which is the entire point.
        assert!(
            spans.iter().any(|s| s.text.starts_with("Reel > Transcript")),
            "no chunk announced the Transcript section"
        );
        assert!(
            spans.iter().any(|s| s.text.starts_with("Reel > Visual content")),
            "no chunk announced the Visual content section"
        );
    }

    #[test]
    fn the_prefix_does_not_repeat_a_title_the_heading_path_already_starts_with() {
        // `title` is usually the note's first `# ` line, so `Doc` + `Doc > Setup` is the common
        // case. Emitting "Doc\n\nDoc > Setup" would spend tokens teaching the model nothing.
        let p = |t: &str, h: &str| chunk_prefix(t, h, &WordTokenizer);
        assert_eq!(p("Doc", "Doc > Setup"), "Doc > Setup\n\n");
        assert_eq!(p("Doc", "Doc"), "Doc\n\n");
        assert_eq!(p("Doc", ""), "Doc\n\n");
        assert_eq!(p("", "Setup"), "Setup\n\n");
        assert_eq!(p("", ""), "");
        // A heading that merely starts with the same letters is NOT the title.
        assert_eq!(p("Doc", "Docker > Setup"), "Doc > Docker > Setup\n\n");
    }

    #[test]
    fn an_over_budget_heading_chain_drops_outer_headings_and_keeps_the_deepest() {
        // The deepest heading is what distinguishes this chunk from every other chunk of the same
        // document. Truncating from the right keeps the generic outer headings and throws away
        // exactly the discriminating one — found by `measure_vault_chunk_tokens` on a real note.
        let outer = "a ".repeat(200); // blows the budget on its own
        let path = format!("{}> Differentiation Matrix", outer);
        let prefix = chunk_prefix("Root", &path, &WordTokenizer);
        assert!(
            prefix.starts_with("Differentiation Matrix"),
            "the specific heading was dropped: {:?}",
            &prefix[..prefix.len().min(80)]
        );
        assert!(WordTokenizer.count(&prefix) <= PREFIX_MAX_TOKENS);
    }

    #[test]
    fn a_pathological_prefix_cannot_eat_the_whole_window() {
        // The bug this closes: `shrink_to_fit` can only shorten the PASSAGE, so an unbounded
        // prefix produced a chunk over the ceiling that nothing could rescue.
        let title = "extremely verbose heading ".repeat(200);
        let prefix = chunk_prefix(&title, &title, &WordTokenizer);
        assert!(
            WordTokenizer.count(&prefix) <= PREFIX_MAX_TOKENS,
            "prefix is {} tokens",
            WordTokenizer.count(&prefix)
        );
    }

    // ── summary blocks get their own chunk (improvement B) ──────────────────

    #[test]
    fn a_key_takeaways_block_becomes_its_own_chunk() {
        let body = format!(
            "# Note\n\n## Body\n\n{}\n\n**Key takeaways:**\n\n\
             - the router picks the cheapest model that clears the gate\n\
             - fallbacks are ordered by measured latency, not by price\n\
             - a dead channel must fail loudly rather than serve keyword hits\n",
            "filler words in a long section ".repeat(200)
        );
        let spans = chunk_for_embedding_spans("Note", &body, &WordTokenizer);
        let summaries: Vec<&ChunkSpan> = spans.iter().filter(|s| s.is_summary).collect();
        assert_eq!(summaries.len(), 1, "expected exactly one summary chunk");
        let s = summaries[0];
        assert!(s.text.contains("**Key takeaways:**"), "{:?}", s.text);
        assert!(s.text.contains("cheapest model"));
        assert!(
            !s.text.contains("filler words"),
            "the summary chunk was diluted with the section it summarises"
        );
        assert_eq!(&body[s.char_start..s.char_end], &body[s.char_start..s.char_end]);
        assert!(body[s.char_start..].starts_with("**Key takeaways:**"));
    }

    #[test]
    fn summary_blocks_end_at_the_next_heading_of_the_same_level() {
        let body = "# T\n\n## Summary\n\nthe short version\n\n## Detail\n\nthe long version";
        let blocks = summary_blocks(body);
        assert_eq!(blocks.len(), 1);
        let (s, e) = blocks[0];
        assert!(body[s..e].contains("the short version"));
        assert!(
            !body[s..e].contains("the long version"),
            "summary block ran past its section: {:?}",
            &body[s..e]
        );
    }

    #[test]
    fn a_bare_summary_label_earns_no_chunk() {
        // A label with one line under it is not a summary. Embedding it spends a vector and a
        // ranking slot on nothing.
        let body = "# T\n\n## Summary\n\nshort.\n\n## Detail\n\nmore text here";
        let spans = chunk_for_embedding_spans("T", body, &WordTokenizer);
        assert!(
            !spans.iter().any(|s| s.is_summary),
            "a {SUMMARY_MIN_TOKENS}-token floor should have rejected this"
        );
    }

    #[test]
    fn a_short_document_that_is_already_its_own_summary_gets_no_duplicate() {
        // One window chunk barely wider than the block — a second vector of the same text buys
        // nothing and costs a ranking slot.
        let body = "## TL;DR\n\nthe whole point of the note is stated right here in one place \
                    and there is nothing else in the document at all, nothing";
        let spans = chunk_for_embedding_spans("T", body, &WordTokenizer);
        assert_eq!(spans.len(), 1, "{:?}", spans.iter().map(|s| s.is_summary).collect::<Vec<_>>());
    }

    #[test]
    fn summary_labels_match_decoration_but_not_a_topic() {
        assert!(is_summary_label("Key takeaways:"));
        assert!(is_summary_label("🔴 Key takeaways"));
        assert!(is_summary_label("TL;DR"));
        assert!(is_summary_label("summary"));
        assert!(is_summary_label("Key Points:"));
        // Named topics are ordinary prose sections, not dense summaries of their neighbours.
        assert!(!is_summary_label("Summary of the 2026-08 disk incident"));
        assert!(!is_summary_label("Architecture"));
        assert!(!is_summary_label(""));
    }

    #[test]
    fn parse_bold_label_needs_the_whole_line() {
        assert_eq!(parse_bold_label("**Key takeaways:**"), Some("Key takeaways:".into()));
        assert_eq!(parse_bold_label("  **TL;DR**  "), Some("TL;DR".into()));
        assert_eq!(parse_bold_label("**Summary**:"), Some("Summary".into()));
        // A sentence that merely OPENS in bold is not a section label.
        assert_eq!(parse_bold_label("**Note** that this is prose"), None);
        assert_eq!(parse_bold_label("plain text"), None);
        assert_eq!(parse_bold_label("****"), None);
    }

    #[test]
    fn parse_atx_heading_rejects_non_headings() {
        assert_eq!(parse_atx_heading("## Section"), Some((2, "Section".into())));
        assert_eq!(
            parse_atx_heading("   ### Indented"),
            Some((3, "Indented".into()))
        );
        assert_eq!(
            parse_atx_heading("## Closed ##"),
            Some((2, "Closed".into()))
        );
        assert_eq!(parse_atx_heading("####### TooDeep"), None);
        assert_eq!(parse_atx_heading("#nospace"), None);
        assert_eq!(parse_atx_heading("## "), None);
        assert_eq!(parse_atx_heading("plain text"), None);
    }


    // ── max-pool per doc ────────────────────────────────────────────────────
    //
    // The aggregation that makes per-chunk vectors usable. Without it a document appears
    // once per chunk in the ranking and one long note can occupy the entire top-k.

    #[test]
    fn a_document_that_fits_the_model_window_is_never_split() {
        // The floor exists because the two window constants are both far below the model's real
        // capacity: the chunker aims at CHUNK_TARGET_TOKENS (350) and CHUNK_BUDGET_CHARS caps
        // window growth at 1800 BYTES, while CHUNK_MAX_TOKENS (510) is what actually fits. A
        // ~1.9 KB note is ~450 tokens and fits whole, but the byte cap split it into two
        // overlapping halves, so neither vector held the whole document.
        //
        // WordTokenizer counts one token per whitespace-delimited word, so 400 words is 400
        // tokens plus the title prefix - comfortably inside 510 and comfortably OVER the
        // 1800-byte budget that used to force the split.
        let tok = WordTokenizer;
        let body = (0..400).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ");
        assert!(body.len() > CHUNK_BUDGET_CHARS,
                "body must exceed the byte budget or this test proves nothing, got {}", body.len());
        assert!(tok.count(&body) <= CHUNK_MAX_TOKENS,
                "body must fit the token window, got {}", tok.count(&body));

        let spans = chunk_for_embedding_spans("Title", &body, &tok);
        let tiling: Vec<_> = spans.iter().filter(|s| !s.is_summary).collect();
        assert_eq!(tiling.len(), 1, "a document that fits the window must be ONE chunk");
        assert_eq!(tiling[0].char_start, 0);
        assert_eq!(tiling[0].char_end, body.len(), "the single chunk must cover the WHOLE body");
        assert!(tok.count(&tiling[0].text) <= CHUNK_MAX_TOKENS, "and must still fit the window");

        // The floor must not swallow documents that genuinely need splitting.
        let long = (0..2000).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ");
        let many = chunk_for_embedding_spans("Title", &long, &tok);
        assert!(many.iter().filter(|s| !s.is_summary).count() > 1,
                "a document over the window must still be split");
        for s in &many {
            assert!(tok.count(&s.text) <= CHUNK_MAX_TOKENS, "every chunk still respects the ceiling");
        }
    }

    #[test]
    fn shrink_to_fit_always_terminates_around_multibyte_characters() {
        // 🔴 THIS IS A HANG REGRESSION, AND IT FIRED ON REAL VAULT CONTENT.
        //
        // `ceil_char_boundary` rounds UP, so the binary-search midpoint can land past `hi` when
        // it falls inside a multi-byte character. `hi = mid - 1` then leaves `hi` unchanged and
        // the search repeats one state forever. Measured 2026-08-20: the full-corpus backfill
        // stopped dead on `skills/fastmcp-server/references/features/sampling.md` (21 KB, 12
        // non-ASCII characters) with one core at 100%, zero bytes of I/O and no database write.
        //
        // The assertion is a CALL BUDGET rather than a plain "it returned", because the failure
        // mode is non-termination: without a budget a regression hangs the whole test suite
        // instead of failing it. A correct binary search over an 83-byte window needs ~7 probes.
        let body = format!("{}\u{2014}{}", "a".repeat(40), "b".repeat(40));
        let len = body.len();
        assert!(len > 80, "em dash must make this longer than its char count");

        let mut checked = 0usize;
        for start in 0..len - 1 {
            for end in (start + 1)..=len {
                for limit in 1..12usize {
                    let calls = std::cell::Cell::new(0usize);
                    let fits = |s: usize, m: usize| {
                        calls.set(calls.get() + 1);
                        assert!(
                            calls.get() < 400,
                            "shrink_to_fit did not terminate: start={start} end={end} limit={limit}"
                        );
                        m - s <= limit
                    };
                    let got = shrink_to_fit(&body, start, end, &fits);
                    assert!(got > start, "must advance past start");
                    assert!(got <= len, "must stay inside the body");
                    assert!(
                        body.is_char_boundary(got),
                        "must return a char boundary, got {got}"
                    );
                    checked += 1;
                }
            }
        }
        // The Python model of this same loop found 1,573 non-terminating (start, end, limit)
        // triples in this exact string before the fix, so the sweep is known to reach the bug.
        assert!(checked > 30_000, "sweep must actually cover the window, got {checked}");
    }

    #[test]
    fn best_per_doc_scores_a_document_by_its_best_chunk() {
        // doc 1's SECOND chunk is the match. The pooled-vector design could not express
        // this at all — it is the whole bug.
        let vectors = vec![
            (1i64, vec![0.0f32, 1.0]),
            (1i64, vec![1.0f32, 0.0]),
            (2i64, vec![0.7f32, 0.7]),
        ];
        let best = best_per_doc(&vectors, &[1.0, 0.0]);
        assert_eq!(best.len(), 2, "one score per document, not per chunk");
        assert!((best[&1] - 1.0).abs() < 1e-5);
        assert!(best[&1] > best[&2], "best chunk wins, not the average");
    }

    #[test]
    fn best_per_doc_of_an_empty_matrix_is_empty() {
        assert!(best_per_doc(&[], &[1.0, 0.0]).is_empty());
    }

    #[test]
    fn best_per_doc_survives_more_chunks_than_threads() {
        // The scoped-thread split partitions by row index and merges a partial map per
        // thread. If the merge were wrong, a doc split across two partitions would report
        // the last thread's score rather than the highest.
        let mut vectors: Vec<(i64, Vec<f32>)> = Vec::new();
        for i in 0..512 {
            // Only the very last chunk points at the query.
            let v = if i == 511 {
                vec![1.0f32, 0.0]
            } else {
                vec![0.0f32, 1.0]
            };
            vectors.push((7i64, v));
        }
        let best = best_per_doc(&vectors, &[1.0, 0.0]);
        assert_eq!(best.len(), 1);
        assert!((best[&7] - 1.0).abs() < 1e-5);
    }

    // ── chunking ────────────────────────────────────────────────────────────
    //
    // These exist because the bug they prevent was invisible for months: every document had a
    // vector, `recall status` reported 99.9% coverage, and 76.5% of the actual words had never
    // been seen by the model. Coverage was measuring the wrong thing — and then the fix for THAT
    // measured the wrong thing too, bounding bytes while the model bounds tokens.

    /// The real BGE tokenizer, or `None` when the model snapshot is not on this machine.
    ///
    /// Skipping is deliberate: these tests must not fail on a checkout that has never run
    /// `recall embed`. The pure-logic coverage above does not depend on it.
    fn real_tokenizer() -> Option<BgeTokenizer> {
        BgeTokenizer::load().ok()
    }

    /// Only the chunks that TILE the body. Summary chunks deliberately re-cover text the window
    /// chunks already cover, so `starts at 0` / `no gaps` / `reaches the end` are statements about
    /// this subset — asserting them over the whole list would be asserting something false.
    fn tiling(spans: Vec<ChunkSpan>) -> Vec<ChunkSpan> {
        spans.into_iter().filter(|s| !s.is_summary).collect()
    }

    #[test]
    fn short_document_becomes_one_chunk_carrying_its_title() {
        let c = chunk_for_embedding_spans("My Note", "hello world", &WordTokenizer);
        assert_eq!(c.len(), 1);
        assert!(c[0].text.starts_with("My Note\n\n"), "{:?}", c[0].text);
        assert!(c[0].text.contains("hello world"));
    }

    /// The whole point. A 20,000-char document used to become ONE truncated vector.
    #[test]
    fn long_document_is_split_and_every_chunk_keeps_the_title() {
        let para = "word ".repeat(100);
        let body = vec![para.as_str(); 40].join("\n\n");
        let chunks = chunk_for_embedding_spans("Design Doc", &body, &WordTokenizer);
        assert!(
            chunks.len() > 5,
            "expected many chunks, got {}",
            chunks.len()
        );
        for (i, c) in chunks.iter().enumerate() {
            assert!(
                c.text.starts_with("Design Doc\n\n"),
                "chunk {i} lost its title: {:?}",
                &c.text[..c.text.len().min(40)]
            );
        }
    }

    /// No content may be silently dropped - that is the exact failure being fixed.
    #[test]
    fn chunking_preserves_every_character_of_the_body() {
        let body = (0..300)
            .map(|i| format!("paragraph number {i} with some words in it"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let chunks = chunk_for_embedding_spans("T", &body, &WordTokenizer);
        for i in [0usize, 7, 150, 299] {
            let needle = format!("paragraph number {i} ");
            assert!(
                chunks.iter().any(|c| c.text.contains(&needle)),
                "lost {needle}"
            );
        }
    }

    /// Windows must TILE the body: contiguous (allowing for overlap), starting at 0 and reaching
    /// the end. A gap here is content that no vector covers.
    #[test]
    fn chunk_spans_tile_the_whole_body_without_gaps() {
        let body = (0..200)
            .map(|i| format!("line {i} of the document with several words"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let chunks = tiling(chunk_for_embedding_spans("T", &body, &WordTokenizer));
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].char_start, 0, "first chunk must start at 0");
        assert_eq!(
            chunks.last().unwrap().char_end,
            body.len(),
            "last chunk must reach the end of the body"
        );
        for w in chunks.windows(2) {
            assert!(
                w[1].char_start <= w[0].char_end,
                "gap between {}..{} and {}..{}",
                w[0].char_start,
                w[0].char_end,
                w[1].char_start,
                w[1].char_end
            );
            assert!(w[1].char_start > w[0].char_start, "window did not advance");
        }
    }

    /// Adjacent chunks must SHARE text, or a sentence on a boundary is retrievable from neither.
    #[test]
    fn adjacent_chunks_overlap() {
        let body = (0..200)
            .map(|i| format!("sentence {i} carries some distinct words"))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = tiling(chunk_for_embedding_spans("T", &body, &WordTokenizer));
        assert!(chunks.len() > 2, "need several chunks, got {}", chunks.len());
        for w in chunks.windows(2) {
            assert!(
                w[1].char_start < w[0].char_end,
                "no overlap: chunk ends at {} and the next starts at {}",
                w[0].char_end,
                w[1].char_start
            );
        }
    }

    #[test]
    fn a_single_oversized_paragraph_is_hard_split_not_dropped() {
        // A minified blob or base64 payload has no paragraph breaks. Refusing to split it would
        // silently drop it, which is the bug wearing a different hat.
        //
        // ⚠️ The assertion is COVERAGE, not chunk count. It used to require >= 2 chunks, but
        // `WordTokenizer` counts one token per whitespace-delimited word, so a 9,000-character
        // run with no spaces is exactly ONE token to it — which the whole-document floor
        // correctly emits as a single fully-covering span. The real invariant this test exists
        // to protect is that no content is dropped and nothing exceeds the ceiling; chunk count
        // was only ever a proxy for it. Under the REAL tokenizer this body is thousands of
        // tokens (`soften_long_runs` breaks the run up) and still hard-splits.
        let body = "y".repeat(9000);
        let chunks = chunk_for_embedding_spans("Blob", &body, &WordTokenizer);
        assert!(!chunks.is_empty(), "the blob produced no chunks at all");
        assert_eq!(chunks[0].char_start, 0);
        assert_eq!(chunks.last().unwrap().char_end, 9000, "content was lost");
        for c in &chunks {
            assert!(
                WordTokenizer.count(&c.text) <= CHUNK_MAX_TOKENS,
                "a chunk exceeded the model window"
            );
        }
    }

    #[test]
    fn hard_split_never_panics_on_multibyte_utf8() {
        let body = "日本語".repeat(3000);
        let chunks = chunk_for_embedding_spans("CJK", &body, &WordTokenizer);
        for c in &chunks {
            // The slice must be valid — this panics if a boundary landed mid-codepoint.
            let _ = &body[c.char_start..c.char_end];
        }
        assert_eq!(chunks.last().unwrap().char_end, body.len());
    }

    #[test]
    fn empty_body_still_yields_one_titled_chunk() {
        // Zero chunks would make the embed work queue re-select this document forever.
        let chunks = chunk_for_embedding_spans("Just A Title", "", &WordTokenizer);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Just A Title"));
    }

    // ── the token ceiling ───────────────────────────────────────────────────

    #[test]
    fn soften_long_runs_breaks_only_over_long_runs() {
        let prose = "the quick brown fox jumps over the lazy dog";
        assert_eq!(soften_long_runs(prose), prose, "prose must be untouched");

        let run = "a".repeat(250);
        let softened = soften_long_runs(&run);
        assert!(softened.contains(' '), "an over-long run must be broken");
        assert_eq!(
            softened.replace(' ', ""),
            run,
            "softening must not lose or alter a single character"
        );
        for word in softened.split_whitespace() {
            assert!(
                word.chars().count() <= MAX_WORD_CHARS,
                "piece is still {} chars",
                word.chars().count()
            );
        }
    }

    #[test]
    fn soften_long_runs_preserves_multibyte_characters() {
        let run = "日".repeat(250);
        let softened = soften_long_runs(&run);
        assert_eq!(softened.replace(' ', ""), run);
    }

    /// 🔴 The ceiling is the whole point of the rewrite. Real tokenizer, real vocabulary, and the
    /// content types measured at 1.31-1.92 bytes/token — the ones a byte budget cannot bound.
    #[test]
    fn real_tokenizer_never_exceeds_the_token_ceiling() {
        let Some(tok) = real_tokenizer() else {
            eprintln!("skipping: BGE model snapshot not present on this machine");
            return;
        };
        let cases: Vec<(&str, String)> = vec![
            (
                "prose",
                "the quick brown fox jumps over the lazy dog ".repeat(400),
            ),
            ("markdown table", "| col | 12.5 | 3.1 | x |\n".repeat(400)),
            (
                "minified json",
                r#"{"a":1,"bb":22,"ccc":333,"dddd":4444},"#.repeat(400),
            ),
            ("cjk", "日本語のテキストです。".repeat(400)),
            ("base64ish", "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5".repeat(200)),
            ("one huge unbroken run", "z".repeat(50_000)),
            ("bullets", "- **bold item**: some explanation here\n".repeat(300)),
            // Added with the heading-path prefix and the summary chunk: both ADD tokens to a
            // chunk, and both are therefore new ways to breach the ceiling.
            (
                "deeply nested headings",
                (0..40)
                    .map(|i| {
                        format!(
                            "# Level one heading number {i}\n## Level two heading number {i}\n### Level three heading number {i}\n#### Level four heading number {i}\n\nsome body text under it that is long enough to matter here\n"
                        )
                    })
                    .collect::<String>(),
            ),
            (
                "one absurd heading",
                format!(
                    "# {}\n\nbody text under an unreasonable heading. ",
                    "extremely long section name ".repeat(300)
                ) + &"more body words here. ".repeat(300),
            ),
            (
                "summary blocks everywhere",
                (0..20)
                    .map(|i| {
                        format!(
                            "## Section {i}\n\n{}\n\n**Key takeaways:**\n\n{}\n",
                            "ordinary prose in the section body. ".repeat(60),
                            "- a takeaway line that is reasonably long and specific\n".repeat(40),
                        )
                    })
                    .collect::<String>(),
            ),
            (
                "a summary block of minified json",
                format!(
                    "## TL;DR\n\n{}\n",
                    r#"{"a":1,"bb":22,"ccc":333,"dddd":4444},"#.repeat(400)
                ),
            ),
        ];
        for (name, body) in cases {
            let spans = chunk_for_embedding_spans("A Note Title", &body, &tok);
            assert!(!spans.is_empty(), "{name}: produced no chunks");
            // 🔴 THE GATE. Every chunk, tiling and summary alike, +2 for the `[CLS]`/`[SEP]` the
            // model adds — because 512 is what the model reads, and 510 is what we may hand it.
            for (i, s) in spans.iter().enumerate() {
                let n = tok.count(&s.text);
                assert!(
                    n <= CHUNK_MAX_TOKENS,
                    "{name}: chunk {i} (is_summary={}) is {n} tokens, over the {CHUNK_MAX_TOKENS} ceiling",
                    s.is_summary
                );
                assert!(
                    n + 2 <= 512,
                    "{name}: chunk {i} is {} tokens once [CLS]/[SEP] are added — the model will truncate it",
                    n + 2
                );
            }
            // And the body must still be covered end to end by the TILING chunks.
            let tiled = tiling(spans);
            assert_eq!(tiled[0].char_start, 0, "{name}: does not start at 0");
            assert_eq!(
                tiled.last().unwrap().char_end,
                body.len(),
                "{name}: does not reach the end"
            );
        }
    }

    /// The `[UNK]` collapse, in the real vocabulary: an unbroken 1800-byte run encodes to
    /// `['[CLS]', '[UNK]', '[SEP]']` — one marker, no content. Softened, it must produce real
    /// tokens instead.
    #[test]
    fn real_tokenizer_softening_rescues_an_unbroken_run() {
        let Some(tok) = real_tokenizer() else {
            eprintln!("skipping: BGE model snapshot not present on this machine");
            return;
        };
        let run = "a3f9b2c8".repeat(225); // 1800 bytes, no whitespace
        assert_eq!(tok.count(&run), 1, "precondition: the raw run is one [UNK]");
        let softened = soften_long_runs(&run);
        assert!(
            tok.count(&softened) > 50,
            "softening produced only {} tokens — still no signal",
            tok.count(&softened)
        );
    }

    /// The corpus-level measurement behind the token-budget rewrite. NOT part of the normal run —
    /// it walks the vault and takes minutes.
    ///
    /// ```text
    /// cargo test --features "voyage,gemini" -- --ignored --nocapture measure_vault_chunk_tokens
    /// ```
    ///
    /// Before the rewrite the same sample gave **25.74% of chunks over 512 tokens**. This asserts
    /// zero, so it is a regression gate and not just a report.
    #[test]
    #[ignore = "walks the vault; run explicitly with --ignored"]
    fn measure_vault_chunk_tokens() {
        let Some(tok) = real_tokenizer() else {
            eprintln!("skipping: BGE model snapshot not present");
            return;
        };
        let root = crate::store::vault_root();
        let mut files = Vec::new();
        collect_markdown(&root, &mut files, 0);
        files.sort();
        assert!(!files.is_empty(), "no markdown found under {root:?}");

        // Deterministic even sample — same docs on every run, no RNG to pin.
        let want = 1200usize;
        let step = (files.len() / want).max(1);
        let sample: Vec<_> = files.iter().step_by(step).take(want).collect();

        let (mut docs, mut chunks, mut over, mut max_tok) = (0usize, 0usize, 0usize, 0usize);
        let mut worst: Vec<(usize, String)> = Vec::new();
        // Added with improvements A and B, so the run reports whether they actually landed on the
        // real corpus and not only on the fixtures.
        let (mut with_heading, mut summary_chunks, mut docs_with_summary) = (0usize, 0usize, 0usize);
        for path in sample {
            let Ok(body) = std::fs::read_to_string(path) else {
                continue;
            };
            if body.len() > 512 * 1024 {
                continue; // recall skips these at ingest
            }
            let title = body
                .lines()
                .take(40)
                .find_map(|l| l.trim().strip_prefix("# ").map(|h| h.trim().to_string()))
                .unwrap_or_else(|| {
                    path.file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned()
                });
            docs += 1;
            let mut doc_had_summary = false;
            for span in chunk_for_embedding_spans(&title, &body, &tok) {
                let n = tok.count(&span.text) + 2; // +2 for [CLS]/[SEP] the model adds
                chunks += 1;
                max_tok = max_tok.max(n);
                if n > 512 {
                    over += 1;
                    worst.push((n, path.display().to_string()));
                }
                if !span.heading_path.is_empty() {
                    with_heading += 1;
                    // Improvement A is only real if the section reaches the TEXT. The column was
                    // already correct before this change and the model never saw it.
                    //
                    // Asserted on the DEEPEST component, not the whole chain: an over-budget
                    // chain legitimately loses its outer headings (see `chunk_prefix`), and
                    // asserting the full string is what turned that design decision into a
                    // failure the first time this ran.
                    let deepest = soften_long_runs(
                        span.heading_path.rsplit(" > ").next().unwrap_or_default(),
                    );
                    let label = span.text.split("\n\n").next().unwrap_or_default();
                    assert!(
                        label.contains(&deepest) || tok.count(&deepest) > PREFIX_MAX_TOKENS,
                        "section {deepest:?} is stored but not embedded in {}: label {:?}",
                        path.display(),
                        &label[..label.len().min(120)]
                    );
                }
                if span.is_summary {
                    summary_chunks += 1;
                    doc_had_summary = true;
                }
            }
            if doc_had_summary {
                docs_with_summary += 1;
            }
        }
        worst.sort_by(|a, b| b.0.cmp(&a.0));
        println!("\n=== vault chunk-token distribution ===");
        println!("  docs sampled        : {docs}");
        println!("  chunks produced     : {chunks}");
        println!("  chunks per doc      : {:.2}", chunks as f64 / docs.max(1) as f64);
        println!("  chunks over 512 tok : {over}  ({:.2}%)", 100.0 * over as f64 / chunks.max(1) as f64);
        println!("  largest chunk       : {max_tok} tokens");
        println!(
            "  chunks with heading : {with_heading}  ({:.2}%)",
            100.0 * with_heading as f64 / chunks.max(1) as f64
        );
        println!(
            "  summary chunks      : {summary_chunks}  (in {docs_with_summary} docs, {:.2}% of docs)",
            100.0 * docs_with_summary as f64 / docs.max(1) as f64
        );
        for (n, p) in worst.iter().take(5) {
            println!("     {n} tok  {p}");
        }
        assert_eq!(over, 0, "chunks still exceed the model's window");
    }

    /// The improvement-B proof on REAL notes: an enrichment `**Key takeaways:**` block must come
    /// back as a chunk of its own, leading with the label and free of the section it summarises.
    ///
    /// ```text
    /// cargo test --features "voyage,gemini" -- --ignored --nocapture measure_takeaway_blocks
    /// ```
    #[test]
    #[ignore = "walks the vault; run explicitly with --ignored"]
    fn measure_takeaway_blocks_become_their_own_chunk() {
        let Some(tok) = real_tokenizer() else {
            eprintln!("skipping: BGE model snapshot not present");
            return;
        };
        let root = crate::store::vault_root();
        let mut files = Vec::new();
        collect_markdown(&root, &mut files, 0);
        files.sort();

        let (mut examined, mut isolated, mut too_small) = (0usize, 0usize, 0usize);
        let mut first_example: Option<String> = None;
        for path in &files {
            if examined >= 200 {
                break;
            }
            let Ok(body) = std::fs::read_to_string(path) else {
                continue;
            };
            if body.len() > 512 * 1024 || !body.contains("**Key takeaways") {
                continue;
            }
            examined += 1;
            let title = body
                .lines()
                .take(40)
                .find_map(|l| l.trim().strip_prefix("# ").map(|h| h.trim().to_string()))
                .unwrap_or_default();
            let spans = chunk_for_embedding_spans(&title, &body, &tok);
            let hit = spans.iter().find(|s| {
                s.is_summary && body[s.char_start..].starts_with("**Key takeaways")
            });
            match hit {
                Some(s) => {
                    isolated += 1;
                    if first_example.is_none() {
                        first_example = Some(s.text.chars().take(300).collect());
                    }
                }
                // The floor is deliberate, not a miss: a label with one line under it is not a
                // summary. Counted separately so the two are never confused.
                None => too_small += 1,
            }
        }
        println!("\n=== `**Key takeaways` blocks ===");
        println!("  docs examined          : {examined}");
        println!("  got their own chunk    : {isolated}");
        println!("  below the {SUMMARY_MIN_TOKENS}-token floor : {too_small}");
        if let Some(e) = &first_example {
            println!("\n  --- first summary chunk, verbatim ---\n{e}\n");
        }
        assert!(examined > 0, "no vault note contained a `**Key takeaways` block");
        assert!(
            isolated > 0,
            "every `**Key takeaways` block was rejected — the label parser is not matching"
        );
    }

    fn collect_markdown(dir: &std::path::Path, out: &mut Vec<PathBuf>, depth: usize) {
        if depth > 12 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') {
                continue;
            }
            if p.is_dir() {
                collect_markdown(&p, out, depth + 1);
            } else if matches!(
                p.extension().and_then(|s| s.to_str()),
                Some("md") | Some("qmd")
            ) {
                out.push(p);
            }
        }
    }

    /// A title long enough to eat the whole budget must not livelock or emit a chunk over ceiling.
    #[test]
    fn real_tokenizer_survives_a_pathological_title() {
        let Some(tok) = real_tokenizer() else {
            eprintln!("skipping: BGE model snapshot not present on this machine");
            return;
        };
        let title = "extremely verbose heading ".repeat(300); // past the window on its own
        let body = "some ordinary body text here. ".repeat(200);
        let spans = chunk_for_embedding_spans(&title, &body, &tok);
        assert!(!spans.is_empty());
        let last_tiled = spans.iter().filter(|s| !s.is_summary).next_back().unwrap();
        assert_eq!(last_tiled.char_end, body.len());
        // 🔴 The ceiling assertion here is NEW, and so is the 300× title.
        //
        // Measured 2026-08-19 by setting `PREFIX_MAX_TOKENS` to 100_000 and re-running: at the
        // old 120× title the chunker stayed (just) inside the window, which is why nothing caught
        // this. At 300× it does not — `shrink_to_fit` can only shorten the PASSAGE, so with an
        // unbounded prefix the `.max(32)` floor makes `fits` permanently false, every chunk
        // collapses to one character, and the run does not finish (>12 min, killed) instead of
        // failing cleanly. Same silent truncation as the byte-budget bug, arriving through the
        // title instead of the body. [`PREFIX_MAX_TOKENS`] is the fix;
        // `a_pathological_prefix_cannot_eat_the_whole_window` is the fast unit gate on it.
        for (i, s) in spans.iter().enumerate() {
            let n = tok.count(&s.text);
            assert!(
                n <= CHUNK_MAX_TOKENS,
                "chunk {i} is {n} tokens — the title ate the window"
            );
        }
    }


    #[test]
    fn mean_pool_averages_and_renormalises() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let m = mean_pool(&[a, b]);
        let norm: f32 = m.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "not unit length: {norm}");
        assert!((m[0] - m[1]).abs() < 1e-6, "should be symmetric");
        assert!(m[2].abs() < 1e-6);
    }

    #[test]
    fn mean_pool_of_one_vector_is_that_vector() {
        let v = vec![0.6f32, 0.8, 0.0];
        assert_eq!(mean_pool(std::slice::from_ref(&v)), v);
    }

    #[test]
    fn ort_version_compatibility_rejects_known_bad_120x() {
        assert!(!ort_version_compatible("1.20.1"));
        assert!(!ort_version_compatible("1.23.2"));
    }

    #[test]
    fn ort_version_compatibility_accepts_124_plus() {
        assert!(ort_version_compatible("1.24.0"));
        assert!(ort_version_compatible("1.24.4"));
        assert!(ort_version_compatible("1.27.0"));
    }

    #[test]
    fn ort_version_compatibility_rejects_unparseable() {
        assert!(!ort_version_compatible("unknown"));
        assert!(!ort_version_compatible("1"));
    }


    #[test]
    fn scored_search_returns_descending_cosines() {
        let per_doc: HashMap<i64, f32> = [(1, 0.2), (2, 0.9), (3, 0.5), (4, 0.9)].into();
        let got = exact_scored(per_doc, 3);
        assert_eq!(got, vec![(2, 0.9), (4, 0.9), (3, 0.5)], "desc by cosine, ties by id");
    }

    #[test]
    fn load_fails_loud_when_ort_dll_missing() {
        // The old behaviour was Ok(None) + an eprintln, after which the rescue tier served
        // keyword hits under the bge name with exit 0.
        let prev = std::env::var_os("ORT_DYLIB_PATH");
        std::env::set_var("ORT_DYLIB_PATH", r"C:\definitely\not\here\onnxruntime.dll");
        let store = crate::store::Store::open_at(std::path::Path::new(":memory:")).unwrap();
        let r = BgeSearch::load(&store);
        match prev {
            Some(v) => std::env::set_var("ORT_DYLIB_PATH", v),
            None => std::env::remove_var("ORT_DYLIB_PATH"),
        }
        let err = match r {
            Err(e) => format!("{e:#}"),
            Ok(_) => panic!("load must fail when the ONNX Runtime DLL is missing"),
        };
        assert!(err.contains("ONNX Runtime"), "{err}");
    }
}
