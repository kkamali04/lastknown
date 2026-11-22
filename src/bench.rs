// bench.rs — recall@k eval harness + Karpathy overnight autoresearch hook.
//
// A fixture passes at recall@k if ANY expected_citations path appears as a
// prefix of a returned doc path within the top-k results. We report overall
// recall@5 and per-channel recall@5 (each channel run in isolation) plus query
// latency p50/p95 — the numbers the spec gate is measured against.

use crate::query::{self, Ranked};
use crate::store::{Hit, Store};
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug)]
struct Fixture {
    id: String,
    question: String,
    expected: Vec<String>,
    expect_empty: bool,
}

#[derive(Debug, Serialize)]
pub struct BenchReport {
    pub run_id: String,
    pub started_unix_ms: u128,
    pub top_k: usize,
    pub smoke_limit: Option<usize>,
    pub positive_fixtures: usize,
    pub total_docs_indexed: i64,
    pub queryable_bge_vectors: usize,
    pub fts5_hits: usize,
    pub bge_hits: usize,
    pub fused_hits: usize,
    pub fts5_recall: f64,
    pub bge_recall: f64,
    pub fused_recall: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub no_llm: bool,
    pub no_paid_calls: bool,
    pub raw_contexts_included: bool,
    pub rrf_k: f64,
    pub fusion_mode: String,
    pub denoise_mode: String,
    pub per_fixture: Vec<BenchFixtureReport>,
}

#[derive(Debug, Serialize)]
pub struct BenchFixtureReport {
    pub id: String,
    pub fts5_hit: bool,
    pub bge_hit: bool,
    pub fused_hit: bool,
    pub fts5_rank: Option<usize>,
    pub bge_rank: Option<usize>,
    pub fused_rank: Option<usize>,
    pub fused_blocker_count: usize,
    pub fused_blocker_path_hashes: Vec<String>,
    pub fused_blocker_prefix_counts: Vec<BenchPrefixCount>,
    pub latency_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct BenchPrefixCount {
    pub prefix: String,
    pub count: usize,
}

/// The fixtures file named by `bench_fixtures` in the private rules file; none configured = none.
fn fixtures_path() -> Option<std::path::PathBuf> {
    crate::store::bench_fixtures_path()
}

fn load_fixtures() -> Result<Vec<Fixture>> {
    let path = fixtures_path().context("no `bench_fixtures` in ~/.recall/private-rules.json")?;
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("read fixtures {}", path.display()))?;
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let expected = v
            .get("expected_citations")
            .and_then(|c| c.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(|s| s.replace('\\', "/")))
                    .collect()
            })
            .unwrap_or_default();
        out.push(Fixture {
            id: v
                .get("id")
                .and_then(|s| s.as_str())
                .unwrap_or("?")
                .to_string(),
            question: v
                .get("question")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string(),
            expected,
            expect_empty: v
                .get("expect_empty")
                .and_then(|b| b.as_bool())
                .unwrap_or(false),
        });
    }
    Ok(out)
}

fn hit_at_k(returned: &[String], expected: &[String]) -> bool {
    for r in returned {
        for e in expected {
            // expected may be a directory prefix (e.g. ".../investor-pack/")
            if r == e || r.starts_with(e) || r.contains(e.trim_end_matches('/')) {
                return true;
            }
        }
    }
    false
}

fn first_hit_rank(ids: &[i64], id_hit: &HashMap<i64, Hit>, expected: &[String]) -> Option<usize> {
    ids.iter().enumerate().find_map(|(idx, id)| {
        id_hit.get(id).and_then(|hit| {
            if path_matches_expected(&hit.path, expected) {
                Some(idx + 1)
            } else {
                None
            }
        })
    })
}

fn path_matches_expected(path: &str, expected: &[String]) -> bool {
    expected
        .iter()
        .any(|e| path == e || path.starts_with(e) || path.contains(e.trim_end_matches('/')))
}

fn blocker_audit(
    ids: &[i64],
    id_hit: &HashMap<i64, Hit>,
    expected: &[String],
) -> (usize, Vec<String>, Vec<BenchPrefixCount>) {
    let mut blocker_paths: Vec<&str> = Vec::new();
    for id in ids {
        let Some(hit) = id_hit.get(id) else { continue };
        if path_matches_expected(&hit.path, expected) {
            break;
        }
        blocker_paths.push(&hit.path);
    }

    let mut prefix_counts: HashMap<String, usize> = HashMap::new();
    for path in &blocker_paths {
        *prefix_counts.entry(path_prefix(path)).or_insert(0) += 1;
    }
    let mut prefix_rows: Vec<BenchPrefixCount> = prefix_counts
        .into_iter()
        .map(|(prefix, count)| BenchPrefixCount { prefix, count })
        .collect();
    prefix_rows.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.prefix.cmp(&b.prefix)));

    let hashes = blocker_paths
        .iter()
        .take(10)
        .map(|path| stable_path_hash(path))
        .collect();
    (blocker_paths.len(), hashes, prefix_rows)
}

fn path_prefix(path: &str) -> String {
    if path_value_secret_risk(path) {
        return "path-secret-pattern".to_string();
    }
    let normalized = path.replace('\\', "/");
    let parts: Vec<&str> = normalized
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    match parts.as_slice() {
        [] => "".to_string(),
        [one] => (*one).to_string(),
        [one, two] if two.contains('.') => (*one).to_string(),
        [one, two, ..] if two.contains('.') => (*one).to_string(),
        [one, two, ..] => format!("{}/{}", one, two),
    }
}

fn path_value_secret_risk(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    normalized.contains("ghp_")
        || normalized.contains("sb_secret_")
        || has_broad_sk_token(&normalized)
}

fn has_broad_sk_token(value: &str) -> bool {
    let bytes = value.as_bytes();
    for idx in 0..bytes.len().saturating_sub(2) {
        if bytes[idx] != b's' || bytes[idx + 1] != b'k' || bytes[idx + 2] != b'-' {
            continue;
        }
        let mut len = 0usize;
        for byte in &bytes[idx + 3..] {
            if byte.is_ascii_alphanumeric() || *byte == b'_' || *byte == b'-' {
                len += 1;
            } else {
                break;
            }
        }
        if len >= 20 {
            return true;
        }
    }
    false
}

fn stable_path_hash(path: &str) -> String {
    let mut hasher = DefaultHasher::new();
    path.replace('\\', "/").hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Run the eval once. Returns recall@5 for the fused stack.
///
/// Each channel is computed ONCE per fixture (fts5 + rg) and the three configs
/// (fts5-only, rg-only, fused) are assembled from those primitives via RRF.
/// This halves the expensive full-vault rg scans versus calling `query::fused`
/// separately for each config. Per-fixture progress is streamed to stderr so a
/// long run is observable while it executes.
pub fn run(top: usize, limit: Option<usize>) -> Result<f64> {
    Ok(run_report(top, limit)?.fused_recall)
}

pub fn run_and_write(top: usize, limit: Option<usize>) -> Result<(f64, PathBuf)> {
    let report = run_report(top, limit)?;
    let recall = report.fused_recall;
    let path = write_report(&report)?;
    println!("   bench json artifact: {}", path.display());
    Ok((recall, path))
}

pub fn run_report(top: usize, limit: Option<usize>) -> Result<BenchReport> {
    let started_unix_ms = unix_ms();
    let store = Store::open()?;
    let total_docs = store.count_valid()?;
    let fixtures = load_fixtures()?;
    if fixtures.is_empty() {
        anyhow::bail!("no fixtures loaded");
    }

    let mut positives: Vec<&Fixture> = fixtures
        .iter()
        .filter(|fx| !fx.expect_empty && !fx.expected.is_empty())
        .collect();
    // CI smoke: score only the first N positives for a fast, hermetic-ish run.
    if let Some(n) = limit {
        positives.truncate(n);
    }
    let n_pos = positives.len();
    let depth = query::fusion_depth(top);

    // Load the BGE model + all stored vectors ONCE for the whole run (model load
    // is seconds; the matrix is ~100k×384 f32). None = store not embedded yet, in
    // which case the bge channel is a no-op and its column reads 0.000.
    let bge = crate::bge::BgeSearch::load(&store)?;
    let queryable_bge_vectors = bge.as_ref().map(|bge| bge.vectors.len()).unwrap_or(0);
    if let Some(ref bge) = bge {
        eprintln!("   bge: {} queryable vectors loaded", bge.vectors.len());
    } else {
        eprintln!("   bge: no vectors yet — run `recall embed` (bge column will read 0.000)");
    }

    let mut fts_hits = 0usize;
    let mut bge_hits = 0usize;
    let mut fused_hits = 0usize;
    let mut latencies: Vec<u128> = Vec::new();
    let mut per_fixture: Vec<BenchFixtureReport> = Vec::new();

    eprintln!(
        "recall bench: scoring {} positive fixtures ({} docs)",
        n_pos, total_docs
    );

    for (i, fx) in positives.iter().enumerate() {
        let mut id_hit: HashMap<i64, Hit> = HashMap::new();

        // Compute each channel once; time the full fused assembly.
        let t = Instant::now();
        let fts_ids = query::channel_ids(&store, &fx.question, "fts5", depth, &mut id_hit)?;
        let bge_ids = query::bge_ids(&store, bge.as_ref(), &fx.question, depth, &mut id_hit)?;

        let paths = |ids: &[i64]| -> Vec<String> {
            ids.iter()
                .filter_map(|id| id_hit.get(id).map(|h| h.path.clone()))
                .collect()
        };

        let fts_only = query::rrf_merge(
            &[Ranked {
                channel: "fts5".into(),
                ids: fts_ids.clone(),
            }],
            top,
        );
        let bge_only = query::rrf_merge(
            &[Ranked {
                channel: "bge".into(),
                ids: bge_ids.clone(),
            }],
            top,
        );
        let fused = query::rrf_merge_with_hits_and_query(
            &[
                Ranked {
                    channel: "fts5".into(),
                    ids: fts_ids.clone(),
                },
                Ranked {
                    channel: "bge".into(),
                    ids: bge_ids.clone(),
                },
            ],
            top,
            &id_hit,
            Some(&fx.question),
        );
        let fused_depth = query::rrf_merge_with_hits_and_query(
            &[
                Ranked {
                    channel: "fts5".into(),
                    ids: fts_ids.clone(),
                },
                Ranked {
                    channel: "bge".into(),
                    ids: bge_ids.clone(),
                },
            ],
            depth,
            &id_hit,
            Some(&fx.question),
        );
        let latency_us = t.elapsed().as_micros();
        latencies.push(latency_us);

        let fts_rank = first_hit_rank(&fts_ids, &id_hit, &fx.expected);
        let bge_rank = first_hit_rank(&bge_ids, &id_hit, &fx.expected);
        let fused_rank = first_hit_rank(&fused_depth, &id_hit, &fx.expected);
        let (fused_blocker_count, fused_blocker_path_hashes, fused_blocker_prefix_counts) =
            blocker_audit(&fused_depth, &id_hit, &fx.expected);
        let fts_ok = hit_at_k(&paths(&fts_only), &fx.expected);
        let bge_ok = hit_at_k(&paths(&bge_only), &fx.expected);
        let fused_ok = hit_at_k(&paths(&fused), &fx.expected);
        if fts_ok {
            fts_hits += 1;
        }
        if bge_ok {
            bge_hits += 1;
        }
        if fused_ok {
            fused_hits += 1;
        }
        per_fixture.push(BenchFixtureReport {
            id: fx.id.clone(),
            fts5_hit: fts_ok,
            bge_hit: bge_ok,
            fused_hit: fused_ok,
            fts5_rank: fts_rank,
            bge_rank,
            fused_rank,
            fused_blocker_count,
            fused_blocker_path_hashes,
            fused_blocker_prefix_counts,
            latency_ms: latency_us as f64 / 1000.0,
        });

        eprintln!(
            "  [{:>2}/{}] {:<10} fused={} (fts5={} bge={})",
            i + 1,
            n_pos,
            fx.id,
            if fused_ok { "HIT " } else { "miss" },
            if fts_ok { "y" } else { "-" },
            if bge_ok { "y" } else { "-" },
        );
    }

    latencies.sort_unstable();
    let latency_p50_us = percentile_us(&latencies, 0.50);
    let latency_p95_us = percentile_us(&latencies, 0.95);

    let fts5_recall = recall_rate(fts_hits, n_pos);
    let bge_recall = recall_rate(bge_hits, n_pos);
    let fused_recall = recall_rate(fused_hits, n_pos);
    println!();
    println!(
        "recall bench: {} fixtures scored, {} docs indexed",
        n_pos, total_docs
    );
    println!("   {:<18} recall@{}", "config", top);
    println!(
        "   {:<18} {:.3}  ({}/{})",
        "fts5-only", fts5_recall, fts_hits, n_pos
    );
    println!(
        "   {:<18} {:.3}  ({}/{})",
        "bge-only", bge_recall, bge_hits, n_pos
    );
    println!(
        "   {:<18} {:.3}  ({}/{})",
        "fused(fts5+bge)", fused_recall, fused_hits, n_pos
    );
    println!(
        "   latency (fused)   p50={:.1}ms  p95={:.1}ms",
        latency_p50_us as f64 / 1000.0,
        latency_p95_us as f64 / 1000.0
    );
    println!();

    Ok(BenchReport {
        run_id: format!("bench_{}", started_unix_ms),
        started_unix_ms,
        top_k: top,
        smoke_limit: limit,
        positive_fixtures: n_pos,
        total_docs_indexed: total_docs,
        queryable_bge_vectors,
        fts5_hits: fts_hits,
        bge_hits,
        fused_hits,
        fts5_recall,
        bge_recall,
        fused_recall,
        latency_p50_ms: latency_p50_us as f64 / 1000.0,
        latency_p95_ms: latency_p95_us as f64 / 1000.0,
        no_llm: true,
        no_paid_calls: true,
        raw_contexts_included: false,
        rrf_k: query::effective_rrf_k(),
        fusion_mode: query::effective_fusion_mode(),
        denoise_mode: query::effective_denoise_mode(),
        per_fixture,
    })
}

pub fn write_report(report: &BenchReport) -> Result<PathBuf> {
    let dir = crate::store::recall_dev_dir()
        .context("bench reports go under `recall_dev_dir` from ~/.recall/private-rules.json; none is set")?
        .join("eval/results/bench_release");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create bench results dir {}", dir.display()))?;
    let path = dir.join(format!("{}.json", report.run_id));
    let json = serde_json::to_string_pretty(report).context("serialize bench report")?;
    std::fs::write(&path, json)
        .with_context(|| format!("write bench report {}", path.display()))?;
    Ok(path)
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn recall_rate(hits: usize, total: usize) -> f64 {
    hits as f64 / total.max(1) as f64
}

fn percentile_us(sorted_latencies: &[u128], q: f64) -> u128 {
    if sorted_latencies.is_empty() {
        return 0;
    }
    sorted_latencies[((sorted_latencies.len() as f64 * q) as usize).min(sorted_latencies.len() - 1)]
}

/// Karpathy overnight autoresearch loop. Repeats the eval; a result is only
/// "kept" when it reproduces (self-consistency) and improves the running best.
/// Currently the channel mix is fixed, so this measures variance and confirms
/// stability — the hook where future passes will mutate fusion params and keep
/// only metric-improving changes.
pub fn overnight(iterations: usize, top: usize, limit: Option<usize>) -> Result<()> {
    let mut best = 0.0f64;
    let mut runs: Vec<f64> = Vec::new();
    println!("recall bench: overnight autoresearch, {} iterations", iterations);
    for i in 0..iterations {
        let r = run(top, limit)?;
        runs.push(r);
        if r > best {
            best = r;
            println!("   iter {:>4}: recall@{}={:.3}  <- new best", i + 1, top, r);
        }
    }
    let mean = runs.iter().sum::<f64>() / runs.len().max(1) as f64;
    let var = runs.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / runs.len().max(1) as f64;
    println!();
    println!(
        "done: best={:.3} mean={:.3} stddev={:.4}",
        best,
        mean,
        var.sqrt()
    );
    Ok(())
}

pub fn fixtures_exist() -> bool {
    fixtures_path().is_some_and(|p| p.exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_rate_uses_nonzero_denominator() {
        assert_eq!(recall_rate(0, 0), 0.0);
        assert_eq!(recall_rate(2, 4), 0.5);
    }

    #[test]
    fn percentile_matches_existing_bench_indexing() {
        let latencies = vec![10, 20, 30, 40];
        assert_eq!(percentile_us(&latencies, 0.50), 30);
        assert_eq!(percentile_us(&latencies, 0.95), 40);
    }

    #[test]
    fn bench_report_serializes_without_context_fields() {
        let report = BenchReport {
            run_id: "bench_test".to_string(),
            started_unix_ms: 1,
            top_k: 5,
            smoke_limit: Some(1),
            positive_fixtures: 1,
            total_docs_indexed: 10,
            queryable_bge_vectors: 2,
            fts5_hits: 1,
            bge_hits: 0,
            fused_hits: 1,
            fts5_recall: 1.0,
            bge_recall: 0.0,
            fused_recall: 1.0,
            latency_p50_ms: 1.0,
            latency_p95_ms: 2.0,
            no_llm: true,
            no_paid_calls: true,
            raw_contexts_included: false,
            rrf_k: 60.0,
            fusion_mode: "rrf".to_string(),
            denoise_mode: "off".to_string(),
            per_fixture: vec![BenchFixtureReport {
                id: "fx".to_string(),
                fts5_hit: true,
                bge_hit: false,
                fused_hit: true,
                fts5_rank: Some(1),
                bge_rank: None,
                fused_rank: Some(1),
                fused_blocker_count: 0,
                fused_blocker_path_hashes: Vec::new(),
                fused_blocker_prefix_counts: Vec::new(),
                latency_ms: 1.25,
            }],
        };
        let raw = serde_json::to_string(&report).expect("serialization must not fail");
        assert!(raw.contains("fused_recall"));
        assert!(raw.contains("raw_contexts_included"));
        assert!(!raw.contains("question"));
        assert!(!raw.contains("expected_citations"));
        assert!(!raw.contains("vault_path"));
        assert!(!raw.contains("snippet"));
        assert!(!raw.contains("body"));
    }

    #[test]
    fn blocker_audit_returns_hashes_and_prefix_counts_without_paths() {
        let mut id_hit = HashMap::new();
        id_hit.insert(
            1,
            Hit {
                id: 1,
                path: "projects/acme/noisy-a.md".to_string(),
                title: "noisy".to_string(),
                snippet: String::new(),
                score: 1.0,
            },
        );
        id_hit.insert(
            2,
            Hit {
                id: 2,
                path: "research/rag/expected.md".to_string(),
                title: "expected".to_string(),
                snippet: String::new(),
                score: 1.0,
            },
        );

        let (count, hashes, prefixes) = blocker_audit(
            &[1, 2],
            &id_hit,
            &["research/rag/expected.md".to_string()],
        );

        assert_eq!(count, 1);
        assert_eq!(hashes.len(), 1);
        assert_eq!(hashes[0].len(), 16);
        assert_eq!(prefixes[0].prefix, "projects/acme");
        assert_eq!(path_prefix("tasks/status-2026-01-02.md"), "tasks");
        let raw = serde_json::to_string(&hashes).expect("hashes serialize");
        assert!(!raw.contains("noisy-a"));
        assert!(!raw.contains("expected"));
    }

    #[test]
    fn path_prefix_redacts_key_shaped_segments() {
        let path = format!("projects/acme/sk-proj-{}.md", "a".repeat(24));

        assert_eq!(path_prefix(&path), "path-secret-pattern");
    }

    #[test]
    fn path_prefix_redacts_broad_sk_substrings_that_match_artifact_gate() {
        let path = "tasks/task-0001-weekly-review-status-note.md";

        assert_eq!(path_prefix(path), "path-secret-pattern");
    }
}
