// eval.rs — the measuring stick for recall quality.
//
// "100% recall" only means something if we can measure it. This module gives
// the metric: recall@k of an approximate ranking (e.g. the HNSW ANN path)
// against the exact brute-force ground truth. We only trust the fast path if
// it returns the same ids the exact O(N) scan would — recall@k == 1.0.
//
// The exact scan (bge::BgeSearch::exact_search_ids) is the ground truth: it
// compares the query against every stored vector, so it cannot miss anything
// that is embedded. Coverage (every doc embedded) + exact scan = 100% recall
// by construction. The ANN path trades some of that for speed; this metric is
// how we prove how much, if any, is lost.

/// recall@k of `approx` against `truth`, both ranked best-first.
///
/// Returns `|approx[..k] ∩ truth[..k]| / min(k, truth.len())` in `[0.0, 1.0]`.
/// `1.0` means the approximate ranking missed nothing the exact scan found in
/// its top-k. Order within the top-k does not matter — this measures set
/// overlap (recall), not rank correlation.
pub fn recall_at_k(approx: &[i64], truth: &[i64], k: usize) -> f32 {
    let denom = k.min(truth.len());
    if denom == 0 {
        // Nothing to recall (empty truth) — vacuously perfect.
        return 1.0;
    }
    use std::collections::HashSet;
    let truth_set: HashSet<i64> = truth.iter().take(k).copied().collect();
    // Set intersection, not multiplicity. The documented formula is
    // |approx[..k] ∩ truth[..k]|, but counting filtered items would tally a
    // repeated id in `approx` twice — silently inflating recall above the true
    // overlap (and even past 1.0). Dedupe approx's top-k first so the metric,
    // our only measuring stick, can never over-report a win.
    let approx_set: HashSet<i64> = approx.iter().take(k).copied().collect();
    let hits = approx_set.intersection(&truth_set).count();
    hits as f32 / denom as f32
}

/// Generic semantic queries for `recall eval` (fast-path fidelity). A corpus-specific list can be
/// set as `eval_queries` in the private rules file; it replaces this one.
const DEFAULT_QUERIES: &[&str] = &[
    "memory architecture",
    "BGE embeddings",
    "vector store",
    "full text search ranking",
    "note encryption",
    "task board",
    "meeting notes",
    "project plan",
];

pub fn fixture_queries() -> Vec<String> {
    let private = crate::store::private_eval_queries();
    if private.is_empty() {
        DEFAULT_QUERIES.iter().map(|q| q.to_string()).collect()
    } else {
        private.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_rankings_is_one() {
        assert_eq!(recall_at_k(&[1, 2, 3], &[1, 2, 3], 3), 1.0);
    }

    #[test]
    fn order_within_topk_does_not_matter() {
        assert_eq!(recall_at_k(&[3, 2, 1], &[1, 2, 3], 3), 1.0);
    }

    #[test]
    fn one_miss_is_partial() {
        // approx returns 9 instead of 3 → 2 of 3 recalled.
        let r = recall_at_k(&[1, 2, 9], &[1, 2, 3], 3);
        assert!((r - 2.0 / 3.0).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn empty_truth_is_vacuously_one() {
        assert_eq!(recall_at_k(&[], &[], 5), 1.0);
    }

    #[test]
    fn k_larger_than_lists_uses_truth_len() {
        // truth has 2 ids, k=5 → denom=2; both found → 1.0.
        assert_eq!(recall_at_k(&[1, 2], &[1, 2], 5), 1.0);
    }

    #[test]
    fn total_miss_is_zero() {
        assert_eq!(recall_at_k(&[7, 8, 9], &[1, 2, 3], 3), 0.0);
    }

    #[test]
    fn duplicate_approx_id_does_not_inflate() {
        // approx repeats id 1; truth has two distinct ids. Only id 1 is truly
        // recalled, so recall must be 1/2 — NOT 2/2 from counting the dup
        // twice. Guards the metric against silently over-reporting a win.
        let r = recall_at_k(&[1, 1], &[1, 2], 2);
        assert!((r - 0.5).abs() < 1e-6, "got {r}");
    }
}
