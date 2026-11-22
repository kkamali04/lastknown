// dream.rs — RecallDreamer: offline memory consolidation ("dreaming").
//
// Design: dreaming is what a brain does while idle — it replays the day, drops
// duplicates, and reconciles contradictions. RecallDreamer maps that onto two
// concrete, fully-offline operations (no LLM, no cloud):
//
//   1. Reconcile near-duplicates  (mem0 extract-then-reconcile write path)
//      For each currently-valid observation, find older observations with high
//      token overlap. The newer one wins; the older is SUPERSEDED, not deleted.
//
//   2. Bi-temporal supersede      (Zep / Graphiti validity)
//      Superseding sets valid_to = now and superseded_by = winner_id. History
//      is preserved (auditable, reversible) but drops out of valid-only recall.
//
// Insight extraction (heuristic clustering of related memories into a summary
// node) is stubbed for a later pass. RecallDreamer runs standalone via
// `recall dream` and is folded into `recall bench --overnight` idle phases.

use crate::store::Store;
use anyhow::Result;
use std::collections::HashSet;

/// Jaccard similarity at/above this threshold = near-duplicate.
const DUP_THRESHOLD: f64 = 0.82;

pub struct DreamReport {
    pub scanned: usize,
    pub superseded: usize,
}

/// Run one consolidation pass over currently-valid, non-vault observations.
/// Vault docs (source='vault') are excluded — they are ground truth, not
/// memories to reconcile.
pub fn consolidate(store: &Store) -> Result<DreamReport> {
    let rows = load_observations(store)?;
    let mut superseded = 0usize;
    let mut dead: HashSet<i64> = HashSet::new();

    for i in 0..rows.len() {
        if dead.contains(&rows[i].0) {
            continue;
        }
        let toks_i = tokenize(&rows[i].1);
        // Compare against OLDER rows only (rows is newest-first), so the newer
        // memory is always the survivor.
        for j in (i + 1)..rows.len() {
            // Never across scopes (tenancy): a newer private duplicate must not retire a shared
            // memory other readers rely on (Codex tenancy r1).
            if dead.contains(&rows[j].0) || rows[j].2 != rows[i].2 {
                continue;
            }
            let toks_j = tokenize(&rows[j].1);
            if jaccard(&toks_i, &toks_j) >= DUP_THRESHOLD {
                supersede(store, rows[j].0, rows[i].0)?;
                dead.insert(rows[j].0);
                superseded += 1;
            }
        }
    }

    Ok(DreamReport {
        scanned: rows.len(),
        superseded,
    })
}

/// A cluster of related memories surfaced by `recall insights`.
pub struct Insight {
    pub size: usize,
    pub keywords: Vec<String>,
    pub representative: String,
}

/// Offline insight extraction: greedily cluster valid observations by token
/// overlap, then describe each cluster by its most frequent shared keywords.
/// No LLM — pure set math, so it runs fully offline in milliseconds.
pub fn insights(store: &Store, min_cluster: usize) -> Result<Vec<Insight>> {
    let rows = load_observations(store)?;
    let toks: Vec<HashSet<String>> = rows.iter().map(|(_, b, _)| tokenize(b)).collect();
    let mut used = vec![false; rows.len()];
    let mut out = Vec::new();

    for i in 0..rows.len() {
        if used[i] {
            continue;
        }
        let mut members = vec![i];
        used[i] = true;
        for j in (i + 1)..rows.len() {
            if used[j] {
                continue;
            }
            // looser than dedupe threshold — we want themes, not just dupes
            if jaccard(&toks[i], &toks[j]) >= 0.30 {
                members.push(j);
                used[j] = true;
            }
        }
        if members.len() < min_cluster {
            continue;
        }
        // keyword frequency across the cluster
        let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for &m in &members {
            for t in &toks[m] {
                if t.len() >= 4 {
                    *freq.entry(t.clone()).or_insert(0) += 1;
                }
            }
        }
        let mut kw: Vec<(String, usize)> = freq.into_iter().collect();
        kw.sort_by_key(|b| std::cmp::Reverse(b.1));
        let keywords: Vec<String> = kw.into_iter().take(5).map(|(k, _)| k).collect();
        out.push(Insight {
            size: members.len(),
            keywords,
            representative: rows[members[0]].1.chars().take(100).collect(),
        });
    }
    out.sort_by_key(|b| std::cmp::Reverse(b.size));
    Ok(out)
}

/// Cap on observations loaded per dream pass. The O(n²) Jaccard scan would
/// OOM or hang for minutes on the full ~101k-doc corpus without this bound.
const DREAM_LOAD_LIMIT: usize = 10_000;

/// (id, body, scope) of valid observations, newest-first.
fn load_observations(store: &Store) -> Result<Vec<(i64, String, String)>> {
    let mut stmt = store.conn.prepare(
        "SELECT id, body, scope FROM documents
         WHERE valid_to IS NULL AND source != 'vault'
         ORDER BY created_at DESC, id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(rusqlite::params![DREAM_LOAD_LIMIT as i64], |r| {
        Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
    })?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}

/// Mark `loser` superseded by `winner` (bi-temporal — no delete).
fn supersede(store: &Store, loser: i64, winner: i64) -> Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        .to_string();
    store.conn.execute(
        "UPDATE documents SET valid_to = ?1, superseded_by = ?2 WHERE id = ?3",
        rusqlite::params![now, winner, loser],
    )?;
    Ok(())
}

fn tokenize(s: &str) -> HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_lowercase())
        .collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_dupes_are_superseded() {
        let store = Store::open_at(std::path::Path::new(":memory:")).unwrap();
        store
            .insert("", "", "the cat sat on the warm mat by the fire", "cli")
            .unwrap();
        store
            .insert("", "", "the cat sat on the warm mat near the fire", "cli")
            .unwrap();
        store
            .insert("", "", "rust borrow checker lifetimes", "cli")
            .unwrap();

        let rep = consolidate(&store).unwrap();
        assert_eq!(rep.superseded, 1);
        assert_eq!(store.count_valid().unwrap(), 2);
    }
}
