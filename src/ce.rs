// ce.rs — optional local cross-encoder (query, document) scorer. Default OFF.
//
// RECALL_CE=off     (default) the model is never loaded.
// RECALL_CE=score   every returned hit gets `scores.ce` (the raw logit); the order is unchanged.
// RECALL_CE=rerank  the first RECALL_CE_DEPTH fused candidates are re-ordered by
//                   1/(60 + fused rank) + RECALL_CE_ALPHA / (60 + CE rank), then scored as above.
//
// Model: any ONNX cross-encoder with a `logits` output, in RECALL_CE_MODEL_DIR (default
// ~/.recall/models/ms-marco-MiniLM-L6-v2): onnx/model.onnx (or RECALL_CE_ONNX), tokenizer.json,
// config.json, special_tokens_map.json, tokenizer_config.json. Loaded once per thread, offline.
// The document text is the stored body, truncated with the query to RECALL_CE_MAXLEN tokens.

use crate::bge::cpu_ep;
use anyhow::{Context, Result};
use fastembed::{RerankInitOptionsUserDefined, TextRerank, TokenizerFiles, UserDefinedRerankingModel};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Off,
    Score,
    Rerank,
}

pub fn mode() -> Mode {
    match std::env::var("RECALL_CE").ok().as_deref().map(str::trim) {
        Some("score") => Mode::Score,
        Some("rerank") => Mode::Rerank,
        _ => Mode::Off,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).filter(|v| *v > 0).unwrap_or(default)
}

pub fn depth() -> usize {
    env_usize("RECALL_CE_DEPTH", 20)
}

pub fn alpha() -> f64 {
    std::env::var("RECALL_CE_ALPHA").ok().and_then(|v| v.trim().parse().ok()).filter(|v: &f64| v.is_finite() && *v >= 0.0).unwrap_or(1.0)
}

fn model_dir() -> Result<PathBuf> {
    if let Ok(d) = std::env::var("RECALL_CE_MODEL_DIR") {
        return Ok(PathBuf::from(d));
    }
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).context("USERPROFILE/HOME not set")?;
    Ok(PathBuf::from(home).join(".recall").join("models").join("ms-marco-MiniLM-L6-v2"))
}

fn load() -> Result<TextRerank> {
    let dir = model_dir()?;
    let read = |name: &str| std::fs::read(dir.join(name)).with_context(|| format!("cross-encoder file missing: {}", dir.join(name).display()));
    let onnx = std::env::var("RECALL_CE_ONNX").unwrap_or_else(|_| "onnx/model.onnx".into());
    let model = UserDefinedRerankingModel::new(
        read(&onnx)?,
        TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        },
    );
    let mut opts = RerankInitOptionsUserDefined::default();
    opts.execution_providers = vec![cpu_ep()];
    opts.max_length = env_usize("RECALL_CE_MAXLEN", 256);
    TextRerank::try_new_from_user_defined(model, opts).context("load cross-encoder")
}

thread_local! {
    static MODEL: RefCell<Option<TextRerank>> = const { RefCell::new(None) };
}

/// Raw logit per doc id, for `ids` whose text is in `texts`. Loads the model on first use.
pub fn score(query: &str, ids: &[i64], texts: &HashMap<i64, String>) -> Result<HashMap<i64, f64>> {
    let ids: Vec<i64> = ids.iter().copied().filter(|id| texts.contains_key(id)).collect();
    if ids.is_empty() {
        return Ok(HashMap::new());
    }
    let docs: Vec<&str> = ids.iter().map(|id| texts[id].as_str()).collect();
    MODEL.with(|m| {
        let mut m = m.borrow_mut();
        if m.is_none() {
            *m = Some(load()?);
        }
        let res = m.as_mut().expect("loaded").rerank(query, docs, false, Some(32))?;
        Ok(res.into_iter().map(|r| (ids[r.index], r.score as f64)).collect())
    })
}

/// Re-order the first `scored.len()` ids of `ids` (the scored head) by the RRF-style blend of their
/// fused rank and their CE rank; the tail keeps its order. Ties: fused order. Pure.
pub fn blend(ids: &[i64], scored: &HashMap<i64, f64>, alpha: f64) -> Vec<i64> {
    let head: Vec<i64> = ids.iter().copied().filter(|id| scored.contains_key(id)).collect();
    let mut by_ce = head.clone();
    by_ce.sort_by(|a, b| scored[b].partial_cmp(&scored[a]).unwrap_or(std::cmp::Ordering::Equal));
    let ce_rank: HashMap<i64, usize> = by_ce.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let mut ranked: Vec<(usize, i64, f64)> = head
        .iter()
        .enumerate()
        .map(|(i, id)| (i, *id, 1.0 / (61.0 + i as f64) + alpha / (61.0 + ce_rank[id] as f64)))
        .collect();
    ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    let mut out: Vec<i64> = ranked.into_iter().map(|(_, id, _)| id).collect();
    out.extend(ids.iter().copied().filter(|id| !scored.contains_key(id)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_moves_ce_winner_up_and_keeps_tail() {
        let ids = vec![10, 11, 12, 13];
        let sc: HashMap<i64, f64> = [(10, -3.0), (11, 1.0), (12, 5.0)].into_iter().collect();
        // alpha 0 = fused order; huge alpha = CE order on the head; unscored 13 stays last
        assert_eq!(blend(&ids, &sc, 0.0), vec![10, 11, 12, 13]);
        assert_eq!(blend(&ids, &sc, 100.0), vec![12, 11, 10, 13]);
        // alpha 1: 10 = 1/61+1/63, 11 = 1/62+1/62, 12 = 1/63+1/61 -> 10 and 12 tie above 11; fused order breaks the tie
        assert_eq!(blend(&ids, &sc, 1.0), vec![10, 12, 11, 13]);
    }

    #[test]
    fn mode_defaults_off() {
        std::env::remove_var("RECALL_CE");
        assert_eq!(mode(), Mode::Off);
    }
}
