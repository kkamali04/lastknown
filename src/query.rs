// query.rs — Reciprocal Rank Fusion (RRF, k=60) across recall's channels.
//
// Each channel returns a ranked list of doc ids. RRF score for a doc is the sum
// over channels of 1/(k + rank). This is rank-based, so channels with wildly
// different score scales (bm25 vs cosine) fuse without normalisation.
//
// Channels currently wired:
//   fts5  — SQLite FTS5 keyword (real)
//   rg    — ripgrep over the vault, path->id mapped (real)
//   bge   — in-process BGE-small dense vectors, cosine top-k (real once embedded)
//   graph — local path-neighborhood graph v0 over indexed vault docs

use crate::bge::BgeSearch;
use crate::store::{Hit, Store};
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

pub const RRF_K: f64 = 60.0;
const RRF_K_ENV: &str = "RECALL_RRF_K";
const FUSION_MODE_ENV: &str = "RECALL_FUSION_MODE";
const DENOISE_MODE_ENV: &str = "RECALL_DENOISE_MODE";
/// Only "off" (or "none") remains: the former path-heuristic modes were removed.
const DEFAULT_DENOISE_MODE: &str = "off";
const MIN_FUSION_DEPTH: usize = 80;
const FUSION_DEPTH_MULTIPLIER: usize = 6;
const MIN_FUSION_DEPTH_ENV: &str = "RECALL_MIN_FUSION_DEPTH";
const FUSION_DEPTH_MULTIPLIER_ENV: &str = "RECALL_FUSION_DEPTH_MULTIPLIER";
const MAX_MIN_FUSION_DEPTH: usize = 200;
const MAX_FUSION_DEPTH_MULTIPLIER: usize = 20;

/// One channel's ranked output: ordered best-first.
pub struct Ranked {
    /// Read by [`source_label`]: every result's `source` names the channel(s) that ranked it.
    pub channel: String,
    pub ids: Vec<i64>,
}

/// Reject a channel name this binary cannot serve, BEFORE any query runs.
///
/// 🔴 This used to be the opposite: unknown or uncompiled channels "contributed nothing", the fts5
/// rescue tier then answered in their place, and `--channels gemini` on a build without the
/// feature returned keyword results with exit 0 and an empty stderr (E2E 7b, 2026-10-29).
pub fn validate_channels(channels: &[String]) -> Result<()> {
    if channels.is_empty() {
        anyhow::bail!("no channels requested (pass --channels, e.g. fts5,bge)");
    }
    for raw in channels {
        let c = raw.trim().to_lowercase();
        let compiled = match c.as_str() {
            "fts5" | "fts" | "sqlite-fts" | "rg" | "ripgrep" | "bge" | "vector" | "lancedb"
            | "qwen3" | "qwen" | "graph" | "surreal" => true,
            "gemini" | "embeddings4" | "vertex" => cfg!(feature = "gemini"),
            "voyage" | "embeddings2" | "dense" => cfg!(feature = "voyage"),
            _ => anyhow::bail!(
                "unknown channel '{raw}' (known: fts5, bge, rg, graph; maintainer channels: qwen3, gemini, voyage)"
            ),
        };
        if !compiled {
            let feature = if matches!(c.as_str(), "gemini" | "embeddings4" | "vertex") {
                "gemini"
            } else {
                "voyage"
            };
            anyhow::bail!(
                "channel '{raw}' is not compiled into this binary (rebuild with --features {feature})"
            );
        }
    }
    Ok(())
}

/// Where one returned hit came from and how each channel scored it. Output only: building it
/// never changes the ranking.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Provenance {
    /// = [`source_label`]: channels whose list holds the doc, channel order, '+'-joined.
    pub source: String,
    /// 0-based position in each channel list that holds the doc (fusion depth, not `top`).
    pub ranks: std::collections::BTreeMap<String, usize>,
    /// RAW per-channel score, never normalised. fts5 / fts5-rescue: -bm25 (title 8, body 1),
    /// higher = better. bge: best-chunk cosine. A channel with no raw score of its own (rg,
    /// graph, qwen3, gemini, voyage) has a rank and NO entry here: absent, never 0.0.
    pub scores: std::collections::BTreeMap<String, f64>,
    /// Sum over `ranks` of 1/(rrf_k + rank + 1): the RRF value, always computed. It is the sort key
    /// under `RECALL_FUSION_MODE=rrf` and when convex falls back to RRF (other channel sets, missing
    /// raw scores); the default (convex, fts5+bge) orders by weighted raw
    /// scores, so for those hits this is informational and the order may differ from it.
    pub rrf: f64,
    /// Subject terms of the query (findable, not [`is_generic_term`]) this hit does NOT hold.
    /// Empty = it holds them all. A question can be answered by a hit that lacks one (questions
    /// paraphrase); the gap is stated here, never hidden (v3 r3, K8). Keyword queries never return
    /// such a hit as their only support: they abstain instead.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lacks: Vec<String>,
    /// Other doc ids for the SAME file (paths equal after case-folding) collapsed into this hit
    /// ([`collapse_case_duplicates`]). A channel that ranked only a copy lends this hit the copy's
    /// rank and raw score, so those numbers may describe one of these ids' text.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub merged: Vec<i64>,
    /// The query had more than [`LABEL_TOKEN_CAP`] label subjects, so only the first were checked
    /// and `lacks` may be incomplete (v3 r8, Codex r7 MEDIUM). Omitted when false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub labels_truncated: bool,
    /// The query FAILED an abstain rule and was answered anyway as a CLOSEST MATCH (lme-improve R2):
    /// this hit is not a confident answer, whatever `lacks` says (it can be empty). Omitted when false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub closest_match: bool,
    /// Main query words (function words dropped) that NO searchable document holds: the judge's exact
    /// zero-df count ([`Judge::unknown`] and the absent words it excused as typos). The same list on every
    /// hit of a query. The CLI's "Not found in the indexed text: ..." line reads it. Omitted when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub absent: Vec<String>,
}

/// [`Provenance`] of `id`, from the channel lists and each channel's raw scores. Pure.
pub(crate) fn provenance(
    id: i64,
    ranked: &[Ranked],
    raw: &HashMap<String, HashMap<i64, f64>>,
    rrf_k: f64,
) -> Provenance {
    let mut p = Provenance { source: source_label(id, ranked), ..Default::default() };
    for r in ranked {
        let Some(pos) = r.ids.iter().position(|x| *x == id) else { continue };
        p.ranks.insert(r.channel.clone(), pos);
        p.rrf += 1.0 / (rrf_k + pos as f64 + 1.0);
        if let Some(s) = raw.get(&r.channel).and_then(|m| m.get(&id)).filter(|s| s.is_finite()) {
            p.scores.insert(r.channel.clone(), *s);
        }
    }
    p
}

/// `source` for one result: the channels whose ranked list contains `id`, in channel order,
/// joined with '+' (e.g. "fts5", "bge", "fts5+bge", "fts5-rescue").
pub fn source_label(id: i64, ranked: &[Ranked]) -> String {
    ranked
        .iter()
        .filter(|r| r.ids.contains(&id))
        .map(|r| r.channel.as_str())
        .collect::<Vec<_>>()
        .join("+")
}

/// Abstain switch. Default ON; `RECALL_ABSTAIN=off` disables it so the evaluation harness can measure
/// the raw score distributions the floors below are calibrated from.
const ABSTAIN_ENV: &str = "RECALL_ABSTAIN";
const ABSTAIN_TRACE_ENV: &str = "RECALL_ABSTAIN_TRACE";
const BGE_MIN_SUPPORT_ENV: &str = "RECALL_BGE_MIN_SUPPORT";
/// bge abstains when none of its top 5 docs holds at least this share of the query's IDF-weighted
/// terms ([`lexical_support`]; df counts live documents only).
///
/// Calibrated 2026-10-30 on a private question set (abstain off) so that about 99% of answerable
/// questions keep answering while a lone emoji, a random hex string and made-up words fall below it.
/// The bge cosine itself cannot gate: the cosines of real questions overlap those of nonsense probes.
///
/// Round 3 (2026-10-30): the rules became a conjunction, so a bge failure now abstains the fused
/// query too. The floor was re-measured against a held-out set of real-word nonsense sentences that
/// no floor was fitted to; 0.19 keeps every answerable question of the calibration set, and lower
/// floors (0.15 / 0.10 / 0.0) let about half of the nonsense through.
/// Other signals measured and rejected: bge top cosine, cosine z-score vs the whole corpus, top-1
/// minus top-10/top-100 mean, the bge cosine of fts5's top 5, IDF-weighted fts5 coverage,
/// matched-term co-occurrence surprise, term proximity.
pub const BGE_MIN_SUPPORT: f64 = 0.19;
/// Most query terms the lexical rule ever demands of fts5's top hit ([`fts5_required_terms`]).
/// Calibrated on a private question set: every answerable question's top hit holds at least 3 of its
/// query terms, so 3 is the largest cap that keeps them all answering; a cap of 4 loses one.
///
/// Round 3 dropped the old `ceil(0.35 * n)` fraction (and `RECALL_FTS5_MIN_COVERAGE`): it asked
/// only 2 matched terms of a 4-5-term query, and half of a held-out set of real-word nonsense
/// sentences got through on 2 incidental words. `min(3, findable)` stops most of them while all 12
/// short positive probes ("kestra passwrod", "AI setpu", "recall daemon", ...) still answer.
pub const FTS5_MATCH_CAP: usize = 3;

fn abstain_enabled() -> bool {
    !matches!(
        std::env::var(ABSTAIN_ENV).ok().as_deref().map(str::trim),
        Some("off") | Some("0") | Some("false") | Some("no")
    )
}

/// What a query that fails the abstain judge gets (2026-11-06): `label` (default) returns the hits
/// anyway, every one `closest_match` ("no confident match"), because an agent reads the output and an empty
/// answer hides what was found; `refuse` (RECALL_ABSTAIN_POLICY=refuse) restores the r9 behaviour, where
/// only queries the calibrated closest-match rule admits come back. A query with no candidates at all is
/// empty either way. The audit (2026-11-06) measured r9's refusals at chance (balanced accuracy ~0.50) on
/// LoCoMo and BEAM, so refusing bought nothing a label does not say.
const ABSTAIN_POLICY_ENV: &str = "RECALL_ABSTAIN_POLICY";

pub(crate) fn abstain_labels() -> bool {
    !matches!(std::env::var(ABSTAIN_POLICY_ENV).ok().as_deref().map(str::trim), Some("refuse"))
}

fn abstain_trace() -> bool {
    matches!(std::env::var(ABSTAIN_TRACE_ENV).ok().as_deref(), Some("1") | Some("on"))
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

/// The rescue tier covers channels that FAILED, never channels that abstained: an abstaining bge
/// returns no ids, and without the `any_abstained` guard the rescue would hand back the very
/// fts5 nonsense hits the abstain exists to suppress.
pub(crate) fn rescue_needed(ranked: &[Ranked], any_abstained: bool, fts5_requested: bool) -> bool {
    !any_abstained && !fts5_requested && ranked.iter().all(|r| r.ids.is_empty())
}

/// Document-frequency cap for [`query_term_weights`]: counting stops here, so a common term costs
/// at most one bounded scan.
const DF_CAP: usize = 5000;

/// [`judged_terms`] with a capped IDF weight each, `ln(1 + DF_CAP / (df + 1))`, and how many of
/// them occur in NO live document.
///
/// A term in 13 docs weighs ~5.9, one in >= 5000 docs ~0.69. A term in NO document weighs 0: no
/// document can hold it, so it says nothing about WHICH document answers (round 3 — at ~8.5 a
/// typo outweighed every real term, so "kestra passwrod" scored 0.37 and "AI setpu" 0.0). The
/// lexical rule still counts such a term as missing, so gibberish cannot pass through it.
/// df goes through `MATCH`, so it is porter-stemmed exactly like the index.
pub(crate) fn query_term_weights(store: &Store, query: &str) -> Result<(Vec<String>, Vec<f64>, usize)> {
    let terms = judged_terms(store, query)?;
    let mut weights = Vec::with_capacity(terms.len());
    let mut zero_df = 0;
    for t in &terms {
        // df counts only documents fts5 could return (live, allowed tier, not path-excluded):
        // excluded text can neither veto a query nor shift a weight (Codex loop2-r1 #6).
        let df = store.term_searchable_df(t, DF_CAP)?;
        zero_df += usize::from(df == 0);
        weights.push(if df == 0 { 0.0 } else { (1.0 + DF_CAP as f64 / (df as f64 + 1.0)).ln() });
    }
    Ok((terms, weights, zero_df))
}

/// A token made only of digits ("2219", "2026") is never evidence (loop2, Codex r3): numbers
/// occur everywhere in the vault, so "plimbex trondulate vasquor 2219" matched a note through
/// "2219" alone. Mixed tokens ("v1", "kmg8", a hex hash) are still judged.
pub(crate) fn is_number_token(t: &str) -> bool {
    !t.is_empty() && t.chars().all(|c| c.is_ascii_digit())
}

/// Shortest absent term that may be excused as a typo of a real word ([`is_plausible_typo`]).
/// Shorter absent tokens are almost always one edit from some vocabulary entry, so the test
/// would excuse everything; at 5+ letters a one-edit neighbour is a real signal.
pub const TYPO_MIN_LEN: usize = 5;
/// Longest absent token tested as a typo (Codex loop2-r1 #7): the one-edit set grows with length
/// (~54 strings per letter), so a several-thousand-character token used to cost hundreds of MB
/// before SQLite ran. Longer absent tokens are simply unknown.
pub const TYPO_MAX_LEN: usize = 24;
/// Shortest all-digit token that constrains the supporting document ([`number_constraints`]).
/// 2 since loop2 r3 (Codex r2 #5: "daemon port 22" and "daemon port 23" judged identically). FTS5
/// never sees a 1-digit token ([`crate::store::fts_query_tokens`] keeps >= 2 bytes). Checked on the
/// live DB: every 2-digit token in the private question set (12 questions: "8:30", "2026-05-19", "tick 10",
/// "0.72.1", ...) and in the builder title/paraphrase positives is held by its answer document.
pub const NUMBER_MIN_LEN: usize = 2;

/// Every string one Damerau edit (delete, transpose, substitute, insert; a-z) from `t`.
pub(crate) fn edit1_candidates(t: &str) -> Vec<String> {
    let c: Vec<char> = t.chars().collect();
    let alpha: Vec<char> = ('a'..='z').collect();
    let mut out = std::collections::BTreeSet::new();
    for i in 0..c.len() {
        let mut d = c.clone();
        d.remove(i);
        out.insert(d.iter().collect::<String>());
        if i + 1 < c.len() {
            let mut s = c.clone();
            s.swap(i, i + 1);
            out.insert(s.iter().collect::<String>());
        }
        for a in &alpha {
            let mut s = c.clone();
            s[i] = *a;
            out.insert(s.iter().collect::<String>());
        }
    }
    for i in 0..=c.len() {
        for a in &alpha {
            let mut s = c.clone();
            s.insert(i, *a);
            out.insert(s.iter().collect::<String>());
        }
    }
    out.remove(t);
    out.into_iter().filter(|s| s.len() >= 2).collect()
}

/// An absent term is a plausible typo when it is alphabetic, >= [`TYPO_MIN_LEN`] letters, and
/// one edit from a term a searchable document holds ("passwrod" -> "password", "setpu" ->
/// "setup"). An absent term that is NOT a typo ("zenthora", a hex hash) means the query is about
/// something the vault does not know, and the query abstains (K8: a wrong confident answer is
/// worse than none).
pub(crate) fn is_plausible_typo(store: &Store, t: &str) -> Result<bool> {
    let n = t.chars().count();
    if !(TYPO_MIN_LEN..=TYPO_MAX_LEN).contains(&n) || !t.chars().all(|c| c.is_ascii_lowercase()) {
        return Ok(false);
    }
    store.any_term_searchable(&edit1_candidates(t))
}

/// The terms the abstain rules actually judge: [`confidence_terms`], with two corrections that
/// need the index.
///
/// 1. When every content word is absent from the corpus but the query also has short tokens
///    ("AI setpu", "Go setpu"), judge all tokens instead. Otherwise only the typo is judged, it
///    can never match, and a query FTS5 answers through "AI" silently abstains. Function words stay
///    out ([`is_label_subject`]): "how do we keep copies of our data" over notes holding none of
///    keep / copies / data used to judge, and label, "how", "do", "of", "our" as missing subjects.
/// 2. One porter stem counts once ([`dedup_stems`]): "cheese cheeses" is one piece of evidence,
///    and padding a nonsense query with an inflection must not buy it a third matched term.
pub(crate) fn judged_terms(store: &Store, query: &str) -> Result<Vec<String>> {
    let mut terms = confidence_terms(query);
    let mut raw = crate::store::fts_query_tokens(query);
    let mut seen = std::collections::HashSet::new();
    raw.retain(|t| !is_number_token(t) && is_label_subject(t) && seen.insert(t.clone()));
    if raw.len() > terms.len() {
        let mut all_absent = true;
        for t in &terms {
            if store.term_searchable(t)? {
                all_absent = false;
                break;
            }
        }
        if all_absent {
            terms = raw;
        }
    }
    dedup_stems(terms)
}

/// Keep the first term of each porter stem, using SQLite's own porter tokenizer (an in-memory
/// FTS5 table plus `fts5vocab`), so "the same stem" means exactly what it means to the index.
pub(crate) fn dedup_stems(terms: Vec<String>) -> Result<Vec<String>> {
    if terms.len() < 2 {
        return Ok(terms);
    }
    let c = rusqlite::Connection::open_in_memory()?;
    c.execute_batch(
        "CREATE VIRTUAL TABLE t USING fts5(x, tokenize='porter unicode61');
         CREATE VIRTUAL TABLE v USING fts5vocab(t, 'instance');",
    )?;
    for (i, t) in terms.iter().enumerate() {
        c.execute("INSERT INTO t(rowid, x) VALUES (?1, ?2)", rusqlite::params![i as i64, t])?;
    }
    let mut stem: HashMap<i64, String> = HashMap::new();
    let mut stmt = c.prepare("SELECT doc, term FROM v ORDER BY doc, offset")?;
    for row in stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
        let (doc, term) = row?;
        stem.entry(doc).or_default().push_str(&format!("{term} "));
    }
    let mut seen = std::collections::HashSet::new();
    Ok(terms
        .into_iter()
        .enumerate()
        .filter(|(i, t)| seen.insert(stem.get(&(*i as i64)).cloned().unwrap_or_else(|| t.clone())))
        .map(|(_, t)| t)
        .collect())
}

/// The terms both abstain rules judge. Drawn from the tokens FTS5 actually searched
/// ([`crate::store::fts_query_tokens`]: first 64 alphanumeric runs of >= 2 bytes), so a token past
/// the FTS cap can neither veto nor support a result, and the work is bounded at 64 terms.
///
/// Normally the content words of those tokens ([`rerank_query_terms`]: stopwords and <= 2-char
/// tokens dropped). When that leaves NOTHING — "AI", "Go", "v1.2" — the raw tokens are judged
/// instead: dropping them made coverage 0.0 and silently abstained on every short query that FTS5
/// answers well. A query with no token at all (a lone emoji) still has nothing to support.
/// Two-character subjects a longer word displaced are still labelled and vetted:
/// [`short_subject_terms`].
pub(crate) fn confidence_terms(query: &str) -> Vec<String> {
    let toks: Vec<String> = crate::store::fts_query_tokens(query)
        .into_iter()
        .filter(|t| !is_number_token(t))
        .collect();
    let content = rerank_query_terms(&toks.join(" "));
    if !content.is_empty() {
        return content;
    }
    let mut seen = std::collections::HashSet::new();
    toks.into_iter().filter(|t| seen.insert(t.clone())).collect()
}

/// Label-only subject tokens of `query` not in `exclude`: EVERY alphanumeric token of any length
/// that [`is_label_subject`] accepts, lowercased, de-duplicated, in query order. Uncapped: the caller
/// filters FIRST and caps after (v3 r8, Codex r7 MEDIUM: capping raw tokens let 64 ignored "a"s
/// use up the budget before "R").
/// v3 r6 (Codex r5 HIGH #1) added two-byte tokens; v3 r7 (Codex r6 HIGH) one-byte tokens ("R
/// plotting", "C smtp", "q z x v j password") and typed-uppercase acronyms ("IT smtp"), which
/// fts_query_tokens (>= 2 bytes) and the lowercase stopword list used to drop.
/// They do NOT count toward the support rules (need, share, rarest term): counting two-byte ones
/// cost two answerable questions ("QA", "PE") their answers. They are LABELLED in
/// `lacks`, and one that no searchable document holds is unknown, so the query abstains.
pub(crate) fn short_subject_terms(query: &str, exclude: &[String]) -> Vec<String> {
    let mut seen: std::collections::HashSet<String> = exclude.iter().cloned().collect();
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .filter(|t| is_label_subject(t))
        .map(str::to_lowercase)
        .filter(|t| seen.insert(t.clone()))
        .collect()
}

/// Most label subjects [`Judge::build`] keeps (after filtering); bounds the per-hit label probes.
pub const LABEL_TOKEN_CAP: usize = 64;

/// One token AS TYPED is a subject for labelling unless it is a function word. The stopword lists
/// ([`SHORT_STOPWORDS`], [`LABEL_STOPWORDS`], [`RERANK_STOPWORDS`], [`GENERIC_TERMS`]) apply only
/// to a token not typed in capitals: "IT", "OR", "US", "R", "C" are always subjects, "it" / "or"
/// never. "I" and "A" are the exception (pronoun and article in any sentence).
pub(crate) fn is_label_subject(tok: &str) -> bool {
    let caps = tok.chars().any(char::is_alphabetic) && !tok.chars().any(char::is_lowercase);
    if caps && tok != "I" && tok != "A" {
        return true;
    }
    let t = tok.to_lowercase();
    let t = t.as_str();
    !(SHORT_STOPWORDS.contains(&t)
        || LABEL_STOPWORDS.contains(&t)
        || RERANK_STOPWORDS.contains(&t)
        || GENERIC_TERMS.contains(&t))
}

/// One-letter function words and contraction pieces ("the owner's" -> "s", "don't" -> "t", "e.g.").
pub(crate) const LABEL_STOPWORDS: &[&str] = &["a", "i", "s", "t", "d", "m", "ll", "re", "ve", "e", "g"];

/// Query terms fts5's top hit must hold, given how many of the judged terms occur in the corpus
/// at all (`findable`): every one of them, up to [`FTS5_MATCH_CAP`]. A term no document holds (a
/// typo, a hash) is not findable, so it can never veto. Pure, so it is unit-testable.
///
/// v3 r1 (Codex r1 HIGH #2): a query of up to [`ALL_TERMS_MAX_FINDABLE`] findable terms needs
/// ALL of them, a query of one more needs all but one. "gmail password rotation runbook" was
/// answered by a Kestra runbook holding 3 of 4 words: the missing word was the subject, and the
/// rarest-term subject rule cannot see that when the subject word is the commonest one.
/// Calibrated on a private tuning set: stricter demands cost answerable queries, so 6+ findable
/// terms keep [`FTS5_MATCH_CAP`] (a known ceiling).
pub(crate) fn fts5_required_terms(findable: usize) -> usize {
    match findable {
        0 => 1,
        f if f <= ALL_TERMS_MAX_FINDABLE => f,
        f if f == ALL_TERMS_MAX_FINDABLE + 1 => f - 1,
        _ => FTS5_MATCH_CAP,
    }
}

/// Largest findable-term count for which a supporting document must hold every findable term.
pub const ALL_TERMS_MAX_FINDABLE: usize = 4;

/// Share of the query's weight mass that one document holds. Pure, so it is unit-testable.
/// An empty term list scores 0.0: a query with no content words (a lone emoji, punctuation) has
/// nothing for a document to support.
pub(crate) fn weighted_share(matched: &[bool], weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    matched.iter().zip(weights).filter(|(m, _)| **m).map(|(_, w)| w).sum::<f64>() / total
}

/// Lexical support: the best [`weighted_share`] of query terms that any of `ids` contains,
/// matched through FTS5 (`MATCH` + `rowid`), so "passwords" in a doc supports "password" in the
/// query exactly as the keyword index would — no second, drifting tokenizer.
#[cfg(test)]
pub(crate) fn lexical_support(
    store: &Store,
    ids: &[i64],
    terms: &[String],
    weights: &[f64],
) -> Result<Option<f64>> {
    let mut best: Option<f64> = None;
    for id in ids {
        let matched = terms
            .iter()
            .map(|t| store.doc_matches_term(*id, t))
            .collect::<Result<Vec<bool>>>()?;
        let s = weighted_share(&matched, weights);
        best = Some(best.map_or(s, |b: f64| b.max(s)));
    }
    Ok(best)
}

/// The lexical rule: fts5's top hit (`top`, None when fts5 found nothing) must hold
/// [`fts5_required_terms`] of the query's [`judged_terms`]. Returns why it failed, or None.
///
/// A zero-df term (a typo, a hash) is NOT a veto: it is not findable, so it is not required.
/// "kestra passwrod" answers. df is probed only for the unmatched terms, and only when the
/// matched count alone does not already pass.
/// How many of fts5's top hits the lexical rule inspects. Codex r3: judging only the first hit
/// let a short one-term note veto the complete answer ranked right below it.
pub const LEXICAL_JUDGE_DEPTH: usize = 5;

/// The pure verdict of the lexical rule, so every branch is unit-testable.
///
/// * `unknown`: judged terms no searchable document holds that are NOT plausible typos -> abstain.
/// * otherwise the best of fts5's top [`LEXICAL_JUDGE_DEPTH`] hits must hold
///   [`fts5_required_terms`]`(findable)` of the findable terms.
pub(crate) fn lexical_verdict(unknown: &[String], best_held: usize, findable: usize) -> Option<String> {
    if !unknown.is_empty() {
        return Some(format!("query term(s) [{}] occur in no searchable document", unknown.join(", ")));
    }
    let need = fts5_required_terms(findable);
    (best_held < need).then(|| {
        format!(
            "no fts5 top-{LEXICAL_JUDGE_DEPTH} hit holds {need} of {findable} findable query terms \
             (best {best_held})"
        )
    })
}

/// All-digit tokens of >= [`NUMBER_MIN_LEN`] digits, in FTS5's token order, deduplicated.
///
/// A number is never EVIDENCE ([`is_number_token`]), but it is a CONSTRAINT (Codex loop2-r1 #1): a
/// document only supports "daemon port 8784" if it holds 8784, so a note about port 2219 cannot
/// answer it through "daemon port". Two-digit tokens count too ([`NUMBER_MIN_LEN`]).
pub(crate) fn number_constraints(query: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    crate::store::fts_query_tokens(query)
        .into_iter()
        .filter(|t| is_number_token(t) && t.len() >= NUMBER_MIN_LEN && seen.insert(t.clone()))
        .collect()
}

/// Everything the abstain rules know about one query, computed once per query.
#[derive(Debug, Clone)]
pub(crate) struct Judge {
    /// [`judged_terms`], in query order.
    pub terms: Vec<String>,
    /// IDF weight per term over searchable documents; 0 for a term no searchable doc holds.
    pub weights: Vec<f64>,
    /// Per term: some searchable document holds it.
    pub findable: Vec<bool>,
    /// Absent terms excused as typos, each with its one-edit candidates. The SUPPORTING document
    /// must hold one of them (Codex loop2-r1 #2: "kestrb password" must not be answered by a
    /// Gmail-password note that holds neither "kestrb" nor "kestra").
    pub typos: Vec<(String, Vec<String>)>,
    /// Absent terms that are not typos: any one abstains the query.
    pub unknown: Vec<String>,
    /// [`number_constraints`]: the supporting document must hold every one.
    pub numbers: Vec<String>,
    /// Per term: findable and not [`is_generic_term`] — a word the query is ABOUT. A returned hit
    /// lacking one is LABELLED ([`Provenance::lacks`]), not refused (v3 r4, the owner's decision).
    pub subject: Vec<bool>,
    /// [`short_subject_terms`]: labelled and vetted, never counted toward support.
    pub short: Vec<String>,
    /// More than [`LABEL_TOKEN_CAP`] label subjects; `short` holds the first ones.
    pub labels_truncated: bool,
}

/// Phrasing words a keyword query's answer may lack: function words the coverage stopword list
/// keeps, counts, and generic "kind of note" / action words ("notes about X", "X steps",
/// "X checklist"). A FIXED list, never derived from document frequency (v3 r3, Codex r2 HIGH #1:
/// a df > 200 rule let 201 unrelated notes make a subject word optional). Calibrated on a private
/// tuning set, where every title answer that lacks a query term lacks only "notes" / "about".
pub const GENERIC_TERMS: &[&str] = &[
    "about", "notes", "note", "steps", "step", "checklist", "guide", "tips", "overview", "summary",
    "because", "instead", "include", "includes", "now", "extra", "many", "more", "most", "much",
    "other", "also", "just", "still", "even", "though", "over", "under", "across", "after",
    "before", "during", "versus", "between", "within", "without", "been", "being", "each",
    "every", "some", "such", "only", "very", "one", "two", "three", "four", "five", "first",
    "second", "third", "time", "times", "day", "days", "way", "thing", "things", "example",
    "examples", "info", "details", "considered", "recorded", "listed", "described", "mentioned",
    "called", "named", "used", "using", "use", "uses", "get", "make", "made", "need", "needs",
    "want", "show", "tell", "find", "like",
];

pub(crate) fn is_generic_term(t: &str) -> bool {
    GENERIC_TERMS.contains(&t)
}

/// A natural-language QUESTION, not a keyword query: a question cue (a wh-word anywhere, "In
/// the X, which ...", or an auxiliary first: "does", "is", ...) plus either a trailing '?' or another
/// function word after the cue. "how gmail password rotation runbook steps" is keyword form;
/// "Which tool is assigned to ..." is a question. Questions paraphrase (on a private question set, most correct
/// answers lack a content word of the question), so they keep the frequency rules and every
/// returned hit is LABELLED with the subject terms it lacks ([`Provenance::lacks`]) instead.
pub(crate) fn is_question_form(query: &str) -> bool {
    const WH: &[&str] = &["what", "which", "why", "how", "when", "where", "who", "whom", "whose"];
    const AUX: &[&str] = &[
        "is", "are", "does", "do", "did", "can", "could", "should", "would", "was", "were", "has",
        "have", "will",
    ];
    const FUNCTION: &[&str] = &[
        "a", "an", "the", "of", "to", "in", "on", "for", "is", "are", "was", "were", "be", "it",
        "by", "as", "at", "or", "and", "with", "that", "this", "does", "do", "did", "has", "have",
        "had", "its", "from", "into", "can", "could", "should", "would", "will", "must", "not",
        "we", "i", "my", "our", "their", "they", "you", "your", "there", "than", "then", "any",
        "all", "but",
    ];
    let lower = query.to_lowercase();
    let toks: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()).collect();
    let cue = toks
        .iter()
        .position(|t| WH.contains(t))
        .or_else(|| toks.first().filter(|t| AUX.contains(t)).map(|_| 0));
    match cue {
        Some(i) => query.trim_end().ends_with('?') || toks[i + 1..].iter().any(|t| FUNCTION.contains(t)),
        None => false,
    }
}

/// What one candidate document holds of a [`Judge`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Evidence {
    /// Per judged term: the doc holds it (always false for a term that is not findable).
    pub held: Vec<bool>,
    /// The doc holds every number constraint and a correction for every typo.
    pub constraints_ok: bool,
}

impl Evidence {
    pub fn held_count(&self) -> usize {
        self.held.iter().filter(|h| **h).count()
    }
}

impl Judge {
    pub fn build(store: &Store, query: &str) -> Result<Judge> {
        let (terms, weights, _) = query_term_weights(store, query)?;
        let findable: Vec<bool> = weights.iter().map(|w| *w > 0.0).collect();
        let (mut typos, mut unknown) = (Vec::new(), Vec::new());
        for (t, f) in terms.iter().zip(&findable) {
            if *f {
                continue;
            }
            if is_plausible_typo(store, t)? {
                typos.push((t.clone(), edit1_candidates(t)));
            } else {
                unknown.push(t.clone());
            }
        }
        let subject = terms.iter().zip(&findable).map(|(t, f)| *f && !is_generic_term(t)).collect();
        // Excluded: judged terms `lacks` already covers (subjects) or the judge already vetted
        // (not findable: typos, unknowns). A judged GENERIC word typed in capitals ("MAKE", "TIME")
        // is still a label subject (v3 r8, Codex r7 HIGH): `lacks` skips generic terms.
        let covered: Vec<String> = terms
            .iter()
            .zip(&findable)
            .filter(|(t, f)| !**f || !is_generic_term(t))
            .map(|(t, _)| t.clone())
            .collect();
        let mut short = short_subject_terms(query, &covered);
        let labels_truncated = short.len() > LABEL_TOKEN_CAP;
        short.truncate(LABEL_TOKEN_CAP);
        for t in &short {
            if !store.term_searchable(t)? {
                unknown.push(t.clone());
            }
        }
        Ok(Judge {
            terms,
            weights,
            findable,
            typos,
            unknown,
            numbers: number_constraints(query),
            subject,
            short,
            labels_truncated,
        })
    }

    /// Document `id` holds every number constraint that SOME searchable document holds (the
    /// closest-match filter; typos excluded). A number no document holds cannot be satisfied by any
    /// hit, so it is labelled (it is in `unknown`, hence in every closest hit's `lacks`) instead of
    /// emptying the list: LoCoMo dates ("2023", "July 13") live in session headers, never in turns.
    pub fn holds_numbers(&self, store: &Store, id: i64) -> Result<bool> {
        for n in &self.numbers {
            if self.unknown.contains(n) {
                continue;
            }
            if !store.doc_matches_term(id, n)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Sum of the IDF weights of the judged terms `e` holds, in nats ([`CLOSEST_MIN_MASS`]).
    pub fn held_mass(&self, e: &Evidence) -> f64 {
        self.weights.iter().zip(&e.held).filter(|(_, h)| **h).map(|(w, _)| w).sum()
    }

    /// May a query that FAILED an abstain rule be answered as an explicit closest match instead of
    /// silence (lme-improve R2)? Only a natural-language question ([`is_question_form`]: keyword
    /// queries keep the old behaviour exactly), whose unknown terms are at most half of its judged
    /// terms, and whose best judged fts5 candidate holds at least [`CLOSEST_MIN_MASS`] nats.
    pub fn may_answer_closest(&self, query: &str, fts: Option<&[Evidence]>) -> bool {
        let Some(fts) = fts.filter(|f| !f.is_empty()) else { return false };
        is_question_form(query)
            && 2 * self.unknown.len() <= self.findable_count() + self.unknown.len()
            && fts.iter().map(|e| self.held_mass(e)).fold(0.0, f64::max) >= CLOSEST_MIN_MASS
    }

    /// [`Judge::may_answer_closest`], OR the corpus-calibrated path (any query form): unknown terms
    /// at most half of the judged ones and [`closest_score`] above the corpus's phantom threshold
    /// ([`phantom_threshold`]). Returns the decision and, for the trace, (score, threshold).
    pub fn may_answer_closest_calibrated(
        &self,
        store: &Store,
        query: &str,
        fts: Option<&[Evidence]>,
    ) -> Result<(bool, Option<(f64, f64)>)> {
        if self.may_answer_closest(query, fts) {
            return Ok((true, None));
        }
        let Some(fts) = fts.filter(|f| !f.is_empty()) else { return Ok((false, None)) };
        if 2 * self.unknown.len() > self.findable_count() + self.unknown.len() {
            return Ok((false, None));
        }
        // Never calibrated for this reader yet (the daemon computes it in the background): the
        // calibrated branch says no, and only the strict rule above can label. Only a view's very
        // first queries see this.
        let Some(null) = phantom_null_or_defer(store)? else { return Ok((false, None)) };
        let Some(thr) = phantom_threshold(&null, closest_phantom_q()) else {
            // A store too small to calibrate (< PHANTOM_MIN_DOCS: a fresh install's first notes)
            // labels instead of refusing, unless a term occurs in no document at all: a new
            // user's first plain query must not come back empty (packaging clean-room, 2 notes).
            let held = fts.iter().any(|e| e.held_count() > 0);
            return Ok((self.unknown.is_empty() && held, None));
        };
        let s = closest_score(self, fts);
        Ok((s > thr, Some((s, thr))))
    }

    /// [`Judge::lacks`] plus the [`Judge::short`] subjects document `id` does not hold.
    pub fn lacks_of(&self, store: &Store, id: i64, e: &Evidence) -> Result<Vec<String>> {
        let mut out = self.lacks(e);
        for t in &self.short {
            if !store.doc_matches_term(id, t)? {
                out.push(t.clone());
            }
        }
        Ok(out)
    }

    /// The subject terms (findable, not generic) `e` does NOT hold, in query order. A function word is
    /// never named, even when [`confidence_terms`] falls back to judging one ("how do"): cli-quality.
    pub fn lacks(&self, e: &Evidence) -> Vec<String> {
        self.terms
            .iter()
            .zip(&self.subject)
            .zip(&e.held)
            .filter(|((t, s), h)| **s && !**h && is_label_subject(t))
            .map(|((t, _), _)| t.clone())
            .collect()
    }

    /// The query carries a hard constraint every returned hit must satisfy (a number, a typo).
    pub fn has_constraints(&self) -> bool {
        !self.numbers.is_empty() || !self.typos.is_empty()
    }

    /// Main query words no searchable document holds (df 0): the unknown terms and the typo-excused ones,
    /// in query order, function words dropped. Display only (cli-quality addendum B).
    pub fn absent_words(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for t in self.terms.iter().zip(&self.findable).filter(|(_, f)| !**f).map(|(t, _)| t).chain(&self.unknown) {
            if is_label_subject(t) && !out.contains(t) {
                out.push(t.clone());
            }
        }
        out
    }

    pub fn findable_count(&self) -> usize {
        self.findable.iter().filter(|f| **f).count()
    }

    /// Terms a supporting document must hold: [`fts5_required_terms`] of the findable ones.
    pub fn need(&self) -> usize {
        fts5_required_terms(self.findable_count())
    }

    pub fn evidence(&self, store: &Store, id: i64) -> Result<Evidence> {
        let mut held = Vec::with_capacity(self.terms.len());
        for (t, f) in self.terms.iter().zip(&self.findable) {
            held.push(*f && store.doc_matches_term(id, t)?);
        }
        let mut constraints_ok = true;
        for n in &self.numbers {
            if !constraints_ok {
                break;
            }
            constraints_ok = store.doc_matches_term(id, n)?;
        }
        for (_, fixes) in &self.typos {
            if !constraints_ok {
                break;
            }
            constraints_ok = store.doc_matches_any(id, fixes)?;
        }
        Ok(Evidence { held, constraints_ok })
    }

    /// [`Judge::evidence`] for the first [`LEXICAL_JUDGE_DEPTH`] of `ids`.
    pub fn evidences(&self, store: &Store, ids: &[i64]) -> Result<Vec<Evidence>> {
        prof::time("judge_evidences", || {
            ids.iter().take(LEXICAL_JUDGE_DEPTH).map(|id| self.evidence(store, *id)).collect()
        })
    }

    /// A candidate SUPPORTS the query when it satisfies every constraint, holds [`Judge::need`]
    /// of the findable terms and, for a short query, its subject ([`Judge::holds_subject`]). Both
    /// rules ask this of a candidate itself, so a partial-overlap decoy (a password-only note for
    /// "kestra password") never supports.
    ///
    /// v3 r4: a supporting hit need NOT hold every subject term. r3 demanded that of keyword-form
    /// queries, and long real keyword queries abstained for the sake of one more refused
    /// nonsense sentence. The gap is stated on every returned hit ([`Provenance::lacks`]) and on
    /// every output surface instead.
    pub fn supports(&self, e: &Evidence) -> bool {
        e.constraints_ok
            && e.held_count() >= self.need()
            && self.holds_subject(e)
            && self.misses_only_common(e)
    }

    /// For a query of at most [`RARE_MISS_MAX_FINDABLE`] findable terms, every findable term the
    /// candidate LACKS must be common (df > [`RARE_TERM_DF`]). v3 r2 (Codex r1b HIGH #2): a Kestra
    /// password-rotation runbook answered "gmail password rotation runbook steps" (4 of 5) and
    /// "... emergency recovery" (3 of 6) because the rarest-term rule cannot see a subject word
    /// that is not the rarest. A rare word the answer lacks is what the query is ABOUT; a common
    /// one ("steps", "notes") is phrasing.
    pub fn misses_only_common(&self, e: &Evidence) -> bool {
        self.findable_count() > RARE_MISS_MAX_FINDABLE
            || self.weights.iter().zip(&e.held).all(|(w, h)| *h || *w < rare_term_min_weight())
    }

    /// For a query of at most [`SUBJECT_MAX_FINDABLE`] findable terms, the candidate holds one of
    /// its heaviest (rarest) findable terms. v3 (Codex r2 HIGH #3): three of four findable words
    /// let a Kestra password-rotation runbook answer "gmail password rotation runbook"; the rarest
    /// word of a short query is what names its subject. Longer queries are exempt: on a private question set the
    /// rarest word of a long question is often a paraphrase word the answer never uses.
    pub fn holds_subject(&self, e: &Evidence) -> bool {
        if self.findable_count() > SUBJECT_MAX_FINDABLE {
            return true;
        }
        let top = self.weights.iter().copied().fold(0.0, f64::max);
        top <= 0.0
            || self.weights.iter().zip(&e.held).any(|(w, h)| *h && *w == top)
    }

    /// One trace line of raw evidence, for offline calibration (`RECALL_ABSTAIN_TRACE=1`).
    pub fn trace_json(&self, query: &str, fts: &[Evidence], bge: &[Evidence]) -> String {
        let ev = |v: &[Evidence]| -> Vec<serde_json::Value> {
            v.iter()
                .map(|e| {
                    serde_json::json!({
                        "held": e.held.iter().map(|h| u8::from(*h)).collect::<Vec<_>>(),
                        "ok": e.constraints_ok,
                    })
                })
                .collect()
        };
        serde_json::json!({
            "q": query,
            "terms": self.terms,
            "w": self.weights,
            "typos": self.typos.iter().map(|t| &t.0).collect::<Vec<_>>(),
            "unknown": self.unknown,
            "numbers": self.numbers,
            "fts": ev(fts),
            "bge": ev(bge),
        })
        .to_string()
    }
}

/// Largest findable-term count for which a supporting candidate must hold the query's subject
/// ([`Judge::holds_subject`]). Calibrated on a private tuning set, never on the held-out set.
pub const SUBJECT_MAX_FINDABLE: usize = 5;

/// Largest findable-term count for which [`Judge::misses_only_common`] applies.
/// "A missing term needs df >= D" for findable <= K, swept on a private tuning set (held-out set
/// never read): K 6, D 200 kept answerable recall unchanged and refused more nonsense; larger D or
/// K cost answers. So 7+ findable terms keep the older rules (a known ceiling).
pub const RARE_MISS_MAX_FINDABLE: usize = 6;
/// A term in at most this many searchable documents is "rare": a short query's answer must hold it.
pub const RARE_TERM_DF: usize = 200;

/// The [`query_term_weights`] weight of a term with df == [`RARE_TERM_DF`]: weights fall as df
/// rises, so `w >= this` means `df <= RARE_TERM_DF`.
pub(crate) fn rare_term_min_weight() -> f64 {
    (1.0 + DF_CAP as f64 / (RARE_TERM_DF as f64 + 1.0)).ln()
}

/// Least IDF mass (sum of [`query_term_weights`], nats) the best judged fts5 candidate of a failed
/// question must hold to be answered as a closest match ([`Judge::may_answer_closest`]). A
/// heuristic tied to [`DF_CAP`] (weights are `ln(1 + DF_CAP / (df + 1))`, not corpus-relative):
/// e.g. two words each in one of ~50 sessions give 15.6, two words in every session 9.2.
/// Picked on LongMemEval_S TUNE + vault `neg_long` from {4,8,10,12,14,16}: 12 keeps 95.3% of
/// abstain-off R_all@5 and lets 5/50 long nonsense sentences through as closest matches
/// (`neg_long_ho` 2/40; that set was also used in earlier calibrations, so it is not independent).
pub const CLOSEST_MIN_MASS: f64 = 12.0;

// ---- Phantom-calibrated closest match (feat/abstain-calibrated) ------------------------------
//
// CLOSEST_MIN_MASS is an absolute number of nats, tuned on ~50 long LongMemEval sessions. In a
// store of one-line chat turns (LoCoMo) the best turn holds ~2 query terms, ~10 nats, so 26% of
// ANSWERABLE questions came back empty — and so did 26% of the unanswerable ones: no signal.
// Instead each corpus calibrates itself: PHANTOM probes (real words of the corpus, each from a
// different document, no single document holding them all, so nothing can answer them) are judged
// exactly like a query, and a query that failed a rule is still a closest match when its
// evidence beats [`CLOSEST_PHANTOM_QUANTILE`] of the corpus's phantoms. Keyword queries too.

/// Phantom probes generated per corpus.
pub const PHANTOM_PROBES: usize = 200;
/// Fewest generated probes for a usable calibration; below it the calibrated path is off.
pub const PHANTOM_MIN_PROBES: usize = 40;
/// Fewest searchable documents a corpus needs before it calibrates itself: a handful of notes says
/// nothing about what "no document answers this" looks like, so tiny stores keep the r8 rules.
pub const PHANTOM_MIN_DOCS: usize = 20;
/// A failed query is a closest match when its [`closest_score`] is above this quantile of the
/// corpus's phantom scores. Tuned on DEV halves only (crc32(store+id) even):
///   q     LoCoMo dialog keptR@5 / phantoms refused   BEIR SciFact+NFCorpus keptR@10 / cross-corpus refused
///   r8    0.889 / 0.816                              0.659 / 0.810
///   0.0   0.992 / 0.004                              0.984 / 0.079
///   0.1   0.968 / 0.097                              0.953 / 0.472
///   0.2   0.959 / 0.194                              0.939 / 0.538
///   0.5   0.925 / 0.423                              0.853 / 0.679
/// 0.1 is the largest q that keeps >= 95% of abstain-off recall on both. The lexical score cannot
/// keep both answers and real-word nonsense refusals (answerable-vs-phantom AUC ~0.84): known-word
/// nonsense now mostly comes back as a LABELLED closest match, never as a confident answer.
pub const CLOSEST_PHANTOM_QUANTILE: f64 = 0.1;
const CLOSEST_PHANTOM_Q_ENV: &str = "RECALL_CLOSEST_PHANTOM_Q";
/// Bumped whenever the probe generator or [`closest_score`] changes, so stored nulls are redone.
const PHANTOM_VERSION: u32 = 5;
const PHANTOM_STOP: &[&str] = &[
    "the", "and", "for", "that", "this", "with", "have", "from", "they", "will", "would", "there",
    "their", "what", "about", "which", "when", "your", "said", "each", "them", "then", "than",
    "been", "were", "more", "some", "into", "could", "time", "very", "just", "like", "only",
    "also", "over", "such", "even", "most", "made", "after", "many", "those", "being", "where",
    "while", "these", "other", "because", "should", "through", "before", "still", "might",
    "shall", "here", "does", "doing", "done", "much", "well", "back", "again", "never", "always",
    "every", "really", "thing", "things", "know", "think", "going", "want", "yeah", "okay", "sure",
];

/// The evidence a closest match is calibrated on: the best judged fts5 candidate's held IDF mass
/// over the query's whole judged mass, where every UNKNOWN term (in no searchable document, not a
/// typo) counts the weight of a df-0 term, `ln(1 + DF_CAP)`. A share, so it does not depend on
/// document length the way [`Judge::held_mass`] does; the unknown penalty keeps "zebra quantum
/// soup protein" (two words the corpus never saw) from scoring 1.0 on the two it did (DEV BEIR
/// with cross-corpus negatives: refusal 27% -> 47% at the same answered rate).
pub(crate) fn closest_score(judge: &Judge, fts: &[Evidence]) -> f64 {
    let total: f64 =
        judge.weights.iter().sum::<f64>() + judge.unknown.len() as f64 * (1.0 + DF_CAP as f64).ln();
    if total <= 0.0 {
        return 0.0;
    }
    fts.iter().map(|e| judge.held_mass(e)).fold(0.0, f64::max) / total
}

/// splitmix64: a fixed, dependency-free generator, so a corpus always gets the same probes.
struct Mix(u64);
impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Content words a probe may draw from a document: alphabetic runs of >= 4 letters, lowercased,
/// minus [`PHANTOM_STOP`].
fn phantom_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphabetic())
        .filter(|t| t.len() >= 4)
        .map(str::to_ascii_lowercase)
        .filter(|t| !PHANTOM_STOP.contains(&t.as_str()))
        .collect()
}

/// Up to [`PHANTOM_PROBES`] impossible queries from the corpus's own words: 3..=6 words, each from
/// a different random searchable document, re-drawn while any one searchable document holds them
/// all. Half keyword form ("w1 w2 w3"), half question form ("What did w1 w2 w3?").
pub(crate) fn phantom_queries(store: &Store, ids: &[i64], seed: u64) -> Result<Vec<String>> {
    let mut out = Vec::new();
    if ids.len() < PHANTOM_MIN_DOCS {
        return Ok(out);
    }
    let mut rng = Mix(seed);
    let mut words: HashMap<i64, Vec<String>> = HashMap::new();
    let mut tries = 0;
    while out.len() < PHANTOM_PROBES && tries < 20 * PHANTOM_PROBES {
        tries += 1;
        let k = 3 + rng.below(4);
        let mut picked: Vec<String> = Vec::new();
        // k DIFFERENT documents per probe (Codex abstain-cal r1 #5)
        let mut docs: Vec<i64> = Vec::with_capacity(k);
        while docs.len() < k {
            let id = ids[rng.below(ids.len())];
            if !docs.contains(&id) {
                docs.push(id);
            }
        }
        for id in docs {
            if !words.contains_key(&id) {
                let body = store.bodies_for_ids(&[id])?.remove(&id).unwrap_or_default();
                words.insert(id, phantom_words(&body));
            }
            let w = &words[&id];
            if !w.is_empty() {
                let t = w[rng.below(w.len())].clone();
                if !picked.contains(&t) {
                    picked.push(t);
                }
            }
        }
        if picked.len() < 3 || store.some_doc_holds_all(&picked)? {
            continue;
        }
        let q = picked.join(" ");
        out.push(if rng.next() & 1 == 0 { q } else { format!("What did {q}?") });
    }
    Ok(out)
}

/// The corpus's phantom null: [`closest_score`] of every [`phantom_queries`] probe, sorted, judged
/// exactly as a query is (Judge::build + fts5's judged top 5), all inside one read snapshot. Stored
/// in the store's own `abstain_calibration` table as `{"n": docs, "null": [..]}` under
/// [`Store::searchable_snapshot`]'s fingerprint. No process-wide cache (Codex abstain-cal r1 #1:
/// two stores with the same ids shared one null); writing is best effort, so a read-only or locked
/// store just recomputes next time.
///
/// ponytail: a fresh null costs one judged query per probe (~60 s cold for 57k FiQA docs on a
/// loaded VPS core; warm queries 0.3 s). So when the exact fingerprint is not stored, the newest
/// stored null is reused if its corpus state is at least [`PHANTOM_REUSE_MIN_SIMILARITY`] similar
/// to the current one (MinHash estimate over every searchable (id, content version) item; Codex
/// abstain-cal r2: a document COUNT bound let a full rewrite or a disjoint tier policy of the
/// same size reuse an unrelated null). A few added or edited notes do not move a 200-probe
/// quantile, and an hourly incremental index must not pay a minute each time. Upgrade path: a
/// real corpus generation counter plus background recalibration.
pub(crate) fn phantom_null(store: &Store) -> Result<Vec<f64>> {
    let (key, null) = store.in_snapshot(|| {
        let PhantomMiss { key, ids, sketch, .. } = match phantom_stored(store)? {
            Ok(null) => return Ok((None, null)),
            Err(miss) => miss,
        };
        #[cfg(test)]
        tests::note_calibration_thread();
        // Deterministic per CORPUS, not per index run: documents in (path, id) order and a seed hashed from
        // every (path, content version), so two fresh indexes of one vault (rg enumerates in a different order
        // each run, so doc ids differ) draw the same probes (integrate-rc1 2026-11-06: 1 of 300 SciFact
        // queries flipped between [] and hits across two such indexes).
        let docs = store.searchable_by_path()?;
        let mut seed: u64 = 0x5EED_AB57;
        for (_, path, ver) in &docs {
            for b in path.bytes().chain([0xfe]).chain(ver.bytes()).chain([0xff]) {
                seed = (seed ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        let ordered: Vec<i64> = docs.iter().map(|d| d.0).collect();
        debug_assert_eq!(ordered.len(), ids.len());
        let mut v = Vec::new();
        for q in phantom_queries(store, &ordered, seed)? {
            let j = Judge::build(store, &q)?;
            let ev = j.evidences(store, &store.fts_judge_ids(&q, LEXICAL_JUDGE_DEPTH)?)?;
            v.push(closest_score(&j, &ev));
        }
        v.sort_by(|a, b| a.total_cmp(b));
        Ok((Some((key, ids.len(), sketch)), v))
    })?;
    if let Some((key, n, sketch)) = key {
        // Still best effort, but loud: a put that silently failed made every query recompute.
        let row = serde_json::to_string(&StoredNull { n, sketch, null: null.clone() })?;
        if let Err(e) = store.calibration_put(&key, &row) {
            eprintln!("recall: could not store the abstain calibration (it will be recomputed): {e:#}");
        }
    }
    Ok(null)
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StoredNull {
    n: usize,
    sketch: Vec<u64>,
    null: Vec<f64>,
}

/// What [`phantom_stored`] found instead of a reusable null.
struct PhantomMiss {
    key: String,
    ids: Vec<i64>,
    sketch: Vec<u64>,
    /// This reader's newest stored null, too far from the current corpus to reuse for good but
    /// the best answer the daemon has while the calibrator recomputes.
    stale: Option<Vec<f64>>,
}

/// The stored null [`phantom_null`] would reuse for the current reader, or on a miss what computing
/// one needs.
fn phantom_stored(store: &Store) -> Result<std::result::Result<Vec<f64>, PhantomMiss>> {
    let (fp, ids, sketch) = store.searchable_snapshot()?;
    let prefix = phantom_prefix(&crate::scope::current());
    let key = format!("{prefix}{fp}");
    let parse = |s: String| serde_json::from_str::<StoredNull>(&s).ok();
    if let Some(v) = store.calibration_get(&key)?.and_then(parse) {
        return Ok(Ok(v.null));
    }
    let near = |v: &StoredNull| {
        crate::store::sketch_similarity(&v.sketch, &sketch) >= PHANTOM_REUSE_MIN_SIMILARITY
    };
    let latest = store
        .calibration_latest(&prefix)?
        .and_then(parse)
        .filter(|v| v.null.len() >= PHANTOM_MIN_PROBES);
    if let Some(v) = latest.as_ref().filter(|v| near(v)) {
        return Ok(Ok(v.null.clone()));
    }
    // A stale null stands in only for an evolved version of the same corpus, never a rewritten
    // one or another tier / exclusion policy under the same prefix (Codex calib r2).
    let stale = latest
        .filter(|v| crate::store::sketch_similarity(&v.sketch, &sketch) >= PHANTOM_STALE_MIN_SIMILARITY)
        .map(|v| v.null);
    Ok(Err(PhantomMiss { key, ids, sketch, stale }))
}

/// Least similarity for the daemon to serve a stale null while it recalibrates
/// ([`phantom_null_or_defer`]); below it a view is judged strict-only until the new null lands.
pub const PHANTOM_STALE_MIN_SIMILARITY: f64 = 0.5;

/// The calibration key prefix of `reader`'s view. Operator keys keep their pre-tenancy form
/// (existing caches stay valid). A scoped reader's keys live under a different prefix that the
/// operator's `v{N}:` scan can never match, and carry the reader's policy, so no two views ever
/// reuse each other's null (Codex tenancy b1).
pub(crate) fn phantom_prefix(reader: &crate::scope::Reader) -> String {
    match reader {
        crate::scope::Reader::Operator => format!("v{PHANTOM_VERSION}:"),
        crate::scope::Reader::Agent { allowed, .. } => format!("s{PHANTOM_VERSION}:{}:", allowed.join(",")),
    }
}

/// How long a view whose calibration failed waits before it is queued again.
pub(crate) const CALIBRATION_RETRY: std::time::Duration = std::time::Duration::from_secs(600);

/// One view's place in the calibrator's queue.
enum Slot {
    /// Queued or being computed: never queued again until the calibrator reports back, however
    /// long that takes (Codex calib r2: a time-based expiry let an 8.5-minute job get a twin).
    Pending,
    /// The last attempt did not store a null; retry after [`CALIBRATION_RETRY`].
    Failed(std::time::Instant),
}

/// Per-view scheduling state, shared by the request thread and the calibrator. It holds no
/// sender, so the calibrator owning a copy cannot keep its own input channel open.
#[derive(Clone, Default)]
pub(crate) struct CalibrationStatus(std::sync::Arc<std::sync::Mutex<HashMap<String, Slot>>>);

impl CalibrationStatus {
    /// The calibrator is done with `reader`'s view: forget it when its null is now stored, else
    /// start its retry clock.
    pub(crate) fn finished(&self, reader: &crate::scope::Reader, stored: bool) {
        let mut slots = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if stored {
            slots.remove(&phantom_prefix(reader));
        } else {
            slots.insert(phantom_prefix(reader), Slot::Failed(std::time::Instant::now()));
        }
    }
}

/// The request thread's handle on the background calibrator: readers queued at most once per
/// view (Codex calib r1 #3: during an 8.5-minute calibration every repeat query queued another
/// 200-probe job). Dropping the last handle disconnects the calibrator, which then exits.
#[derive(Clone)]
pub(crate) struct CalibrationQueue {
    tx: std::sync::mpsc::Sender<crate::scope::Reader>,
    status: CalibrationStatus,
    warned: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl CalibrationQueue {
    pub(crate) fn new() -> (Self, std::sync::mpsc::Receiver<crate::scope::Reader>) {
        let (tx, rx) = std::sync::mpsc::channel();
        (Self { tx, status: CalibrationStatus::default(), warned: Default::default() }, rx)
    }

    /// The scheduling state for the calibrator thread (no sender inside).
    pub(crate) fn status(&self) -> CalibrationStatus {
        self.status.clone()
    }

    /// Queue `reader` unless its view is pending or in its retry wait. Never blocks: the channel
    /// is unbounded (coalescing bounds it at one entry per view) and a state lock held by the
    /// calibrator at this instant just skips the submit; the view's next miss tries again.
    pub(crate) fn submit(&self, reader: crate::scope::Reader) {
        let prefix = phantom_prefix(&reader);
        let Ok(mut slots) = self.status.0.try_lock() else { return };
        match slots.get(&prefix) {
            Some(Slot::Pending) => return,
            Some(Slot::Failed(at)) if at.elapsed() < CALIBRATION_RETRY => return,
            _ => {}
        }
        if self.tx.send(reader).is_err() {
            if !self.warned.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!("recall daemon: the abstain calibrator is not running; uncalibrated views stay strict");
            }
            return;
        }
        slots.insert(prefix, Slot::Pending);
    }

    /// Test-only: move every timestamp `by` into the past, as if that much time had gone by.
    #[cfg(test)]
    fn age(&self, by: std::time::Duration) {
        for slot in self.status.0.lock().unwrap().values_mut() {
            if let Slot::Failed(at) = slot {
                *at -= by;
            }
        }
    }
}

thread_local! {
    /// Set on the daemon's request thread ([`defer_calibration_to`]): a missing null is queued for
    /// the background calibrator instead of computed inline.
    static CALIBRATOR: std::cell::RefCell<Option<CalibrationQueue>> = const { std::cell::RefCell::new(None) };
}

/// Clears [`CALIBRATOR`] on this thread when dropped.
pub(crate) struct DeferGuard(());

impl Drop for DeferGuard {
    fn drop(&mut self) {
        CALIBRATOR.with(|c| *c.borrow_mut() = None);
    }
}

/// While the guard lives, on THIS thread, a query that needs an uncomputed phantom null queues
/// the reader on `q` and is judged without a fresh one. The daemon's single request thread must
/// never spend the ~200 judged probes (8.5 min of CPU on the live 70k-note vault) of a fresh
/// calibration: every other client waits behind it (live hang 2026-11-06).
pub(crate) fn defer_calibration_to(q: CalibrationQueue) -> DeferGuard {
    CALIBRATOR.with(|c| *c.borrow_mut() = Some(q));
    DeferGuard(())
}

/// Whether `reader`'s view has a reusable stored null now (the calibrator's success check).
pub(crate) fn phantom_is_stored(store: &Store) -> Result<bool> {
    Ok(phantom_stored(store)?.is_ok())
}

/// [`phantom_null`], except that on a thread with a calibrator ([`defer_calibration_to`]) a miss
/// queues the calibration and answers with this reader's last stored null (stale, but the same
/// reader's), or `None` when it never had one.
pub(crate) fn phantom_null_or_defer(store: &Store) -> Result<Option<Vec<f64>>> {
    let Some(q) = CALIBRATOR.with(|c| c.borrow().clone()) else {
        return phantom_null(store).map(Some);
    };
    Ok(match phantom_stored(store)? {
        Ok(null) => Some(null),
        Err(miss) => {
            q.submit(crate::scope::current());
            miss.stale
        }
    })
}

/// Least estimated Jaccard similarity (current vs stored corpus state) for reusing a stored null.
pub const PHANTOM_REUSE_MIN_SIMILARITY: f64 = 0.95;

/// The score a failed query must beat to be a closest match: the
/// [`CLOSEST_PHANTOM_QUANTILE`] (or `RECALL_CLOSEST_PHANTOM_Q`) of the sorted phantom null, or None
/// when the corpus could not make [`PHANTOM_MIN_PROBES`] probes (the calibrated path is off).
pub(crate) fn phantom_threshold(null: &[f64], q: f64) -> Option<f64> {
    (null.len() >= PHANTOM_MIN_PROBES)
        .then(|| null[((q.clamp(0.0, 1.0) * null.len() as f64) as usize).min(null.len() - 1)])
}

pub(crate) fn closest_phantom_q() -> f64 {
    env_f64(CLOSEST_PHANTOM_Q_ENV, CLOSEST_PHANTOM_QUANTILE)
}

/// Findable-term count from which the lexical rule also asks for [`LONG_QUERY_MIN_SHARE`].
pub const LONG_QUERY_FINDABLE: usize = 8;
/// Share of a LONG query's IDF weight ([`weighted_share`]) the supporting fts5 hit must hold.
///
/// loop2 r3 (G2a, Codex r2 HIGH #3): three matched words out of eight to ten let a long sentence
/// of ordinary words ("when the volcano borrows slowly the rusty carpet argues beside a golden
/// pigeon") match any long note or book through incidental words — 32 of 50 builder long-nonsense
/// sentences answered on fts5. Calibrated on private question sets only (never the sealed
/// set), from abstain-json evidence with abstain off, best
/// supporting share per query by findable-term count:
///   findable 8 / 9 / 10: answerable positives min 0.56 / 0.56 / 0.54; long nonsense max 0.47 / 0.49 / 0.39
///   findable 7: positives min 0.21 (a paraphrase) vs nonsense max 0.60 -> NOT separable, exempt
/// 0.52 sits between the two bands at 8+ terms, and costs no answerable, title or paraphrase answer.
pub const LONG_QUERY_MIN_SHARE: f64 = 0.52;

/// [`Judge::supports`], plus the long-query share floor. The lexical rule asks this of fts5's hits;
/// the bge rule keeps plain `supports` and its own [`BGE_MIN_SUPPORT`] floor.
pub(crate) fn lexically_supports(judge: &Judge, e: &Evidence) -> bool {
    judge.supports(e)
        && (judge.findable_count() < LONG_QUERY_FINDABLE
            || weighted_share(&e.held, &judge.weights) >= LONG_QUERY_MIN_SHARE)
}

/// The lexical rule over fts5's top-[`LEXICAL_JUDGE_DEPTH`] evidence: no unknown term, and one
/// of those hits [`lexically_supports`] the query. Returns why it failed, or None.
pub(crate) fn lexical_judgement(judge: &Judge, fts: &[Evidence]) -> Option<String> {
    if fts.is_empty() {
        return Some("no document holds any query term".into());
    }
    if !judge.unknown.is_empty() {
        // fts5 DID return hits here (`fts` is non-empty): "occur in no searchable document" read as a
        // contradiction next to a hit "via fts5", so say what the keyword hits lack instead.
        return Some(format!(
            "the keyword hits match only other words of the query, none of [{}]",
            judge.unknown.join(", ")
        ));
    }
    if fts.iter().any(|e| lexically_supports(judge, e)) {
        return None;
    }
    if fts.iter().any(|e| judge.supports(e)) {
        return Some(format!(
            "long query ({} findable terms): no fts5 top-{LEXICAL_JUDGE_DEPTH} hit holds {LONG_QUERY_MIN_SHARE} of its weighted terms",
            judge.findable_count()
        ));
    }
    match fts.iter().filter(|e| e.constraints_ok).map(Evidence::held_count).max() {
        // Enough terms but no hit supports: the count is not why it failed, the subject is.
        Some(best) => lexical_verdict(&[], best, judge.findable_count()).or_else(|| {
            let closest = fts
                .iter()
                .filter(|e| e.constraints_ok)
                .map(|e| judge.lacks(e))
                .filter(|l| !l.is_empty())
                .min_by_key(Vec::len);
            Some(match closest {
                Some(l) => format!(
                    "the closest fts5 top-{LEXICAL_JUDGE_DEPTH} match lacks [{}]",
                    l.join(", ")
                ),
                None => format!(
                    "no fts5 top-{LEXICAL_JUDGE_DEPTH} hit holds the subject of this short query \
                     (a rare findable term)"
                ),
            })
        }),
        None => Some(format!(
            "no fts5 top-{LEXICAL_JUDGE_DEPTH} hit holds the number(s) [{}] and a correction for \
             each typo [{}]",
            judge.numbers.join(", "),
            judge.typos.iter().map(|t| t.0.as_str()).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Test-shaped wrapper: build the judge and judge `top_ids` as fts5's ranking.
#[cfg(test)]
fn lexical_rule(store: &Store, query: &str, top_ids: &[i64]) -> Result<Option<String>> {
    let judge = Judge::build(store, query)?;
    let ev = judge.evidences(store, top_ids)?;
    Ok(lexical_judgement(&judge, &ev))
}

/// The bge rule's pure verdict: one of bge's top 5 must hold `floor` of the weighted terms.
pub(crate) fn bge_verdict(support: f64, floor: f64) -> Option<String> {
    (support < floor).then(|| format!("bge top-5 lexical support {support:.3} < {floor:.3}"))
}

/// The bge rule over bge's own top 5: support = the best [`weighted_share`] among the candidates
/// that themselves [`Judge::supports`] the query (Codex loop2-r1 #4: a password-only decoy used
/// to lend "kestra password" 0.5 while the real answer sat at bge rank 6). Pure, so the decision
/// is tested without an ONNX model.
pub(crate) fn bge_judgement(judge: &Judge, bge: &[Evidence], floor: f64) -> (f64, Option<String>) {
    let support = bge
        .iter()
        .filter(|e| judge.supports(e))
        .map(|e| weighted_share(&e.held, &judge.weights))
        .fold(0.0, f64::max);
    (support, bge_verdict(support, floor))
}

/// [`bge_judgement`] over bge's ranked output (`scored`, best first).
#[cfg(test)]
pub(crate) fn bge_rule(
    store: &Store,
    judge: &Judge,
    scored: &[(i64, f32)],
    floor: f64,
) -> Result<(f64, Option<String>)> {
    let ids: Vec<i64> = scored.iter().map(|s| s.0).collect();
    Ok(bge_judgement(judge, &judge.evidences(store, &ids)?, floor))
}

/// Every rule that applies, judged over the evidence `fused_sourced` collected: the lexical rule
/// when fts5's ranking was judged, the bge rule when bge ran. Returns one reason per failed rule.
pub(crate) fn query_reasons(
    judge: &Judge,
    fts: Option<&[Evidence]>,
    bge: Option<&[Evidence]>,
    floor: f64,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if let Some(f) = fts {
        reasons.extend(lexical_judgement(judge, f));
    }
    if let Some(b) = bge {
        reasons.extend(bge_judgement(judge, b, floor).1);
    }
    reasons
}

/// How many fused candidates to rank before the constraint filter: `top`, or the whole candidate
/// pool when the query carries number/typo constraints, so a dropped hit is replaced from below.
pub(crate) fn constrained_pool(judge: Option<&Judge>, ranked: &[Ranked], top: usize) -> usize {
    match judge {
        Some(j) if j.has_constraints() => top.max(ranked.iter().map(|r| r.ids.len()).sum()),
        _ => top,
    }
}

fn want_bge(want: &[String]) -> bool {
    want.iter().any(|c| c == "bge" || c == "vector" || c == "lancedb")
}

/// The query-level decision: abstain iff ANY applicable rule failed. No channel can buy the
/// query back (Codex r3 P2: `fts5,graph` used to answer whenever graph returned anything, and
/// released the rejected fts5 candidates into fusion).
pub(crate) fn query_abstains(abstain_on: bool, reasons: &[String]) -> bool {
    abstain_on && !reasons.is_empty()
}

/// Run the requested channels and fuse via RRF. `channels` is the comma-split
/// list from --channels; callers validate it first with [`validate_channels`].
#[cfg(test)] // the binary calls `fused_sourced`; tests keep the old shape
pub fn fused(
    store: &Store,
    bge: Option<&BgeSearch>,
    query: &str,
    top: usize,
    channels: &[String],
) -> Result<Vec<Hit>> {
    fused_with_qwen3(store, bge, None, query, top, channels)
}

/// `fused` with the Qwen3 channel available. Drops the per-result source label; callers that
/// print results use [`fused_sourced`].
#[cfg(test)] // the binary calls `fused_sourced`; tests keep the old shape
pub fn fused_with_qwen3(
    store: &Store,
    bge: Option<&BgeSearch>,
    qwen: Option<&crate::qwen3::Qwen3Search>,
    query: &str,
    top: usize,
    channels: &[String],
) -> Result<Vec<Hit>> {
    Ok(fused_sourced(store, bge, qwen, query, top, channels)?
        .into_iter()
        .map(|(h, _)| h)
        .collect())
}

/// The real implementation: every hit comes back with its [`Provenance`]: the channel(s) that
/// produced it, each channel's rank and raw score, and the RRF value.
///
/// `Hit.score` is the LEGACY single score, unchanged: the value supplied by whichever channel
/// FIRST inserted the doc into `id_hit`, in execution order fts5, rg, bge, qwen3, gemini, voyage,
/// graph. fts5 gives -bm25, bge its cosine, rg / qwen3 / gemini / voyage 0.0 (`hit_by_id`); a
/// graph SEED is a cloned fts5 hit and keeps its BM25, a NEIGHBOR gives 0.0. So an `rg+bge` hit
/// has `score` 0.0 while `scores.bge` holds the real cosine. Use `scores` to compare like with like.
/// Drop every candidate the current reader may not see, from the channel's id list AND from
/// `id_hit`. A no-op (no SQL) for the operator. Runs per channel, right where the ids are produced,
/// so no abstain judge, dedup merge or score map ever holds an out-of-scope id.
fn scope_gate(store: &Store, ids: &mut Vec<i64>, id_hit: &mut HashMap<i64, Hit>) -> Result<()> {
    if let Some(ok) = store.allowed_ids(ids)? {
        ids.retain(|id| {
            let keep = ok.contains(id);
            if !keep {
                id_hit.remove(id);
            }
            keep
        });
    }
    Ok(())
}

pub fn fused_sourced(
    store: &Store,
    bge: Option<&BgeSearch>,
    qwen: Option<&crate::qwen3::Qwen3Search>,
    query: &str,
    top: usize,
    channels: &[String],
) -> Result<Vec<(Hit, Provenance)>> {
    let out = prof::time("query total", || fused_sourced_inner(store, bge, qwen, query, top, channels));
    prof::report(query, top);
    out
}

/// `RECALL_PROFILE=1`: wall time and call count per stage of each query, one stderr line per query
/// (the daemon's log, or the CLI's stderr). Counters, not sampling: the stages are the expensive
/// calls a query makes, so a slow query names its own hot spot.
pub(crate) mod prof {
    use std::cell::RefCell;
    use std::sync::OnceLock;

    thread_local! {
        static STAGES: RefCell<Vec<(&'static str, std::time::Duration, u32)>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("RECALL_PROFILE").is_some_and(|v| v != "0"))
    }

    thread_local! {
        /// The running daemon query's cancel flag (set by the daemon's watchdog).
        static CANCEL: RefCell<Option<std::sync::Arc<std::sync::atomic::AtomicBool>>> = const { RefCell::new(None) };
    }

    pub(crate) fn set_cancel(flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>) {
        CANCEL.with(|c| *c.borrow_mut() = flag);
    }

    pub(crate) fn cancelled() -> bool {
        CANCEL.with(|c| c.borrow().as_ref().is_some_and(|f| f.load(std::sync::atomic::Ordering::SeqCst)))
    }

    /// Run one stage of a query, timed under `RECALL_PROFILE`. A cancelled query stops at the next
    /// stage boundary: the fts searches, the per-hit term probes, hit fetches and the vector search
    /// pass through here, so cancelling them does not depend on SQLite noticing an interrupt (Codex
    /// conc r2 #1: short statements never reach the progress handler). Other short SQL on the path
    /// (judge ids, term frequencies, the snapshot) relies on the interrupt alone.
    pub(crate) fn time<T>(stage: &'static str, f: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
        if cancelled() {
            anyhow::bail!("query cancelled");
        }
        if !on() {
            return f();
        }
        let t = std::time::Instant::now();
        let out = f();
        let took = t.elapsed();
        STAGES.with(|s| {
            let mut s = s.borrow_mut();
            match s.iter_mut().find(|(n, _, _)| *n == stage) {
                Some((_, d, c)) => {
                    *d += took;
                    *c += 1;
                }
                None => s.push((stage, took, 1)),
            }
        });
        out
    }

    pub(crate) fn report(query: &str, top: usize) {
        if !on() {
            return;
        }
        let stages = STAGES.with(|s| std::mem::take(&mut *s.borrow_mut()));
        let parts: Vec<String> =
            stages.iter().map(|(n, d, c)| format!("{n}={:.1}ms/{c}", d.as_secs_f64() * 1000.0)).collect();
        eprintln!("recall profile k={top} q={:?}: {}", query.chars().take(60).collect::<String>(), parts.join(" "));
    }
}

fn fused_sourced_inner(
    store: &Store,
    bge: Option<&BgeSearch>,
    qwen: Option<&crate::qwen3::Qwen3Search>,
    query: &str,
    top: usize,
    channels: &[String],
) -> Result<Vec<(Hit, Provenance)>> {
    let want: Vec<String> = channels.iter().map(|c| c.trim().to_lowercase()).collect();
    // Tenancy: resident vector channels skip rows this reader cannot see, so they still fill k.
    let _denied = crate::scope::DeniedGuard::install(store.denied_ids()?);
    // Raw per-channel scores for [`Provenance`]; output only.
    let mut raw: HashMap<String, HashMap<i64, f64>> = HashMap::new();
    let depth = fusion_depth(top); // pull deeper per channel than we return
    let abstain_on = abstain_enabled();
    let trace = abstain_trace();
    let fts5_requested = want
        .iter()
        .any(|c| c == "fts5" || c == "fts" || c == "sqlite-fts");

    let mut ranked: Vec<Ranked> = Vec::new();
    let mut id_hit: HashMap<i64, Hit> = HashMap::new();
    let mut fts5_seed_hits: Option<Vec<Hit>> = None;
    // Why the query abstained: one entry per rule that failed, printed on one stderr line.
    //
    // Abstention is a decision about the QUERY, not about a channel, and it is all-or-nothing:
    // when every rule that applies passes, the normal fused ranking comes back unchanged, so
    // abstain can never reorder or thin out a confident answer. Round 3 made it a conjunction —
    // the lexical rule (fts5's top hit) runs for every channel set that includes fts5 or bge, the
    // bge rule whenever bge ran, and ANY failure abstains. Round 2 abstained only when every
    // rule failed, so fused passed 10 of 12 fresh nonsense sentences that fts5 alone stopped.
    let mut abstained: Vec<String> = Vec::new();
    let judging = abstain_on || trace;
    // Built once, only when a rule will run. bge-only builds it even when the model is
    // unavailable (loop2 r3, Codex r2 HIGH #4): the lexical rule still applies, so a nonsense
    // query cannot reach the ungated fts5 rescue just because the vector channel is down.
    // Graph too (Codex v3 r1 HIGH #4): graph-only used to build no judge, so invented words plus
    // one searchable word found a seed, produced graph hits, skipped the gated rescue and answered.
    // v3 r4 (Codex r4 HIGH #1): EVERY channel set, so rg / qwen3 / gemini / voyage answers carry
    // `lacks` and answer to the lexical rule like the rest; they used to build no judge at all.
    let graph_requested = want.iter().any(|c| c == "graph" || c == "surreal");
    let mut judge: Option<Judge> = if judging { Some(Judge::build(store, query)?) } else { None };
    // Evidence each rule judges; None = that rule does not apply to this channel set.
    let (mut fts_ev, mut bge_ev): (Option<Vec<Evidence>>, Option<Vec<Evidence>>) = (None, None);
    let floor = env_f64(BGE_MIN_SUPPORT_ENV, BGE_MIN_SUPPORT);
    let mut trace_fields: Vec<String> = Vec::new();
    let bge_requested = want_bge(&want);

    // Backlog #8 cap split (PLAN.md section 2): the ranking MATCH reads up to 256 tokens, the
    // judge keeps its 64-token terms AND evidence, so no abstain decision changes. A constrained
    // long query (numbers / typos) also keeps the 64-token RANKING (R1): a widened ranking could
    // push the one doc holding "8784" below the fusion depth behind suffix-matching decoys.
    let long_query = crate::store::exceeds_judge_cap(query);
    // Abstain only LABELS or REFUSES; it never changes WHICH hits come back or their order (lead 2026-11-06,
    // VibeMem: abstain on vs off changed the top-1 hit). So the ranking search is the same with abstain on or
    // off; the judge reads its own evidence search (`fts_judge_ids`).
    let keep_judged_ranking = |_: &Option<Judge>| false;
    if fts5_requested {
        let hits = if keep_judged_ranking(&judge) {
            store.fts_search_judged(query, depth)?
        } else {
            store.fts_search(query, depth)?
        };
        let mut ids: Vec<i64> = hits.iter().map(|h| h.id).collect();
        for h in &hits {
            id_hit.entry(h.id).or_insert_with(|| h.clone());
        }
        scope_gate(store, &mut ids, &mut id_hit)?;
        let hits: Vec<Hit> = hits.into_iter().filter(|h| ids.contains(&h.id)).collect();
        raw.insert("fts5".into(), hits.iter().map(|h| (h.id, h.score)).collect());
        if let Some(j) = &judge {
            let judge_ids = if long_query { store.fts_judge_ids(query, depth)? } else { ids.clone() };
            fts_ev = Some(j.evidences(store, &judge_ids)?);
        }
        fts5_seed_hits = Some(hits);
        ranked.push(Ranked {
            channel: "fts5".into(),
            ids,
        });
    }
    if want.iter().any(|c| c == "rg" || c == "ripgrep") {
        let mut ids = channel_ids(store, query, "rg", depth, &mut id_hit)?;
        scope_gate(store, &mut ids, &mut id_hit)?;
        ranked.push(Ranked {
            channel: "rg".into(),
            ids,
        });
    }
    if bge_requested {
        // bge alone still answers to the lexical rule: whether the query's terms co-occur in any
        // document does not depend on which channel ranks the documents.
        if let (Some(j), false) = (&judge, fts5_requested) {
            let top = store.fts_judge_ids(query, LEXICAL_JUDGE_DEPTH)?;
            fts_ev = Some(j.evidences(store, &top)?);
        }
        let mut scored = bge_scored(store, bge, query, depth, &mut id_hit)?;
        // Before the judge sees bge's evidence: abstaining on a doc the reader cannot see would
        // reveal that it exists.
        let mut bge_ids: Vec<i64> = scored.iter().map(|s| s.0).collect();
        scope_gate(store, &mut bge_ids, &mut id_hit)?;
        scored.retain(|s| bge_ids.contains(&s.0));
        if bge.is_some() {
            raw.insert("bge".into(), scored.iter().map(|(id, c)| (*id, *c as f64)).collect());
        }
        // Lexical support among bge's top 5. The cosine itself cannot gate: on a private question set the top
        // cosine of real questions (p1 0.680, p50 0.780) overlaps every nonsense probe (0.700 to
        // 0.814 — a 40-char hex string scores 0.814 against notes full of hashes).
        if let (Some(j), true) = (&judge, bge.is_some()) {
            let ids: Vec<i64> = scored.iter().map(|s| s.0).collect();
            let ev = j.evidences(store, &ids)?;
            if trace {
                let support = bge_judgement(j, &ev, floor).0;
                let cos = scored.first().map_or(0.0, |s| s.1);
                trace_fields.push(format!("bge_top_cos={cos:.4} bge_support5={support:.4}"));
            }
            bge_ev = Some(ev);
        }
        ranked.push(Ranked {
            channel: "bge".into(),
            ids: scored.into_iter().map(|(id, _)| id).collect(),
        });
    }
    if want.iter().any(|c| c == "qwen3" || c == "qwen") {
        let mut ids = qwen3_ids(store, qwen, query, depth, &mut id_hit)?;
        scope_gate(store, &mut ids, &mut id_hit)?;
        ranked.push(Ranked {
            channel: "qwen3".into(),
            ids,
        });
    }
    #[cfg(feature = "gemini")]
    if want
        .iter()
        .any(|c| c == "gemini" || c == "embeddings4" || c == "vertex")
    {
        // Same fail-open contract as voyage: never Err on a cloud failure, return an
        // empty list and let the rescue tier below serve fts5 instead (loudly, see below).
        let mut ids = crate::gemini::gemini_ids(store, query, depth, &mut id_hit)?;
        scope_gate(store, &mut ids, &mut id_hit)?;
        ranked.push(Ranked {
            channel: "gemini".into(),
            ids,
        });
    }
    #[cfg(feature = "voyage")]
    if want
        .iter()
        .any(|c| c == "voyage" || c == "embeddings2" || c == "dense")
    {
        // Fail-open by contract: voyage_ids never returns Err for a cloud
        // failure, it returns an empty list and the fusion below proceeds
        // with whatever other channels produced.
        let mut ids = crate::voyage::voyage_ids(store, query, depth, &mut id_hit)?;
        scope_gate(store, &mut ids, &mut id_hit)?;
        ranked.push(Ranked {
            channel: "voyage".into(),
            ids,
        });
    }

    if graph_requested {
        let mut ids = if let Some(seeds) = fts5_seed_hits.as_deref() {
            graph_ids_from_seeds(store, seeds, depth, &mut id_hit)?
        } else {
            channel_ids(store, query, "graph", depth, &mut id_hit)?
        };
        scope_gate(store, &mut ids, &mut id_hit)?;
        ranked.push(Ranked {
            channel: "graph".into(),
            ids,
        });
    }

    // Any channel set without fts5 (graph, rg, qwen3, gemini, voyage) answers to the lexical rule
    // exactly as bge alone does: whether the query's terms co-occur in some document does not
    // depend on which channel ranks the documents.
    if let (Some(j), None) = (&judge, &fts_ev) {
        let top = store.fts_judge_ids(query, LEXICAL_JUDGE_DEPTH)?;
        fts_ev = Some(j.evidences(store, &top)?);
    }
    if trace {
        eprintln!(
            "abstain-trace q={} {}",
            serde_json::to_string(query).unwrap_or_default(),
            trace_fields.join(" ")
        );
        if let Some(j) = &judge {
            let (f, b) = (fts_ev.as_deref().unwrap_or(&[]), bge_ev.as_deref().unwrap_or(&[]));
            eprintln!("abstain-json {}", j.trace_json(query, f, b));
        }
    }
    if let Some(j) = &judge {
        abstained = query_reasons(j, fts_ev.as_deref(), bge_ev.as_deref(), floor);
    }
    // ONE decision for the whole query (lme-improve R2): confident, closest match, or abstain. A
    // natural-language question that failed a rule but holds real lexical evidence is answered as
    // an explicit CLOSEST MATCH (every hit `closest_match`), not with silence (K8: confident-wrong is
    // worst, silence next). The confident path is unchanged. feat/abstain-calibrated: the
    // corpus-calibrated path ([`Judge::may_answer_closest_calibrated`]) also answers keyword
    // queries and short-turn corpora whose evidence beats the corpus's phantom probes.
    let labels = abstain_labels();
    let may_close = |j: &Option<Judge>, fts: Option<&[Evidence]>| -> Result<bool> {
        let Some(j) = j.as_ref().filter(|_| abstain_on) else { return Ok(false) };
        if labels {
            return Ok(true);
        }
        let (ok, cal) = j.may_answer_closest_calibrated(store, query, fts)?;
        if let (true, Some((s, t))) = (trace, cal) {
            eprintln!("answerability: closest_score={s:.4} phantom_threshold={t:.4} pass={ok}");
        }
        Ok(ok)
    };
    let mut closest = query_abstains(abstain_on, &abstained) && may_close(&judge, fts_ev.as_deref())?;
    if closest {
        closest_note(&format!("closest: no confident match ({}); closest matches follow", abstained.join("; ")));
    } else if query_abstains(abstain_on, &abstained) {
        eprintln!("abstain: no confident match ({})", abstained.join("; "));
        return Ok(Vec::new());
    }

    // RESCUE TIER: if every requested channel FAILED (came back empty without abstaining),
    // fall back to fts5 — and say so, on stderr and in every result's `source`. The rescue answers
    // to the lexical rule like any fts5 ranking (loop2 r3, Codex r2 HIGH #4): `source=fts5-rescue`
    // names the channel, it is not a no-match marker, so a rescue the rule rejects abstains.
    //
    // Deliberately a rescue, NOT fusion: adding fts5 as a co-equal RRF channel
    // measurably degrades semantic queries (see the default-channel note in
    // main.rs — 4 of 5 probe queries regressed), because plain RRF lets a 0.32
    // channel outvote a 0.71 one. Here fts5 only runs when the alternative is an
    // empty result set.
    // Under the label policy a failed rule only LABELS, so it must not suppress the rescue either (Codex nr1 #1:
    // `zorblat cat luna` on an unavailable qwen3 was rescued with abstain off and empty with it on).
    if rescue_needed(&ranked, !abstained.is_empty() && !labels, fts5_requested) {
        // A long query builds its judge BEFORE the rescue ranking is chosen, so R1 can see its
        // constraints even on a qwen3/gemini/voyage-only request (PLAN.md r3 #1).
        if judging && long_query && judge.is_none() {
            judge = Some(Judge::build(store, query)?);
        }
        let hits = if keep_judged_ranking(&judge) {
            store.fts_search_judged(query, depth)?
        } else {
            store.fts_search(query, depth)?
        };
        let mut ids = Vec::new();
        for h in &hits {
            ids.push(h.id);
            id_hit.entry(h.id).or_insert_with(|| h.clone());
        }
        scope_gate(store, &mut ids, &mut id_hit)?;
        let hits: Vec<Hit> = hits.into_iter().filter(|h| ids.contains(&h.id)).collect();
        raw.insert("fts5-rescue".into(), hits.iter().map(|h| (h.id, h.score)).collect());
        if !ids.is_empty() && judging {
            if judge.is_none() {
                judge = Some(Judge::build(store, query)?);
            }
            if let Some(j) = &judge {
                let judge_ids = if long_query { store.fts_judge_ids(query, depth)? } else { ids.clone() };
                let ev = j.evidences(store, &judge_ids)?;
                let reasons: Vec<String> = lexical_judgement(j, &ev).into_iter().collect();
                // The rescue ranking WAS judged (Codex v3 r3 HIGH #2): record it, so the returned
                // set gets the supporting-hit check below even for a short query.
                fts_ev = Some(ev);
                if query_abstains(abstain_on, &reasons) && labels {
                    closest_note(&format!("closest: no confident match (fts5 rescue: {}); closest matches follow", reasons.join("; ")));
                    closest = true;
                } else if query_abstains(abstain_on, &reasons) {
                    eprintln!(
                        "abstain: no confident match (fts5 rescue: {})",
                        reasons.join("; ")
                    );
                    return Ok(Vec::new());
                }
            }
        }
        if !ids.is_empty() {
            eprintln!(
                "RESCUE: every requested channel ({}) returned nothing; serving fts5 keyword \
                 results, labelled source=fts5-rescue",
                want.join(",")
            );
            ranked.push(Ranked {
                channel: "fts5-rescue".into(),
                ids,
            });
        }
    }

    // A closest-match decision with nothing to show (every requested channel came back empty; the
    // rescue never runs once a rule failed) is an abstention, said as one (Codex lme r1 #7).
    if closest && ranked.iter().all(|r| r.ids.is_empty()) {
        eprintln!("abstain: no confident match ({}); no candidates to offer", abstained.join("; "));
        return Ok(Vec::new());
    }

    // The kept copy of a case-duplicate file is the best-ranked one whether or not abstain is on (it used to
    // prefer the judge-approved copy, so abstain on vs off could return a different doc id).
    let merged = collapse_case_duplicates(&mut ranked, &mut raw, &id_hit, &|_| false);

    // LEXICAL RERANK, default OFF. Runs over the FULL candidate pool, before truncation to
    // `top` — reranking the already-truncated list could only reorder ten documents that were
    // going to be returned anyway, and the whole point is to promote something from rank 11-50.
    // That band holds 38% of the questions neither channel answers at top-10.
    let fused_ids = if rerank_enabled() {
        let mut pool: Vec<i64> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for r in &ranked {
            for id in &r.ids {
                if seen.insert(*id) {
                    pool.push(*id);
                }
            }
        }
        let text = store.bodies_for_ids(&pool)?;
        let terms = rerank_query_terms(query);
        // Not truncated here (Codex v3 r1 #5): the constraint filter below must be able to
        // replace a dropped hit from deeper in the reranked pool.
        rerank_order(&ranked, &terms, &text, rerank_coverage_weight())
    } else {
        // the CE reranker reorders a pool deeper than `top` (abstain-independent: no constrained_pool, which
        // widened the pool only when abstain was on)
        let pool = if crate::ce::mode() == crate::ce::Mode::Rerank { top.max(crate::ce::depth()) } else { top };
        fuse_with_scores(&ranked, &raw, pool, &id_hit, Some(query))
    };
    // Optional cross-encoder (crate::ce, default off): rerank re-orders the fused head here, before the
    // constraint filter and truncation; score only labels the returned hits further down.
    let ce_mode = crate::ce::mode();
    let mut ce_scores: HashMap<i64, f64> = HashMap::new();
    let fused_ids = if ce_mode == crate::ce::Mode::Rerank {
        let head: Vec<i64> = fused_ids.iter().copied().take(crate::ce::depth()).collect();
        ce_scores = crate::ce::score(query, &head, &doc_bodies(store, &head)?)?;
        crate::ce::blend(&fused_ids, &ce_scores, crate::ce::alpha())
    } else {
        fused_ids
    };
    // Hard constraints (numbers, typo corrections) and "some returned hit supports the query" are judged on the
    // returned list AS RANKED (lead 2026-11-06): a hit that misses a number or a typo's correction is LABELLED
    // (`lacks`), never dropped, and no deeper hit is swapped in. A list without a supporting hit is a closest
    // match (label policy) or, under RECALL_ABSTAIN_POLICY=refuse, a closest match when the calibrated rule
    // admits it and an abstention otherwise. The ranked ids are identical with abstain on and off.
    // (Until 2026-11-06 failing hits were dropped and the best supporting candidate below the cut took the last
    // slot: loop2 r3 / Codex v3 r1-r3. The K8 label decision replaces that with labels.)
    let mut fused_ids: Vec<i64> = fused_ids.into_iter().take(top).collect();
    let mut evidence: HashMap<i64, Evidence> = HashMap::new();
    let judged = fts_ev.is_some() || bge_ev.is_some() || long_query;
    if let Some(j) = &judge {
        for id in &fused_ids {
            evidence.insert(*id, j.evidence(store, *id)?);
        }
        let ok = |e: &Evidence| if long_query { lexically_supports(j, e) } else { j.supports(e) };
        if abstain_on && judged && !closest && !fused_ids.is_empty() && !fused_ids.iter().any(|id| ok(&evidence[id])) {
            if may_close(&judge, fts_ev.as_deref())? {
                closest_note("closest: no confident match (no returned hit holds the judged terms); closest matches follow");
                closest = true;
            } else {
                eprintln!("abstain: no confident match (no returned hit holds the judged terms)");
                return Ok(Vec::new());
            }
        }
    }
    fused_ids.truncate(top);
    if ce_mode != crate::ce::Mode::Off {
        let missing: Vec<i64> = fused_ids.iter().copied().filter(|id| !ce_scores.contains_key(id)).collect();
        if !missing.is_empty() {
            let more = crate::ce::score(query, &missing, &doc_bodies(store, &missing)?)?;
            ce_scores.extend(more);
        }
    }
    // Optional answerability judge (crate::judge, default off): a confident result set whose passages the
    // judge says do not answer the question is LABELLED, never emptied; a judge that fails labels too.
    if crate::judge::kind() == crate::judge::Kind::Haiku && !closest && !fused_ids.is_empty() {
        let head: Vec<i64> = fused_ids.iter().copied().take(crate::judge::k()).collect();
        let bodies = doc_bodies(store, &head)?;
        let passages: Vec<String> = head.iter().filter_map(|id| bodies.get(id).cloned()).collect();
        let thr = crate::judge::threshold();
        match crate::judge::haiku(query, &passages) {
            Ok(p) if p <= thr => {
                eprintln!("judge: haiku p={p:.2} <= {thr:.2}: the passages may not answer this; closest matches follow");
                closest = true;
            }
            Ok(p) => {
                if trace {
                    eprintln!("judge: haiku p={p:.2} > {thr:.2}");
                }
            }
            Err(e) => {
                eprintln!("judge: haiku unavailable ({e:#}); results labelled as closest matches");
                closest = true;
            }
        }
    }
    let rrf_k = effective_rrf_k();
    let mut out = Vec::new();
    for id in fused_ids {
        if let Some(h) = id_hit.get(&id) {
            let mut p = provenance(id, &ranked, &raw, rrf_k);
            if let Some(c) = ce_scores.get(&id) {
                p.scores.insert("ce".into(), *c);
            }
            if let (Some(j), Some(e)) = (&judge, evidence.get(&id)) {
                p.lacks = j.lacks_of(store, id, e)?;
                // a hit without a correction of the query's typo, or without one of its numbers, is labelled
                // with it (both used to be dropped); explicit, because the label-subject list is capped (nr1 #2)
                for (t, fixes) in &j.typos {
                    if is_label_subject(t) && !p.lacks.contains(t) && !store.doc_matches_any(id, fixes)? {
                        p.lacks.push(t.clone());
                    }
                }
                for n in &j.numbers {
                    if !p.lacks.contains(n) && !store.doc_matches_term(id, n)? {
                        p.lacks.push(n.clone());
                    }
                }
                p.labels_truncated = j.labels_truncated;
                p.absent = j.absent_words();
            }
            p.merged = merged.get(&id).cloned().unwrap_or_default();
            out.push((h.clone(), p));
        }
    }
    // ONE weak-or-not decision for the whole list (cli-quality): a list in which EVERY hit lacks a query
    // subject term is weak too. That used to be a second, CLI-only "partial" banner on top of this flag.
    if judge.is_some() && abstain_on && !closest && partial_notice(out.iter().map(|(_, p)| p.lacks.as_slice())).is_some() {
        closest_note(&format!(
            "closest: no confident match (no returned hit mentions every query subject term; the first lacks [{}])",
            out[0].1.lacks.join(", ")
        ));
        closest = true;
    }
    // Pre-registered `agree` rule (written down and picked on the public DEV split before the TEST split was
    // scored): a top hit ranked
    // FIRST by both the keyword and the meaning channel is not called a closest match. No constant.
    let weak_list = closest;
    if closest && out.first().is_some_and(|(_, p)| top_hit_agrees(&p.ranks)) {
        closest_note("closest: overruled (the top hit is first in both fts5 and bge)");
        closest = false;
    }
    for (_, p) in &mut out {
        if let (true, Some(j)) = (weak_list, &judge) {
            // no hit holds an unknown word: say so on every hit of a weak list, an overruled one included
            for u in &j.unknown {
                if is_label_subject(u) && !p.lacks.contains(u) {
                    p.lacks.push(u.clone());
                }
            }
        }
        p.closest_match = closest;
    }
    Ok(out)
}

/// The `agree` rule: rank 1 (0-based 0) in BOTH fts5 and bge.
pub(crate) fn top_hit_agrees(ranks: &std::collections::BTreeMap<String, usize>) -> bool {
    ranks.get("fts5") == Some(&0) && ranks.get("bge") == Some(&0)
}

/// The stderr line for a result set whose EVERY hit lacks a query subject term, or None. Printed by
/// the CLI from the hits it prints, so a daemon-served answer carries it too (Codex v3 r3 HIGH #1:
/// a daemon-side line never reached the client).
pub fn partial_notice<'a>(lacks: impl IntoIterator<Item = &'a [String]>) -> Option<String> {
    let all: Vec<&[String]> = lacks.into_iter().collect();
    (!all.is_empty() && all.iter().all(|l| !l.is_empty())).then(|| {
        format!(
            "partial: no returned hit mentions every query subject term; closest matches only (first lacks [{}])",
            all[0].join(", ")
        )
    })
}

/// The per-hit label for a hit lacking subject terms: "closest match — does not mention X".
pub fn truncation_note(any_truncated: bool) -> Option<&'static str> {
    any_truncated.then_some(
        "labels truncated: the query has more than 64 subject words; only the first 64 were checked",
    )
}

/// The per-hit label for a hit lacking subject terms: "closest match — does not mention X".
pub fn lacks_label(lacks: &[String]) -> Option<String> {
    (!lacks.is_empty()).then(|| format!("closest match — does not mention {}", lacks.join(", ")))
}

/// `RECALL_VERBOSE=1` (or `recall query --verbose`): start-up narration and the closest-match reasons on stderr.
pub fn verbose_on() -> bool {
    matches!(std::env::var("RECALL_VERBOSE").ok().as_deref().map(str::trim), Some("1") | Some("on") | Some("true"))
}

/// The `closest_labels` setting: print the per-hit "closest match" label and the "No confident match"
/// banner. Default OFF (decided by a threshold fixed before the TEST split was scored: the pre-registered rule scored below 0.60
/// balanced accuracy on public TEST data, so the warning is not printed on every search). `closest_match`
/// stays in the JSON either way.
pub fn closest_labels_on() -> bool {
    crate::settings::Settings::load().closest_labels
}

/// A "closest: ..." reason line: stderr only when the labels are on, or verbose.
fn closest_note(msg: &str) {
    if closest_labels_on() || verbose_on() {
        eprintln!("{msg}");
    }
}

/// The one plain line (cli-quality addendum B, post-hoc): the main query words no note holds, a fact (df 0),
/// not a judgement. Printed whatever the `closest_labels` setting, when the list is not empty.
pub fn absent_line(absent: &[String]) -> Option<String> {
    (!absent.is_empty()).then(|| format!("Not found in the indexed text: {}. These are the nearest matches.", absent.join(", ")))
}

/// How the CLI shows hit `rank` (0-based): (muted "(weak)" tag, note line). Only a WEAK list (`closest`, one
/// decision for the whole list) calls its hits closest matches. In a confident list the top hit is the answer
/// and is never tagged; a lower hit lacking a subject term sits below the confidence cut and is muted as
/// "(weak)", not presented as an equal. Every gap is still named.
pub fn hit_note(rank: usize, closest: bool, lacks: &[String]) -> (bool, Option<String>) {
    if closest {
        return (false, lacks_label(lacks));
    }
    let weak = rank > 0 && !lacks.is_empty();
    (weak, (!lacks.is_empty()).then(|| format!("does not mention {}", lacks.join(", "))))
}

/// How many fused candidates below the cut the supporting-hit replacement in [`fused_sourced`]
/// inspects before the query abstains.
pub const REPLACEMENT_DEPTH: usize = 20;

/// One file, one result: doc ids whose paths differ only in letter case (or slash direction)
/// are the same file on Windows (NTFS is case-insensitive), so a rename like `agents.md` ->
/// `AGENTS.md` leaves two live rows. v3 r2: on the tuning title set 65 of 270 production
/// answers spent a top-5 slot on such a second copy. Every channel's list is rewritten to the
/// first id seen for each case-folded path (channels in execution order, best rank first), later
/// copies are dropped, and a channel's raw score moves to the kept id, so fusion counts the file
/// once and provenance still shows each channel's real score.
///
/// v3 r3 (Codex r2 HIGH #2): the kept id is the first copy `prefer` accepts (the one whose own
/// content the judge approves), else the first seen, so the approved evidence is never the copy
/// that gets dropped. A copy's raw score fills in only where the kept id has none of its own in
/// that channel, and every collapsed id is returned (kept id -> copies) for [`Provenance::merged`].
pub(crate) fn collapse_case_duplicates(
    ranked: &mut [Ranked],
    raw: &mut HashMap<String, HashMap<i64, f64>>,
    id_hit: &HashMap<i64, Hit>,
    prefer: &dyn Fn(i64) -> bool,
) -> HashMap<i64, Vec<i64>> {
    // An EMPTY path is not a file: every `recall save` observation has `path = ''`, and keying on it
    // collapsed all saved memories into whichever one ranked first (found by the tenancy E2E).
    let key = |id: i64| {
        id_hit
            .get(&id)
            .filter(|h| !h.path.is_empty())
            .map(|h| h.path.replace('\\', "/").to_lowercase())
    };
    // Every copy of each file, in channel order then rank order.
    let mut groups: HashMap<String, Vec<i64>> = HashMap::new();
    for r in ranked.iter() {
        for &id in &r.ids {
            if let Some(k) = key(id) {
                let g = groups.entry(k).or_default();
                if !g.contains(&id) {
                    g.push(id);
                }
            }
        }
    }
    let mut canon: HashMap<String, i64> = HashMap::new();
    let mut merged: HashMap<i64, Vec<i64>> = HashMap::new();
    for (k, g) in groups {
        if g.len() < 2 {
            continue;
        }
        let keep = g.iter().copied().find(|id| prefer(*id)).unwrap_or(g[0]);
        merged.insert(keep, g.iter().copied().filter(|id| *id != keep).collect());
        canon.insert(k, keep);
    }
    for r in ranked.iter_mut() {
        let mut seen = std::collections::HashSet::new();
        let mut ids = Vec::with_capacity(r.ids.len());
        for &id in &r.ids {
            let c = key(id).and_then(|k| canon.get(&k).copied()).unwrap_or(id);
            if !seen.insert(c) {
                continue;
            }
            ids.push(c);
        }
        if let Some(m) = raw.get_mut(&r.channel) {
            for &c in &ids {
                if m.contains_key(&c) {
                    continue;
                }
                // The kept id has no score of its own here: lend the best-ranked copy's.
                let copy = merged.get(&c).and_then(|cs| r_ids_first(&r.ids, cs));
                if let Some(s) = copy.and_then(|id| m.get(&id).copied()) {
                    m.insert(c, s);
                }
            }
        }
        r.ids = ids;
    }
    merged
}

/// The first id of `ranked_ids` that is in `among`.
fn r_ids_first(ranked_ids: &[i64], among: &[i64]) -> Option<i64> {
    ranked_ids.iter().copied().find(|id| among.contains(id))
}

/// Env flag for the lexical reranker. **Default OFF.**
///
/// Off by default on purpose: it is one variable away from reverted if it interacts badly with
/// real load, which is cheaper than a rebuild on a machine where the daemon holds a resident
/// vector matrix and a cold restart costs ~59 s.
const RERANK_ENV: &str = "RECALL_RERANK";
/// Weight on the coverage term, overridable for sweeps without a rebuild.
const RERANK_WEIGHT_ENV: &str = "RECALL_RERANK_COVERAGE_WEIGHT";
/// Tuned on HALF a private question set (even-indexed questions) and validated on the held-out half.
///
/// 🔴 The coverage score here is deliberately UNWEIGHTED — "does this query term appear at all",
/// not an IDF-weighted variant. Measured 2026-08-21, tuning each on split A and reporting split B
/// which the tuner never saw:
///
/// | variant | tuned w | split A | HELD-OUT B | delta vs fts5 on B |
/// |---|---|---|---|---|
/// | IDF-weighted | 0.5 | 0.7273 | 0.6207 | +0.0115 |
/// | **unweighted** | **0.3** | 0.7159 | **0.6552** | **+0.0460** |
///
/// The IDF variant scored higher on the split it was tuned on and generalised FOUR TIMES WORSE,
/// which is what selection effect looks like. The cheap one is also the honest one, and it needs
/// no corpus-wide document-frequency table in the query path.
const RERANK_COVERAGE_WEIGHT: f64 = 0.3;
/// Rank-offset for the reciprocal position term. Positions are comparable across channels;
/// raw channel scores are not (BM25 ~17-22 vs cosine 0.0), so only positions are fused.
const RERANK_RANK_OFFSET: f64 = 10.0;

fn rerank_enabled() -> bool {
    matches!(
        std::env::var(RERANK_ENV).ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

fn rerank_coverage_weight() -> f64 {
    std::env::var(RERANK_WEIGHT_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| *v >= 0.0 && *v <= 10.0)
        .unwrap_or(RERANK_COVERAGE_WEIGHT)
}

/// Two-byte function words when typed in lowercase ([`is_label_subject`]): every OTHER two-character
/// token is a subject ("AI", "Go", "UI", "qx", "IT"). A fixed list, never a length cut.
pub(crate) const SHORT_STOPWORDS: &[&str] = &[
    "of", "in", "to", "on", "at", "is", "it", "an", "or", "as", "by", "be", "we", "do", "if", "my",
    "me", "no", "so", "up", "am", "us", "he", "vs",
];

/// Function words of three or more letters ([`rerank_query_terms`], [`is_label_subject`]).
pub(crate) const RERANK_STOPWORDS: &[&str] = &[
    "and", "are", "was", "were", "the", "for", "you", "your", "with", "that", "this", "these",
    "those", "its", "how", "what", "which", "who", "whom", "when", "where", "why", "can",
    "could", "should", "would", "may", "might", "must", "shall", "will", "have", "has", "had",
    "does", "did", "not", "any", "all", "from", "into", "onto", "than", "then", "there",
    "here", "they", "them", "their", "our", "his", "her", "she", "him", "but", "out",
];

/// Query terms for coverage: lowercased, de-duplicated, stopwords and 1-2 char tokens dropped.
///
/// Without the stopword cut, "what is the" would be satisfied by every document in the corpus and
/// the coverage term would carry no signal at all — every candidate would score ~1.0.
pub(crate) fn rerank_query_terms(query: &str) -> Vec<String> {
    const STOP: &[&str] = RERANK_STOPWORDS;
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let lower = query.to_lowercase();
    for raw in lower.split(|c: char| !c.is_alphanumeric()) {
        if raw.len() <= 2 || STOP.contains(&raw) {
            continue;
        }
        if seen.insert(raw.to_string()) {
            out.push(raw.to_string());
        }
    }
    out
}

/// Fraction of distinct query terms that appear anywhere in `text`. `text` must already be
/// lowercased. Returns 1.0 for an empty term list so a stopword-only query cannot zero every
/// candidate and hand the ordering entirely to the rank term by accident.
pub(crate) fn rerank_coverage(terms: &[String], text: &str) -> f64 {
    if terms.is_empty() {
        return 1.0;
    }
    let hit = terms.iter().filter(|t| text.contains(t.as_str())).count();
    hit as f64 / terms.len() as f64
}

/// Score every candidate by fused rank position plus query-term coverage, best first.
///
/// Pure and separately testable: the caller supplies the per-channel id lists and the document
/// text, so the ordering logic can be unit-tested without a database or a live index.
pub(crate) fn rerank_order(
    ranked: &[Ranked],
    terms: &[String],
    text: &HashMap<i64, String>,
    weight: f64,
) -> Vec<i64> {
    let mut score: HashMap<i64, f64> = HashMap::new();
    for r in ranked {
        for (pos, id) in r.ids.iter().enumerate() {
            *score.entry(*id).or_insert(0.0) += 1.0 / (RERANK_RANK_OFFSET + pos as f64);
        }
    }
    for (id, s) in score.iter_mut() {
        if let Some(tx) = text.get(id) {
            *s += weight * rerank_coverage(terms, tx);
        }
    }
    let mut ids: Vec<(i64, f64)> = score.into_iter().collect();
    // Sort by score desc, then by id asc. The id tiebreak keeps the order DETERMINISTIC — a
    // HashMap's iteration order is not stable across runs, and a ranking that shuffles between
    // identical queries is untestable and unmeasurable.
    ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    ids.into_iter().map(|(id, _)| id).collect()
}

pub fn fusion_depth(top: usize) -> usize {
    top.saturating_mul(effective_fusion_depth_multiplier())
        .max(effective_min_fusion_depth())
}

/// Run a single channel and return its ranked ids, populating `id_hit` so the
/// caller can resolve ids back to `Hit`s. Exposed so the bench can compute each
/// channel ONCE per fixture and assemble multiple configs from the result,
/// instead of re-running the expensive rg scan for every config.
pub fn channel_ids(
    store: &Store,
    query: &str,
    channel: &str,
    depth: usize,
    id_hit: &mut HashMap<i64, Hit>,
) -> Result<Vec<i64>> {
    match channel {
        "fts5" | "fts" | "sqlite-fts" => {
            let hits = store.fts_search(query, depth)?;
            let mut ids = Vec::new();
            for h in hits {
                ids.push(h.id);
                id_hit.entry(h.id).or_insert(h);
            }
            Ok(ids)
        }
        "rg" | "ripgrep" => rg_channel(store, query, depth, id_hit),
        "graph" | "surreal" => graph_ids(store, query, depth, id_hit),
        _ => Ok(vec![]),
    }
}

/// Graph channel v0: seed with FTS anchor docs, then expand to neighboring docs
/// from the same indexed path neighborhoods. This keeps graph retrieval
/// deterministic and no-spend while the fuller entity/relation graph is still
/// being designed.
pub fn graph_ids(
    store: &Store,
    query: &str,
    depth: usize,
    id_hit: &mut HashMap<i64, Hit>,
) -> Result<Vec<i64>> {
    if depth == 0 {
        return Ok(Vec::new());
    }

    let seed_depth = depth.min(12).max(3);
    // Graph-only seeds stay on the judge's 64-token search (PLAN.md R2): wider seeds would make
    // the rescue unnecessary and so bypass the rescue's lexical gate on a long nonsense prefix.
    let seeds = store.fts_search_judged(query, seed_depth)?;
    graph_ids_from_seeds(store, &seeds, depth, id_hit)
}

fn graph_ids_from_seeds(
    store: &Store,
    seeds: &[Hit],
    depth: usize,
    id_hit: &mut HashMap<i64, Hit>,
) -> Result<Vec<i64>> {
    if depth == 0 {
        return Ok(Vec::new());
    }

    let mut ids = Vec::new();
    let seed_limit = depth.min(12).max(3);
    for h in seeds.iter().take(seed_limit) {
        if ids.len() >= depth {
            return Ok(ids);
        }
        ids.push(h.id);
        id_hit.entry(h.id).or_insert_with(|| h.clone());
    }

    let hits = store.graph_path_neighbors(&seeds[..seeds.len().min(seed_limit)], depth)?;
    for h in hits {
        if ids.len() >= depth {
            break;
        }
        if id_hit.contains_key(&h.id) {
            continue;
        }
        ids.push(h.id);
        id_hit.entry(h.id).or_insert(h);
    }
    Ok(ids)
}

/// bge channel: embed the query, cosine-rank all stored vectors, resolve the
/// top-`depth` ids back to `Hit`s. Returns empty when no vectors are loaded
/// (un-embedded store) so the channel is a graceful no-op until `recall embed`
/// has run. Exposed so the bench can reuse one loaded `BgeSearch` per run.
pub fn bge_ids(
    store: &Store,
    bge: Option<&BgeSearch>,
    query: &str,
    depth: usize,
    id_hit: &mut HashMap<i64, Hit>,
) -> Result<Vec<i64>> {
    Ok(bge_scored(store, bge, query, depth, id_hit)?
        .into_iter()
        .map(|(id, _)| id)
        .collect())
}

/// `bge_ids` with each document's best-chunk cosine. A doc the bge channel is first to surface
/// carries that cosine as its `score` (it used to print 0.0, which was the only channel tell).
fn bge_scored(
    store: &Store,
    bge: Option<&BgeSearch>,
    query: &str,
    depth: usize,
    id_hit: &mut HashMap<i64, Hit>,
) -> Result<Vec<(i64, f32)>> {
    let Some(bge) = bge else { return Ok(vec![]) };
    let scored = prof::time("bge_search", || bge.search_scored(query, depth))?;
    for (id, cos) in &scored {
        if !id_hit.contains_key(id) {
            if let Some(mut h) = prof::time("bge_hit_by_id", || store.hit_by_id(*id))? {
                h.score = *cos as f64;
                id_hit.insert(*id, h);
            }
        }
    }
    Ok(scored)
}

pub fn qwen3_ids(
    store: &Store,
    qwen: Option<&crate::qwen3::Qwen3Search>,
    query: &str,
    depth: usize,
    id_hit: &mut HashMap<i64, Hit>,
) -> Result<Vec<i64>> {
    let Some(qwen) = qwen else { return Ok(vec![]) };
    let ids = qwen.search_ids(query, depth)?;
    for id in &ids {
        if !id_hit.contains_key(id) {
            if let Some(h) = store.hit_by_id(*id)? {
                id_hit.insert(*id, h);
            }
        }
    }
    Ok(ids)
}

/// Pure RRF merge — exposed for unit testing.
pub fn rrf_merge(ranked: &[Ranked], top: usize) -> Vec<i64> {
    let base = rrf_merge_with_k(
        ranked,
        top.max(ranked.iter().map(|r| r.ids.len()).max().unwrap_or(0)),
        effective_rrf_k(),
    );
    apply_fusion_mode(ranked, &base, top)
}

#[allow(dead_code)] // Kept for focused tests and simple callers that do not have query text.
pub fn rrf_merge_with_hits(ranked: &[Ranked], top: usize, id_hit: &HashMap<i64, Hit>) -> Vec<i64> {
    rrf_merge_with_hits_and_query(ranked, top, id_hit, None)
}

pub fn rrf_merge_with_hits_and_query(
    ranked: &[Ranked],
    top: usize,
    id_hit: &HashMap<i64, Hit>,
    query: Option<&str>,
) -> Vec<i64> {
    let depth = top.max(ranked.iter().map(|r| r.ids.len()).max().unwrap_or(0));
    let base = rrf_merge_with_k(ranked, depth, effective_rrf_k());
    let fused = apply_fusion_mode(ranked, &base, depth);
    apply_denoise_mode(&fused, top, id_hit, query, ranked)
}

/// Raw stored body per id (the cross-encoder's document text).
fn doc_bodies(store: &Store, ids: &[i64]) -> Result<HashMap<i64, String>> {
    Ok(store.title_body_rows(ids)?.into_iter().map(|(id, _, body)| (id, body)).collect())
}

/// Fusion entry point of [`fused_sourced`]: [`convex_merge`] when `RECALL_FUSION_MODE=convex` and both
/// fts5 and bge returned raw scores, else the rank-based [`rrf_merge_with_hits_and_query`].
pub(crate) fn fuse_with_scores(
    ranked: &[Ranked],
    raw: &HashMap<String, HashMap<i64, f64>>,
    top: usize,
    id_hit: &HashMap<i64, Hit>,
    query: Option<&str>,
) -> Vec<i64> {
    warn_removed_denoise_mode(); // before any return, so convex fusion reports it too
    if effective_fusion_mode() == "convex" {
        if let Some(ids) = convex_merge(ranked, raw, top, effective_convex_bge_weight()) {
            return ids;
        }
    }
    rrf_merge_with_hits_and_query(ranked, top, id_hit, query)
}

const CONVEX_BGE_WEIGHT_ENV: &str = "RECALL_CONVEX_BGE_WEIGHT";
/// Chosen ONCE on pooled DEV halves (seed 20261105) of LongMemEval_S, LoCoMo dialog + session, SciFact,
/// NFCorpus, SciDocs, FiQA (offline re-fusion of captured 07a728a lists): mean
/// +1.67 pp over RRF k=60, no set below it. 0.5: +1.33; 0.7: +1.22. Frozen before any TEST-half run.
pub const CONVEX_BGE_WEIGHT: f64 = 0.6;
/// Default fusion since retrieval-quality (2026-11-06): convex beats RRF k=60 on every public DEV set
/// above; `RECALL_FUSION_MODE=rrf` restores the old order. Any channel set other than fts5+bge is RRF.
const DEFAULT_FUSION_MODE: &str = "convex";

pub fn effective_convex_bge_weight() -> f64 {
    let w = env_f64(CONVEX_BGE_WEIGHT_ENV, CONVEX_BGE_WEIGHT);
    if (0.0..=1.0).contains(&w) { w } else { CONVEX_BGE_WEIGHT }
}

/// Convex score fusion: each channel's RAW scores min-max normalised over its own fused-depth list,
/// then `w * bge + (1 - w) * fts5`; a doc missing from a channel's list scores 0 there. Ties: doc id
/// ascending. None unless exactly fts5 and bge ranked with raw scores (any other channel set falls back
/// to RRF, which needs no scores).
pub fn convex_merge(
    ranked: &[Ranked],
    raw: &HashMap<String, HashMap<i64, f64>>,
    top: usize,
    w: f64,
) -> Option<Vec<i64>> {
    let mut chans: Vec<&str> = ranked.iter().filter(|r| !r.ids.is_empty()).map(|r| r.channel.as_str()).collect();
    chans.sort_unstable();
    if chans != ["bge", "fts5"] {
        return None;
    }
    let mut score: HashMap<i64, f64> = HashMap::new();
    for r in ranked.iter().filter(|r| !r.ids.is_empty()) {
        let cw = if r.channel == "bge" { w } else { 1.0 - w };
        let m = raw.get(&r.channel)?;
        let vals: Vec<f64> = r.ids.iter().map(|id| m.get(id).copied().filter(|v| v.is_finite())).collect::<Option<_>>()?;
        let (lo, hi) = vals.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        if !(hi - lo).is_finite() {
            return None;
        }
        for (id, v) in r.ids.iter().zip(&vals) {
            let n = if hi > lo { (v - lo) / (hi - lo) } else { 1.0 };
            *score.entry(*id).or_insert(0.0) += cw * n;
        }
    }
    let mut scored: Vec<(i64, f64)> = score.into_iter().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    Some(scored.into_iter().take(top).map(|(id, _)| id).collect())
}

pub fn effective_rrf_k() -> f64 {
    std::env::var(RRF_K_ENV)
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(RRF_K)
}

pub fn effective_fusion_mode() -> String {
    std::env::var(FUSION_MODE_ENV)
        .ok()
        .map(|raw| raw.trim().to_ascii_lowercase())
        .filter(|mode| !mode.is_empty())
        .unwrap_or_else(|| DEFAULT_FUSION_MODE.to_string())
}

pub fn effective_denoise_mode() -> String {
    std::env::var(DENOISE_MODE_ENV)
        .ok()
        .map(|raw| raw.trim().to_ascii_lowercase())
        .filter(|mode| !mode.is_empty())
        .unwrap_or_else(|| DEFAULT_DENOISE_MODE.to_string())
}

pub fn effective_min_fusion_depth() -> usize {
    effective_bounded_usize(MIN_FUSION_DEPTH_ENV, MIN_FUSION_DEPTH, MAX_MIN_FUSION_DEPTH)
}

pub fn effective_fusion_depth_multiplier() -> usize {
    effective_bounded_usize(
        FUSION_DEPTH_MULTIPLIER_ENV,
        FUSION_DEPTH_MULTIPLIER,
        MAX_FUSION_DEPTH_MULTIPLIER,
    )
}

fn effective_bounded_usize(env_name: &str, default: usize, max: usize) -> usize {
    std::env::var(env_name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0 && *value <= max)
        .unwrap_or(default)
}

/// Pure RRF merge with an explicit k value. Exposed for controlled eval sweeps.
pub fn rrf_merge_with_k(ranked: &[Ranked], top: usize, rrf_k: f64) -> Vec<i64> {
    let mut score: HashMap<i64, f64> = HashMap::new();
    for r in ranked {
        for (rank, id) in r.ids.iter().enumerate() {
            *score.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + (rank as f64) + 1.0);
        }
    }
    let mut scored: Vec<(i64, f64)> = score.into_iter().collect();
    // Primary: RRF score descending. Secondary: doc id ascending — stable,
    // deterministic tie-break so equal-score results always come out in the
    // same order regardless of HashMap iteration order.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.into_iter().take(top).map(|(id, _)| id).collect()
}

fn apply_fusion_mode(ranked: &[Ranked], base: &[i64], top: usize) -> Vec<i64> {
    let mode = effective_fusion_mode();
    if mode.is_empty() || mode == "rrf" || ranked.len() < 2 {
        return base.iter().copied().take(top).collect();
    }
    match mode.as_str() {
        "bge_first" => ranked_channel_ids(ranked, "bge")
            .map(|ids| merge_prefix_then_base(&[], ids, ids.len(), base, top))
            .unwrap_or_else(|| base.iter().copied().take(top).collect()),
        "rrf1_bge4" => {
            merge_channel_prefix(&base[..base.len().min(1)], ranked, "bge", 4, base, top)
        }
        "rrf2_bge3" => {
            merge_channel_prefix(&base[..base.len().min(2)], ranked, "bge", 3, base, top)
        }
        "rrf3_bge2" => {
            merge_channel_prefix(&base[..base.len().min(3)], ranked, "bge", 2, base, top)
        }
        "rrf4_bge1" => {
            merge_channel_prefix(&base[..base.len().min(4)], ranked, "bge", 1, base, top)
        }
        "rrf2_bge4" => {
            merge_channel_prefix(&base[..base.len().min(2)], ranked, "bge", 4, base, top)
        }
        "rrf3_bge4" => {
            merge_channel_prefix(&base[..base.len().min(3)], ranked, "bge", 4, base, top)
        }
        "rrf4_bge4" => {
            merge_channel_prefix(&base[..base.len().min(4)], ranked, "bge", 4, base, top)
        }
        "rrf1_fts4" => {
            merge_channel_prefix(&base[..base.len().min(1)], ranked, "fts5", 4, base, top)
        }
        "rrf2_fts3" => {
            merge_channel_prefix(&base[..base.len().min(2)], ranked, "fts5", 3, base, top)
        }
        "rrf3_fts2" => {
            merge_channel_prefix(&base[..base.len().min(3)], ranked, "fts5", 2, base, top)
        }
        _ => base.iter().copied().take(top).collect(),
    }
}

fn ranked_channel_ids<'a>(ranked: &'a [Ranked], channel: &str) -> Option<&'a [i64]> {
    ranked
        .iter()
        .find(|r| r.channel.eq_ignore_ascii_case(channel))
        .map(|r| r.ids.as_slice())
}

fn merge_channel_prefix(
    prefix: &[i64],
    ranked: &[Ranked],
    channel: &str,
    inject_consider: usize,
    base: &[i64],
    top: usize,
) -> Vec<i64> {
    ranked_channel_ids(ranked, channel)
        .map(|ids| merge_prefix_then_base(prefix, ids, inject_consider, base, top))
        .unwrap_or_else(|| base.iter().copied().take(top).collect())
}

fn apply_denoise_mode(
    base: &[i64],
    top: usize,
    _id_hit: &HashMap<i64, Hit>,
    _query: Option<&str>,
    _source_ranked: &[Ranked],
) -> Vec<i64> {
    warn_removed_denoise_mode();
    base.iter().copied().take(top).collect()
}

/// The ops_noise modes were path heuristics tuned to one vault, off by default since 2026-10-29
/// and removed for the public release. A leftover setting is reported once, never ignored.
fn warn_removed_denoise_mode() {
    let mode = effective_denoise_mode();
    if !(mode.is_empty() || mode == "off" || mode == "none") {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!("recall: {DENOISE_MODE_ENV}={mode} is no longer supported (the ops_noise modes were removed); using off")
        });
    }
}

fn merge_prefix_then_base(
    prefix: &[i64],
    injected: &[i64],
    inject_consider: usize,
    base: &[i64],
    top: usize,
) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::new();
    for id in prefix {
        push_unique(&mut out, *id, top);
    }
    for id in injected.iter().take(inject_consider) {
        push_unique(&mut out, *id, top);
    }
    for id in base {
        push_unique(&mut out, *id, top);
    }
    out
}

fn push_unique(out: &mut Vec<i64>, id: i64, top: usize) {
    if out.len() < top && !out.contains(&id) {
        out.push(id);
    }
}

/// Kill rg if it hasn't exited within this duration — prevents a hung vault
/// scan from freezing the query path indefinitely on the live 1.6 GB corpus.
const RG_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// rg channel: shell out to rg.exe over the vault, map matched files to doc ids.
/// Files not yet indexed contribute nothing (no id to fuse on).
fn rg_channel(
    store: &Store,
    query: &str,
    depth: usize,
    id_hit: &mut HashMap<i64, Hit>,
) -> Result<Vec<i64>> {
    let terms: Vec<&str> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3)
        .collect();
    if terms.is_empty() {
        return Ok(vec![]);
    }
    let pattern = terms.join("|");
    let root = crate::store::vault_root();
    let mut child = match std::process::Command::new("rg")
        .args([
            "--files-with-matches",
            "-i",
            "--glob",
            "*.md",
            "--glob",
            "*.qmd",
            "-e",
            &pattern,
        ])
        .arg(&root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => {
            // rg unavailable: the channel contributes nothing, and says so once.
            crate::store::warn_rg_missing();
            return Ok(vec![]);
        }
    };

    // Drain stdout on a separate thread so a broad query whose match list
    // overflows the OS pipe buffer (~64 KB) can't deadlock rg: otherwise rg
    // blocks writing to a full pipe, never exits, and always trips RG_TIMEOUT
    // below — silently dropping the channel even though rg would finish fast if
    // drained. With a concurrent reader, rg streams to completion (or is killed
    // on a real timeout, which closes the pipe and ends the read).
    let Some(mut out_pipe) = child.stdout.take() else {
        let _ = child.kill();
        return Ok(vec![]);
    };
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = std::io::Read::read_to_end(&mut out_pipe, &mut buf);
        buf
    });

    let deadline = std::time::Instant::now() + RG_TIMEOUT;
    loop {
        match child.try_wait() {
            // 0 = matches, 1 = no match; anything else (2 = error) is a broken channel, said aloud.
            Ok(Some(status)) => {
                if !matches!(status.code(), Some(0) | Some(1)) {
                    eprintln!("recall: rg channel failed ({status}); its results are missing from this query");
                }
                break;
            }
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            _ => {
                // Timed out or wait error — kill to avoid a zombie, join the
                // reader so its thread/handle can't leak, and return empty.
                eprintln!("recall: rg channel timed out; its results are missing from this query");
                let _ = child.kill();
                let _ = reader.join();
                return Ok(vec![]);
            }
        }
    }

    let buf = reader.join().unwrap_or_default();
    let stdout = String::from_utf8_lossy(&buf);
    let mut ids = Vec::new();
    for line in stdout.lines().take(depth) {
        let rel = to_rel(&root, line);
        if let Some(hit) = lookup_path(store, &rel)? {
            ids.push(hit.id);
            id_hit.entry(hit.id).or_insert(hit);
        }
    }
    Ok(ids)
}

fn to_rel(root: &Path, abs: &str) -> String {
    let abs_norm = abs.replace('\\', "/");
    let root_norm = root.to_string_lossy().replace('\\', "/");
    abs_norm
        .strip_prefix(&root_norm)
        .map(|s| s.trim_start_matches('/').to_string())
        .unwrap_or(abs_norm)
}

fn lookup_path(store: &Store, rel: &str) -> Result<Option<Hit>> {
    // Scoped: rg reads files on disk, and this is where a disk path becomes a row a reader may see.
    let mut stmt = store.conn.prepare(&format!(
        "SELECT id, path, title FROM documents WHERE path = ?1 AND valid_to IS NULL{} LIMIT 1",
        crate::scope::clause("documents")
    ))?;
    let mut rows = stmt.query(rusqlite::params![rel])?;
    if let Some(r) = rows.next()? {
        Ok(Some(Hit {
            id: r.get(0)?,
            path: r.get(1)?,
            title: r.get(2)?,
            snippet: String::new(),
            score: 0.0,
        }))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    /// Shared with daemon.rs tests: every test that reads RECALL_* env through `fused_sourced`.
    /// The env lock plus a snapshot of every variable the abstain policy and the judge read; dropping it restores
    /// them, also when the test panics (Codex judge r1 #3).
    pub(crate) struct EnvLock {
        _g: std::sync::MutexGuard<'static, ()>,
        saved: Vec<(std::ffi::OsString, Option<std::ffi::OsString>)>,
    }

    /// Every RECALL_* variable name, compared as raw bytes (a non-UTF-8 name is still caught).
    fn recall_vars() -> Vec<std::ffi::OsString> {
        std::env::vars_os().map(|(k, _)| k).filter(|k| k.as_encoded_bytes().starts_with(b"RECALL_")).collect()
    }

    impl Drop for EnvLock {
        fn drop(&mut self) {
            // a RECALL_* knob a test set that was absent before is removed too (Codex env1 #2)
            for k in recall_vars() {
                std::env::remove_var(k);
            }
            for (k, v) in &self.saved {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    pub(crate) fn env_lock() -> EnvLock {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        // A failed test must not poison every later one (the guard protects env, not data).
        let g = LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner());
        // EVERY inherited RECALL_* variable (a shell that once had RECALL_ABSTAIN=off silently turned the judge
        // off in the tests) plus the judge's API-key refusal list and its own knobs.
        let mut keys = recall_vars();
        for k in [ABSTAIN_POLICY_ENV, "RECALL_JUDGE", "RECALL_JUDGE_CLAUDE", "RECALL_JUDGE_THRESHOLD", "RECALL_JUDGE_K"]
            .into_iter()
            .chain(crate::judge::API_KEY_VARS.iter().copied())
        {
            if !keys.iter().any(|x| x == k) {
                keys.push(k.into());
            }
        }
        let saved = keys.iter().map(|k| (k.clone(), std::env::var_os(k))).collect();
        for k in &keys {
            std::env::remove_var(k);
        }
        // The abstain tests below pin the judge's REFUSE-mode decisions (which queries abstain at all);
        // label mode (the default) returns those same queries labelled, tested in `label_policy_*`.
        std::env::set_var(ABSTAIN_POLICY_ENV, "refuse");
        EnvLock { _g: g, saved }
    }

    fn test_hit(id: i64, path: &str) -> Hit {
        Hit {
            id,
            path: path.to_string(),
            title: String::new(),
            snippet: String::new(),
            score: 0.0,
        }
    }

    /// Test fixture store.
    ///
    /// 🔴 Built through `Store::open_at(":memory:")` — the REAL schema — not a hand-rolled copy
    /// of it. The copy that used to live here carried its own warning that it drifts, and it
    /// did: it died once on `tier`, and again on `content_sha256`. A fixture schema that is a
    /// second implementation of the production schema will always drift, and every drift shows
    /// up as a test failure that looks like a bug in the change under review.
    pub(crate) fn in_memory_store() -> Store {
        Store::open_at(std::path::Path::new(":memory:")).unwrap()
    }

    #[test]
    fn graph_channel_returns_anchor_and_path_neighbors() {
        let store = in_memory_store();
        let seed = store
            .insert(
                "projects/acme/memory/spec.md",
                "Memory Spec",
                "alpha gateway anchor",
                "vault",
            )
            .unwrap();
        let neighbor = store
            .insert(
                "projects/acme/memory/architecture.md",
                "Memory Architecture",
                "adjacent memory architecture",
                "vault",
            )
            .unwrap();
        let unrelated = store
            .insert(
                "projects/acme/other/unrelated.md",
                "Unrelated",
                "adjacent but not same folder",
                "vault",
            )
            .unwrap();

        let mut id_hit = HashMap::new();
        let ids = channel_ids(&store, "alpha gateway", "graph", 5, &mut id_hit).unwrap();

        assert!(ids.contains(&seed));
        assert!(ids.contains(&neighbor));
        assert!(!ids.contains(&unrelated));
        assert!(id_hit.contains_key(&seed));
        assert!(id_hit.contains_key(&neighbor));
    }

    #[test]
    fn fused_fts5_graph_reuses_anchor_hits_and_adds_neighbors() {
        let store = in_memory_store();
        let seed = store
            .insert(
                "projects/acme/memory/spec.md",
                "Memory Spec",
                "alpha gateway anchor",
                "vault",
            )
            .unwrap();
        let neighbor = store
            .insert(
                "projects/acme/memory/architecture.md",
                "Memory Architecture",
                "adjacent memory architecture",
                "vault",
            )
            .unwrap();

        let hits = fused(
            &store,
            None,
            "alpha gateway",
            5,
            &["fts5".to_string(), "graph".to_string()],
        )
        .unwrap();
        let ids: Vec<i64> = hits.iter().map(|hit| hit.id).collect();

        assert!(ids.contains(&seed));
        assert!(ids.contains(&neighbor));
    }

    #[test]
    fn to_rel_strips_root_and_normalises_separators() {
        let root = Path::new(r"C:\Users\alice\Notes");
        let abs = r"C:\Users\alice\Notes\projects\recall\foo.md";
        assert_eq!(to_rel(root, abs), "projects/recall/foo.md");
    }

    #[test]
    fn to_rel_passes_through_path_outside_root() {
        // A match that isn't under the vault root keeps its (normalised) form so
        // the lookup simply misses rather than producing a bogus relative path.
        let root = Path::new(r"C:\Users\alice\Notes");
        let abs = r"D:\elsewhere\note.md";
        assert_eq!(to_rel(root, abs), "D:/elsewhere/note.md");
    }

    #[test]
    fn rrf_rewards_agreement() {
        // doc 1 ranked top by both channels should win over doc 2 (top in one).
        let a = Ranked {
            channel: "a".into(),
            ids: vec![1, 2, 3],
        };
        let b = Ranked {
            channel: "b".into(),
            ids: vec![1, 3, 2],
        };
        let merged = rrf_merge(&[a, b], 3);
        assert_eq!(merged[0], 1);
    }

    #[test]
    fn rrf_empty_channel_is_noop() {
        let a = Ranked {
            channel: "a".into(),
            ids: vec![5, 6],
        };
        let empty = Ranked {
            channel: "b".into(),
            ids: vec![],
        };
        let merged = rrf_merge(&[a, empty], 2);
        assert_eq!(merged, vec![5, 6]);
    }

    #[test]
    fn rrf_equal_scores_are_deterministic_by_doc_id() {
        // Docs 10 and 20 appear at the same rank in their respective channels,
        // so they receive identical RRF scores. The tie-break must order them
        // by doc id ascending (10 before 20) and must be stable across calls.
        let a = Ranked {
            channel: "a".into(),
            ids: vec![10],
        };
        let b = Ranked {
            channel: "b".into(),
            ids: vec![20],
        };
        let first = rrf_merge(&[a, b], 2);
        // Rebuild identically — HashMap iteration order may differ between calls.
        let a2 = Ranked {
            channel: "a".into(),
            ids: vec![10],
        };
        let b2 = Ranked {
            channel: "b".into(),
            ids: vec![20],
        };
        let second = rrf_merge(&[a2, b2], 2);
        // Both runs must agree: lower id comes first.
        assert_eq!(first, vec![10, 20], "first run: tie not broken by doc id");
        assert_eq!(second, vec![10, 20], "second run: tie not broken by doc id");
        assert_eq!(first, second, "tie-break must be stable across runs");
    }

    #[test]
    fn rrf_merge_with_k_preserves_agreement_bias() {
        let a = Ranked {
            channel: "a".into(),
            ids: vec![2, 1],
        };
        let b = Ranked {
            channel: "b".into(),
            ids: vec![3, 1],
        };
        let merged = rrf_merge_with_k(&[a, b], 3, 10.0);
        assert_eq!(merged[0], 1);
    }

    #[test]
    fn rerank_promotes_a_deep_candidate_that_actually_contains_the_query_terms() {
        // The whole justification for the reranker: 38% of the questions neither channel answers
        // at top-10 DO have their answer at rank 11-50. This is that case in miniature — doc 99
        // sits last in both channels and is the only one containing both query terms.
        let ranked = vec![
            Ranked { channel: "bge".into(), ids: (1..=20).collect() },
            Ranked { channel: "fts5".into(), ids: (1..=20).rev().collect() },
        ];
        let terms = rerank_query_terms("elsevier risc-v textbook");
        assert!(terms.contains(&"elsevier".to_string()));
        assert!(!terms.contains(&"the".to_string()), "stopwords must be dropped");

        let mut text: HashMap<i64, String> = (1..=20)
            .map(|i| (i, "unrelated filler prose about something else".to_string()))
            .collect();
        text.insert(20, "elsevier risc textbook patterson hennessy".to_string());

        let order = rerank_order(&ranked, &terms, &text, RERANK_COVERAGE_WEIGHT);
        let pos = order.iter().position(|&x| x == 20).unwrap();
        assert!(pos < 5, "the only matching doc must be promoted into the top 5, got rank {pos}");

        // Without the coverage term it is NOT promoted - which proves the coverage signal, not the
        // rank fusion, is what moved it. A test that passes either way would prove nothing.
        let flat = rerank_order(&ranked, &terms, &text, 0.0);
        let flat_pos = flat.iter().position(|&x| x == 20).unwrap();
        assert!(
            flat_pos > pos,
            "coverage weight 0 must NOT promote it (got {flat_pos} vs {pos})"
        );
    }

    #[test]
    fn rerank_ordering_is_deterministic_across_runs() {
        // `rerank_order` accumulates into a HashMap, whose iteration order is randomised per
        // process. Without the explicit id tiebreak, equal-scoring documents would come back in a
        // different order on every run and no measurement of this reranker would be reproducible.
        let ranked = vec![Ranked { channel: "bge".into(), ids: (1..=30).collect() }];
        let terms = rerank_query_terms("identical text everywhere");
        let text: HashMap<i64, String> = (1..=30)
            .map(|i| (i, "identical text everywhere".to_string()))
            .collect();
        let first = rerank_order(&ranked, &terms, &text, RERANK_COVERAGE_WEIGHT);
        for _ in 0..5 {
            assert_eq!(first, rerank_order(&ranked, &terms, &text, RERANK_COVERAGE_WEIGHT));
        }
    }

    #[test]
    fn rerank_coverage_counts_distinct_terms_not_occurrences() {
        // A document that repeats one term must not outrank one that contains several. Coverage
        // is "how much of the question is addressed", not "how loudly".
        let terms = rerank_query_terms("alpha beta gamma");
        assert_eq!(terms.len(), 3);
        let spammy = "alpha alpha alpha alpha alpha alpha alpha";
        let broad = "alpha beta gamma";
        assert!(rerank_coverage(&terms, broad) > rerank_coverage(&terms, spammy));
        assert!((rerank_coverage(&terms, broad) - 1.0).abs() < 1e-9);
        assert!((rerank_coverage(&terms, spammy) - 1.0 / 3.0).abs() < 1e-9);
        // An empty term list must not zero everything.
        assert_eq!(rerank_coverage(&[], "anything"), 1.0);
    }

    #[test]
    fn fusion_depth_keeps_a_wider_candidate_window_for_top_five() {
        let _guard = env_lock();
        std::env::remove_var(MIN_FUSION_DEPTH_ENV);
        std::env::remove_var(FUSION_DEPTH_MULTIPLIER_ENV);

        assert_eq!(fusion_depth(5), 80);
        assert_eq!(fusion_depth(10), 80);
    }

    #[test]
    fn fusion_depth_accepts_bounded_env_overrides_for_experiments() {
        let _guard = env_lock();
        std::env::set_var(MIN_FUSION_DEPTH_ENV, "40");
        std::env::set_var(FUSION_DEPTH_MULTIPLIER_ENV, "6");

        assert_eq!(effective_min_fusion_depth(), 40);
        assert_eq!(effective_fusion_depth_multiplier(), 6);
        assert_eq!(fusion_depth(5), 40);
        assert_eq!(fusion_depth(10), 60);

        std::env::remove_var(MIN_FUSION_DEPTH_ENV);
        std::env::remove_var(FUSION_DEPTH_MULTIPLIER_ENV);
    }

    #[test]
    fn fusion_depth_ignores_invalid_or_oversized_env_overrides() {
        let _guard = env_lock();
        std::env::set_var(MIN_FUSION_DEPTH_ENV, "0");
        std::env::set_var(FUSION_DEPTH_MULTIPLIER_ENV, "999");

        assert_eq!(effective_min_fusion_depth(), MIN_FUSION_DEPTH);
        assert_eq!(effective_fusion_depth_multiplier(), FUSION_DEPTH_MULTIPLIER);
        assert_eq!(fusion_depth(5), 80);

        std::env::remove_var(MIN_FUSION_DEPTH_ENV);
        std::env::remove_var(FUSION_DEPTH_MULTIPLIER_ENV);
    }

    #[test]
    fn denoise_mode_default_is_off() {
        let _guard = env_lock();
        std::env::remove_var(DENOISE_MODE_ENV);

        assert_eq!(effective_denoise_mode(), "off");
    }

    #[test]
    fn fusion_mode_can_anchor_strong_fts_prefix() {
        let _guard = env_lock();
        std::env::set_var(FUSION_MODE_ENV, "rrf2_fts3");
        std::env::remove_var(DENOISE_MODE_ENV);
        let fts = Ranked {
            channel: "fts5".into(),
            ids: vec![10, 11, 12, 13],
        };
        let bge = Ranked {
            channel: "bge".into(),
            ids: vec![20, 21, 22, 23],
        };

        let merged = rrf_merge(&[fts, bge], 5);

        std::env::remove_var(FUSION_MODE_ENV);
        assert!(merged.contains(&10));
        assert!(merged.contains(&11));
        assert!(merged.contains(&12));
        assert_eq!(merged.len(), 5);
    }

    #[test]
    fn fusion_mode_missing_anchor_channel_falls_back_to_base() {
        let _guard = env_lock();
        std::env::set_var(FUSION_MODE_ENV, "rrf2_fts3");
        std::env::remove_var(DENOISE_MODE_ENV);
        let bge = Ranked {
            channel: "bge".into(),
            ids: vec![20, 21, 22],
        };

        let merged = rrf_merge(&[bge], 3);

        std::env::remove_var(FUSION_MODE_ENV);
        assert_eq!(merged, vec![20, 21, 22]);
    }

    #[test]
    fn rrf_merge_with_hits_preserves_order_when_denoise_is_off() {
        let _guard = env_lock();
        std::env::set_var(DENOISE_MODE_ENV, "off");
        let ranked = vec![Ranked {
            channel: "fts5".into(),
            ids: vec![1, 2, 3],
        }];
        let mut id_hit = HashMap::new();
        id_hit.insert(1, test_hit(1, "logs/agent-activity.jsonl"));
        id_hit.insert(2, test_hit(2, "tasks/status-2026-01-02.md"));
        id_hit.insert(3, test_hit(3, "projects/a/architecture.md"));

        let merged = rrf_merge_with_hits(&ranked, 3, &id_hit);

        std::env::remove_var(DENOISE_MODE_ENV);
        assert_eq!(merged, vec![1, 2, 3]);
    }

    /// The removed path-heuristic modes now leave the fused order alone (and warn once).
    #[test]
    fn a_removed_denoise_mode_keeps_the_fused_order() {
        let _guard = env_lock();
        std::env::set_var(DENOISE_MODE_ENV, "ops_noise_path_terms_v2");
        let ranked = vec![Ranked { channel: "fts5".into(), ids: vec![1, 2, 3] }];
        let mut id_hit = HashMap::new();
        // eval/results/ was the old mode's heaviest demotion; it must now stay first
        id_hit.insert(1, test_hit(1, "eval/results/run.json"));
        id_hit.insert(2, test_hit(2, "docs/guide.md"));
        id_hit.insert(3, test_hit(3, "projects/a/architecture.md"));
        let merged = rrf_merge_with_hits(&ranked, 3, &id_hit);
        std::env::remove_var(DENOISE_MODE_ENV);
        assert_eq!(merged, vec![1, 2, 3]);
    }

    // ── G3: channel names are validated, never silently dropped ─────────────

    #[test]
    fn validate_channels_rejects_unknown_name() {
        let err = validate_channels(&["bgee".to_string()]).unwrap_err().to_string();
        assert!(err.contains("unknown channel 'bgee'"), "{err}");
        assert!(validate_channels(&[]).is_err(), "an empty channel list must not run");
    }

    #[test]
    fn validate_channels_accepts_all_compiled_aliases() {
        for c in [
            "fts5", "fts", "sqlite-fts", "rg", "ripgrep", "bge", "vector", "lancedb", "qwen3",
            "qwen", "graph", "surreal", " FTS5 ",
        ] {
            validate_channels(&[c.to_string()]).unwrap_or_else(|e| panic!("{c}: {e}"));
        }
    }

    #[test]
    fn validate_channels_rejects_uncompiled_gemini_and_voyage() {
        for (name, compiled) in [
            ("gemini", cfg!(feature = "gemini")),
            ("vertex", cfg!(feature = "gemini")),
            ("voyage", cfg!(feature = "voyage")),
            ("dense", cfg!(feature = "voyage")),
        ] {
            let r = validate_channels(&[name.to_string()]);
            if compiled {
                assert!(r.is_ok(), "{name} is compiled in and must be accepted");
            } else {
                let err = r.unwrap_err().to_string();
                assert!(err.contains("not compiled into this binary"), "{name}: {err}");
            }
        }
    }

    // ── G6: provenance ───────────────────────────────────────────────────────

    #[test]
    fn source_label_joins_channels_for_shared_doc() {
        let ranked = vec![
            Ranked { channel: "fts5".into(), ids: vec![1, 2] },
            Ranked { channel: "bge".into(), ids: vec![2, 3] },
        ];
        assert_eq!(source_label(1, &ranked), "fts5");
        assert_eq!(source_label(2, &ranked), "fts5+bge");
        assert_eq!(source_label(3, &ranked), "bge");
    }

    #[test]
    fn fused_sourced_labels_fts5_only_hit() {
        let _guard = env_lock();
        let store = in_memory_store();
        store
            .insert("notes/kestra.md", "Kestra", "kestra password rotation runbook", "vault")
            .unwrap();
        let out = fused_sourced(&store, None, None, "kestra password", 5, &["fts5".into()]).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().all(|(_, p)| p.source == "fts5"), "{out:?}");
    }

    #[test]
    fn fused_sourced_labels_rescue() {
        // bge requested but unavailable (None): the rescue tier answers, and SAYS so.
        let _guard = env_lock();
        let store = in_memory_store();
        store
            .insert("notes/kestra.md", "Kestra", "kestra password rotation runbook", "vault")
            .unwrap();
        let out = fused_sourced(&store, None, None, "kestra password", 5, &["bge".into()]).unwrap();
        assert!(!out.is_empty());
        assert!(out.iter().all(|(_, p)| p.source == "fts5-rescue"), "{out:?}");
        // T5: the rescue's raw score is keyed fts5-rescue and is that doc's own BM25.
        let bm25: HashMap<i64, f64> = store
            .fts_search("kestra password", fusion_depth(5))
            .unwrap()
            .iter()
            .map(|h| (h.id, h.score))
            .collect();
        for (h, p) in &out {
            assert_eq!(p.scores.get("fts5-rescue"), Some(&bm25[&h.id]));
            assert!(!p.scores.contains_key("bge"));
        }
    }

    // -- query cap split (backlog #8, PLAN.md T15, T17) --

    /// 150-token query: tokens 1-64 are `pre`, tokens 65-150 are suffix words held by decoys.
    fn long_q(pre: &str) -> String {
        let suffix: Vec<String> = (0..86).map(|i| format!("sfx{i}")).collect();
        let mut pre: Vec<String> = pre.split(' ').map(String::from).collect();
        let fill = 64 - pre.len();
        pre.extend((0..fill).map(|i| format!("fill{i}")));
        format!("{} {}", pre.join(" "), suffix.join(" "))
    }

    /// T15a/b: the judge sees only the 64-token prefix, in both directions, on every path.
    #[test]
    fn long_query_decisions_follow_the_judged_prefix() {
        let _guard = env_lock();
        let store = in_memory_store();
        let fills: Vec<String> = (0..64).map(|i| format!("fill{i}")).collect();
        store.insert("a.md", "Kestra", &format!("kestra password rotation {}", fills.join(" ")), "vault").unwrap();
        let sfx: Vec<String> = (0..86).map(|i| format!("sfx{i}")).collect();
        for i in 0..6 {
            store.insert(&format!("d{i}.md"), "decoy", &sfx.join(" "), "vault").unwrap();
        }
        let good = long_q("kestra password rotation");
        assert!(crate::store::exceeds_judge_cap(&good));
        assert_eq!(store.fts_judge_ids(&good, 5).unwrap(), vec![store.fts_search_judged(&good, 1).unwrap()[0].id]);
        for ch in [vec!["fts5".to_string()], vec!["bge".into()], vec!["qwen3".into()]] {
            assert!(!fused(&store, None, &good, 5, &ch).unwrap().is_empty(), "{ch:?}: supported prefix answers");
        }
        let bad = long_q("plimbex trondulate vasquor");
        // graph restored here (Codex v3 r1 HIGH #4): the fill words find a graph seed.
        for ch in [vec!["fts5".to_string()], vec!["bge".into()], vec!["qwen3".into()], vec!["graph".into()]] {
            assert!(fused(&store, None, &bad, 5, &ch).unwrap().is_empty(), "{ch:?}: unsupported prefix abstains");
        }
        // T15d, graph-only (builds no judge of its own): a prefix of 64 unknown tokens finds no
        // seed on the judged search, so the gated rescue runs and abstains (R2).
        let nonsense: Vec<String> = (0..64).map(|i| format!("qzx{i}")).collect();
        let sfx: Vec<String> = (0..86).map(|i| format!("sfx{i}")).collect();
        let bad = format!("{} {}", nonsense.join(" "), sfx.join(" "));
        assert!(fused(&store, None, &bad, 5, &["graph".into()]).unwrap().is_empty(), "graph-only");
    }

    /// T15c: a long query with a number constraint keeps the 64-token ranking (R1), so the doc
    /// holding 8784 is not pushed out by fusion_depth + 20 suffix decoys.
    #[test]
    fn long_constrained_query_keeps_its_answer_past_fusion_depth() {
        let _guard = env_lock();
        let store = in_memory_store();
        let fills: Vec<String> = (0..64).map(|i| format!("fill{i}")).collect();
        store.insert("a.md", "Daemon", &format!("daemon port 8784 runbook {}", fills.join(" ")), "vault").unwrap();
        filler(&store, &["daemon", "port", "runbook"]);
        let sfx: Vec<String> = (0..86).map(|i| format!("sfx{i}")).collect();
        for i in 0..(fusion_depth(5) + 20) {
            store.insert(&format!("d{i}.md"), "daemon port", &format!("daemon port 2219 {}", sfx.join(" ")), "vault").unwrap();
        }
        let q = long_q("daemon port 8784 runbook");
        for ch in [vec!["fts5".to_string()], vec!["fts5".into(), "bge".into()], vec!["bge".into()], vec!["qwen3".into()]] {
            let out = fused(&store, None, &q, 5, &ch).unwrap();
            assert_eq!(out.first().map(|h| h.path.as_str()), Some("a.md"), "{ch:?}: {out:?}");
        }
    }

    /// T17: for an unconstrained long query, scores.fts5 come from the 256-token ranking search.
    #[test]
    fn long_query_fts5_scores_come_from_the_ranking_search() {
        let _guard = env_lock();
        let store = in_memory_store();
        let fills: Vec<String> = (0..64).map(|i| format!("fill{i}")).collect();
        let sfx: Vec<String> = (0..86).map(|i| format!("sfx{i}")).collect();
        store.insert("a.md", "Kestra", &format!("kestra password rotation {} sfx3", fills.join(" ")), "vault").unwrap();
        store.insert("b.md", "B", &sfx.join(" "), "vault").unwrap();
        let q = long_q("kestra password rotation");
        let list = store.fts_search(&q, fusion_depth(5)).unwrap();
        let judged = store.fts_search_judged(&q, fusion_depth(5)).unwrap();
        let out = fused_sourced(&store, None, None, &q, 5, &["fts5".into()]).unwrap();
        assert!(out.iter().any(|(h, _)| h.path == "b.md"), "suffix-only doc is ranked: {out:?}");
        for (h, p) in &out {
            let pos = list.iter().position(|x| x.id == h.id).unwrap();
            assert_eq!(p.scores["fts5"], list[pos].score);
            assert_eq!(p.ranks["fts5"], pos);
        }
        let a = judged.iter().find(|h| h.path == "a.md").unwrap().score;
        let a256 = list.iter().find(|h| h.path == "a.md").unwrap().score;
        assert_ne!(a, a256, "fixture: the two searches score a.md differently");
    }

    /// Codex v3 r1 HIGH #3: the 64-token judge approves the Kestra doc, the 256-token ranking puts
    /// Banana first. A returned set without any supporting hit abstains; with room for both, the
    /// supported doc is returned.
    #[test]
    fn a_long_query_never_returns_only_hits_its_judge_did_not_approve() {
        let _guard = env_lock();
        std::env::remove_var(ABSTAIN_POLICY_ENV);
        let store = in_memory_store();
        store.insert("a.md", "A", "kestra password rotation", "vault").unwrap();
        store.insert("b.md", "B", "banana", "vault").unwrap();
        let pre: Vec<&str> = ["kestra", "password", "rotation"].iter().cycle().take(64).copied().collect();
        let q = format!("{} {}", pre.join(" "), vec!["banana"; 192].join(" "));
        assert!(crate::store::exceeds_judge_cap(&q));
        std::env::set_var(ABSTAIN_ENV, "off");
        let raw = fused(&store, None, &q, 1, &["fts5".into()]);
        std::env::remove_var(ABSTAIN_ENV);
        let raw = raw.unwrap();
        // Fixture check: if the wider ranking did not reverse the order, this test proves nothing.
        assert_eq!(raw.first().map(|h| h.path.as_str()), Some("b.md"), "fixture: {raw:?}");
        // 2026-11-06: the ranking is the same with abstain on; Banana alone comes back LABELLED (no
        // returned hit holds the judged terms), never as a confident answer, and nothing is swapped in.
        let one = fused_sourced(&store, None, None, &q, 1, &["fts5".into()]).unwrap();
        assert_eq!(one.iter().map(|(h, _)| h.path.as_str()).collect::<Vec<_>>(), vec!["b.md"]);
        assert!(one[0].1.closest_match, "{:?}", one[0].1);
        let five = fused(&store, None, &q, 5, &["fts5".into()]).unwrap();
        assert!(five.iter().any(|h| h.path == "a.md"), "{five:?}");
    }

    /// Codex v3 r1 HIGH #1 + MEDIUM #5: `top` is a hard limit even with abstain off and tracing on,
    /// and with the reranker the constraint filter replaces a dropped hit from the whole pool.
    #[test]
    fn top_is_a_hard_limit_and_rerank_keeps_the_constrained_answer() {
        let _guard = env_lock();
        let store = in_memory_store();
        for i in 0..2 {
            store.insert(&format!("w{i}.md"), "daemon port", "daemon port 2219 daemon port", "vault").unwrap();
        }
        let weak_pad: Vec<String> = (0..400).map(|i| format!("weak{i}")).collect();
        for i in 0..10 {
            store.insert(&format!("weak{i}.md"), "x", &format!("daemon {}", weak_pad.join(" ")), "vault").unwrap();
        }
        // A long runbook (BM25 length normalisation) among enough unrelated notes that "daemon"
        // and "port" carry IDF: the short wrong-port notes then outrank it.
        let pad: Vec<String> = (0..300).map(|i| format!("pad{i}")).collect();
        let body = format!("the recall daemon listens on port 8784 {}", pad.join(" "));
        store.insert("runbook.md", "Runbook", &body, "vault").unwrap();
        for i in 0..60 {
            store.insert(&format!("z{i}.md"), "other", &format!("zebra{i} unrelated note"), "vault").unwrap();
        }
        let q = "daemon port 8784";
        std::env::set_var(ABSTAIN_ENV, "off");
        std::env::set_var(ABSTAIN_TRACE_ENV, "1");
        let off = fused(&store, None, q, 5, &["fts5".into()]);
        std::env::remove_var(ABSTAIN_TRACE_ENV);
        std::env::remove_var(ABSTAIN_ENV);
        let off = off.unwrap();
        assert!(off.len() <= 5, "{} hits for --top 5", off.len());
        assert_ne!(off[0].path, "runbook.md", "fixture: a wrong-port note must rank first");
        // 2026-11-06: with the reranker on, --top 1 is one hit; if it is a wrong-port note it SAYS so.
        std::env::set_var(RERANK_ENV, "1");
        std::env::set_var(RERANK_WEIGHT_ENV, "0");
        let one = fused_sourced(&store, None, None, q, 1, &["fts5".into()]);
        std::env::remove_var(RERANK_WEIGHT_ENV);
        std::env::remove_var(RERANK_ENV);
        let one = one.unwrap();
        assert_eq!(one.len(), 1);
        assert!(one[0].0.path == "runbook.md" || one[0].1.lacks.contains(&"8784".to_string()), "{:?}", one[0]);
    }

    /// Codex v3 r1 HIGH #4: graph-only with invented words plus searchable ones finds a seed;
    /// it must still answer to the lexical rule.
    #[test]
    fn graph_only_answers_to_the_lexical_rule() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("k.md", "Kestra", "kestra password rotation", "vault").unwrap();
        filler(&store, &["kestra", "password"]);
        let g = vec!["graph".to_string()];
        assert!(fused(&store, None, "plimbex trondulate kestra password", 5, &g).unwrap().is_empty());
        assert!(!fused(&store, None, "kestra password", 5, &g).unwrap().is_empty());
    }

    // -- provenance (backlog #21, PLAN.md T1-T6) --

    #[test]
    fn provenance_ranks_scores_and_rrf() {
        use std::collections::BTreeMap;
        let ranked = vec![
            Ranked { channel: "fts5".into(), ids: vec![1, 2] },
            Ranked { channel: "bge".into(), ids: vec![2, 3] },
        ];
        let mut raw: HashMap<String, HashMap<i64, f64>> = HashMap::new();
        raw.insert("fts5".into(), HashMap::from([(1, 10.0), (2, 5.0)]));
        raw.insert("bge".into(), HashMap::from([(2, 0.8), (3, 0.7)]));
        let p = provenance(2, &ranked, &raw, 60.0);
        assert_eq!(p.source, "fts5+bge");
        assert_eq!(p.ranks, BTreeMap::from([("bge".to_string(), 0), ("fts5".to_string(), 1)]));
        assert_eq!(p.scores, BTreeMap::from([("bge".to_string(), 0.8), ("fts5".to_string(), 5.0)]));
        assert!((p.rrf - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
        let p3 = provenance(3, &ranked, &raw, 60.0);
        assert_eq!(p3.scores, BTreeMap::from([("bge".to_string(), 0.7)]));
        assert_eq!(p3.ranks, BTreeMap::from([("bge".to_string(), 1)]));
    }

    /// v3 r2: two rows for one file (a case-only rename) take one result slot, not two, and the
    /// freed slot goes to the next document; the kept hit's fts5 score is a real BM25.
    #[test]
    fn a_case_only_duplicate_path_takes_one_slot() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("lib/agents.md", "Agents", "openrouter printing press agents", "vault").unwrap();
        store.insert("lib/AGENTS.md", "Agents", "openrouter printing press agents", "vault").unwrap();
        store.insert("lib\\Agents.md", "Agents", "openrouter printing press agents", "vault").unwrap();
        store.insert("other.md", "Other", "openrouter printing press notes", "vault").unwrap();
        let q = "openrouter printing press";
        let out = fused_sourced(&store, None, None, q, 2, &["fts5".into()]).unwrap();
        let paths: Vec<String> = out.iter().map(|(h, _)| h.path.to_lowercase().replace('\\', "/")).collect();
        assert_eq!(paths.len(), 2, "{paths:?}");
        assert_ne!(paths[0], paths[1], "one file, one slot: {paths:?}");
        assert!(paths.contains(&"other.md".to_string()), "the freed slot is refilled: {paths:?}");
        for (_, p) in &out {
            assert!(p.scores.get("fts5").is_some_and(|s| s.is_finite() && *s != 0.0), "{p:?}");
        }
        // Unit level: the kept id takes the first variant's position and score in every channel.
        let mut ranked = vec![
            Ranked { channel: "fts5".into(), ids: vec![1, 2, 3] },
            Ranked { channel: "bge".into(), ids: vec![2, 3] },
        ];
        let id_hit: HashMap<i64, Hit> =
            [(1, "a/X.md"), (2, "a/x.md"), (3, "b.md")].into_iter().map(|(i, p)| (i, test_hit(i, p))).collect();
        let mut raw: HashMap<String, HashMap<i64, f64>> = HashMap::new();
        raw.insert("bge".into(), HashMap::from([(2, 0.9), (3, 0.5)]));
        let merged = collapse_case_duplicates(&mut ranked, &mut raw, &id_hit, &|_| false);
        assert_eq!(ranked[0].ids, vec![1, 3]);
        assert_eq!(ranked[1].ids, vec![1, 3]);
        assert_eq!(raw["bge"].get(&1), Some(&0.9));
        assert_eq!(merged.get(&1), Some(&vec![2]), "the lent score is disclosed");
    }

    #[test]
    fn provenance_rank_without_raw_score_is_absent_not_zero() {
        let ranked = vec![Ranked { channel: "rg".into(), ids: vec![9] }];
        let mut raw: HashMap<String, HashMap<i64, f64>> = HashMap::new();
        raw.insert("fts5".into(), HashMap::from([(9, f64::NAN)]));
        let p = provenance(9, &ranked, &raw, 60.0);
        assert_eq!(p.ranks.get("rg"), Some(&0));
        assert!(p.scores.is_empty(), "{p:?}");
        // T3: rank keys == the channels named in source.
        let keys: Vec<&str> = p.ranks.keys().map(String::as_str).collect();
        assert_eq!(keys, p.source.split('+').collect::<Vec<_>>());
    }

    /// T4 + T6: every fts5 hit's raw score is ITS OWN BM25 (looked up by id), its rank is its
    /// index in the fts5 list, and the legacy `score` is unchanged.
    #[test]
    fn fused_sourced_fts5_scores_are_the_docs_own_bm25() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("a.md", "kestra password", "kestra password rotation", "vault").unwrap();
        store.insert("b.md", "B", "kestra password kestra password", "vault").unwrap();
        store.insert("c.md", "C", "kestra password and many more unrelated words here", "vault").unwrap();
        store.insert("d.md", "D", "password kestra notes", "vault").unwrap();
        let q = "kestra password";
        let list = store.fts_search(q, fusion_depth(5)).unwrap();
        let out = fused_sourced(&store, None, None, q, 5, &["fts5".into()]).unwrap();
        assert_eq!(out.len(), 4);
        for (h, p) in &out {
            let pos = list.iter().position(|x| x.id == h.id).unwrap();
            assert_eq!(p.scores.get("fts5"), Some(&list[pos].score), "{}", h.path);
            assert_eq!(p.ranks.get("fts5"), Some(&pos));
            assert_eq!(h.score, list[pos].score, "legacy score unchanged");
            assert!((p.rrf - 1.0 / (effective_rrf_k() + pos as f64 + 1.0)).abs() < 1e-12);
        }
        let distinct: std::collections::HashSet<u64> = out.iter().map(|(h, _)| h.score.to_bits()).collect();
        assert!(distinct.len() > 1, "fixture must give distinct BM25 values");
    }

    // ── G2: abstain ──────────────────────────────────────────────────────────

    #[test]
    fn abstained_channel_does_not_trigger_rescue() {
        let ranked = vec![Ranked { channel: "bge".into(), ids: vec![] }];
        assert!(!rescue_needed(&ranked, true, false));
    }

    #[test]
    fn failed_channel_still_triggers_rescue() {
        let ranked = vec![Ranked { channel: "bge".into(), ids: vec![] }];
        assert!(rescue_needed(&ranked, false, false));
        assert!(!rescue_needed(&ranked, false, true), "fts5 itself was asked for: no rescue");
    }

    #[test]
    fn weighted_share_counts_the_weight_mass_held() {
        assert_eq!(weighted_share(&[true, false], &[3.0, 1.0]), 0.75);
        assert_eq!(weighted_share(&[], &[]), 0.0, "no content words = nothing to support");
    }

    #[test]
    fn lexical_support_uses_the_index_stemmer_and_whole_terms() {
        let _guard = env_lock();
        let store = in_memory_store();
        let a = store.insert("a.md", "A", "rotating passwords weekly", "vault").unwrap();
        let b = store.insert("b.md", "B", "syntax of a honeymoon", "vault").unwrap();
        let (terms, weights, zero_df) = query_term_weights(&store, "password tax").unwrap();
        assert_eq!(zero_df, 1, "'tax' inside 'syntax' is not an occurrence of the term");
        let sa = lexical_support(&store, &[a], &terms, &vec![1.0; 2]).unwrap().unwrap();
        let sb = lexical_support(&store, &[b], &terms, &vec![1.0; 2]).unwrap().unwrap();
        assert_eq!(sa, 0.5, "'passwords' supports 'password' through the porter stemmer");
        assert_eq!(sb, 0.0, "'syntax' must not support 'tax'");
        let best = lexical_support(&store, &[b, a], &terms, &weights).unwrap().unwrap();
        assert!(best > 0.0, "support is the best doc of the set");
    }

    #[test]
    fn fts5_abstains_when_top_hit_covers_few_query_terms() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("shots/ocr.md", "OCR", "port 7741 screenshot", "vault").unwrap();
        filler(&store, &["nonexistent", "term"]);
        let out = fused_sourced(
            &store,
            None,
            None,
            "zzqx-nonexistent-term-7741",
            5,
            &["fts5".into()],
        )
        .unwrap();
        assert!(out.is_empty(), "1 of 4 query terms present must abstain: {out:?}");
    }

    #[test]
    fn fts5_does_not_abstain_on_full_coverage() {
        let _guard = env_lock();
        let store = in_memory_store();
        store
            .insert("notes/kestra.md", "Kestra", "kestra password rotation runbook", "vault")
            .unwrap();
        let out = fused(&store, None, "kestra password", 5, &["fts5".into()]).unwrap();
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn fts5_abstains_on_termless_query() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("notes/u.md", "Unicorn", "unicorn \u{1F984} emoji", "vault").unwrap();
        let out = fused(&store, None, "\u{1F984}", 5, &["fts5".into()]).unwrap();
        assert!(out.is_empty(), "{out:?}");
    }

    #[test]
    fn abstain_can_be_switched_off_for_calibration() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("shots/ocr.md", "OCR", "port 7741 screenshot", "vault").unwrap();
        std::env::set_var(ABSTAIN_ENV, "off");
        let out = fused(&store, None, "zzqx-nonexistent-term-7741", 5, &["fts5".into()]);
        std::env::remove_var(ABSTAIN_ENV);
        assert_eq!(out.unwrap().len(), 1, "RECALL_ABSTAIN=off must return the raw fts5 hit");
    }

    // ── G2 round 2: the abstain rules must not throw away legitimate answers ────

    #[test]
    fn fts5_required_terms_is_capped_at_the_gold_minimum() {
        assert_eq!(fts5_required_terms(9), 3, "v5-012: 3 of 9 must be enough");
        assert_eq!(fts5_required_terms(6), 3, "purple elephant: 2 of 6 is not");
        assert_eq!(fts5_required_terms(5), 4, "five findable terms: all but one (v3 r1)");
        assert_eq!(fts5_required_terms(4), 4, "gmail password rotation runbook: all four (v3 r1)");
        assert_eq!(fts5_required_terms(3), 3);
        assert_eq!(fts5_required_terms(2), 2, "two findable terms: both");
        assert_eq!(fts5_required_terms(1), 1, "\"kestra passwrod\": the typo is not findable");
        assert_eq!(fts5_required_terms(0), 1, "no terms can never be confident");
        assert_eq!(fts5_required_terms(40), FTS5_MATCH_CAP);
    }

    #[test]
    fn confidence_terms_match_what_fts5_searched() {
        assert_eq!(confidence_terms("kestra password"), vec!["kestra", "password"]);
        assert_eq!(confidence_terms("AI"), vec!["ai"], "short query keeps its only token");
        assert_eq!(confidence_terms("Go"), vec!["go"]);
        assert_eq!(confidence_terms("v1.2"), vec!["v1"], "'2' is below FTS5's 2-char floor");
        assert!(confidence_terms("\u{1F984}").is_empty(), "a lone emoji has no token");
        let capped = format!("{} qxzvbnwplkjh", vec!["kestra"; 64].join(" "));
        assert_eq!(
            confidence_terms(&capped),
            vec!["kestra"],
            "the 65th token is past the FTS cap: it was never searched, so it cannot veto"
        );
        let many: String = (0..200).map(|i| format!("word{i} ")).collect();
        assert!(confidence_terms(&many).len() <= 64, "work is bounded by the FTS cap");
    }

    /// One document per word, so each word is findable in the corpus (as every real English word
    /// is in the live vault) without lifting any single document's coverage.
    fn filler(store: &Store, words: &[&str]) {
        for w in words {
            store.insert(&format!("filler/{w}.md"), w, w, "vault").unwrap();
        }
    }

    #[test]
    fn short_queries_need_every_findable_term() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("k.md", "Kestra", "kestra password rotation runbook", "vault").unwrap();
        filler(&store, &["banana", "telescope", "symphony"]);
        assert_eq!(fts5_only(&store, "kestra password").len(), 1);
        assert_eq!(fts5_only(&store, "kestra passwrod").len(), 1, "the typo is not findable");
        assert!(
            never_confident(&store, "banana telescope symphony"),
            "1 of 3 findable terms in the top hit"
        );
        assert!(never_confident(&store, "kestra banana"), "no document holds both");
    }

    /// abstain r9: a store too small to calibrate labels a failed query instead of refusing it, so
    /// the r8 refusal cases below now hold as "refused, or every hit an explicit closest match" —
    /// never a confident answer (K8).
    fn never_confident(store: &Store, q: &str) -> bool {
        fts5_sourced(store, q).iter().all(|(_, p)| p.closest_match)
    }

    fn fts5_only(store: &Store, q: &str) -> Vec<Hit> {
        fused(store, None, q, 5, &["fts5".into()]).unwrap()
    }

    #[test]
    fn fts5_answers_short_queries() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("AI_SETUP_GUIDE.md", "AI setup", "AI setup guide for Go v1.2", "vault").unwrap();
        for q in ["AI", "Go", "v1.2"] {
            assert_eq!(fts5_only(&store, q).len(), 1, "{q} must not abstain");
        }
    }

    #[test]
    fn fts5_tolerates_a_typo_and_the_token_cap() {
        let _guard = env_lock();
        let store = in_memory_store();
        store
            .insert("tasks/kestra.md", "Kestra", "kestra password rotation runbook", "vault")
            .unwrap();
        assert_eq!(fts5_only(&store, "kestra passwrod").len(), 1, "a zero-df typo is no veto");
        // v3 r8 (Codex r7 MEDIUM): label subjects are de-duplicated BEFORE the 64 cap, so an
        // unknown word behind 64 repeats of one word is the 2nd subject, not the 65th token. It is
        // unknown, so the query abstains; the ranking cap itself is unchanged.
        let capped = format!("{} qxzvbnwplkjh", vec!["kestra"; 64].join(" "));
        assert!(fts5_only(&store, &capped).is_empty());
        let known = format!("{} password", vec!["kestra"; 64].join(" "));
        assert_eq!(fts5_only(&store, &known).len(), 1);
    }

    /// lme-improve R2 store: one answer note plus unrelated fillers, so every word has a tiny df.
    fn small_memory_store() -> Store {
        let store = in_memory_store();
        store
            .insert("s1.md", "s1", "I adopted a cat named luna and the kestra upgrade cost 400 dollars", "vault")
            .unwrap();
        filler(&store, &["banana", "telescope", "symphony", "harbor"]);
        store
    }

    fn fts5_sourced(store: &Store, q: &str) -> Vec<(Hit, Provenance)> {
        fused_sourced(store, None, None, q, 5, &["fts5".into()]).unwrap()
    }

    #[test]
    fn a_failed_question_with_real_evidence_is_a_closest_match() {
        let _guard = env_lock();
        let store = small_memory_store();
        let hits = fts5_sourced(&store, "What is the name of my zorblat cat luna?");
        assert_eq!(hits.len(), 1, "answered as a closest match, not silence");
        assert!(hits[0].1.closest_match, "the state is explicit, never implied by lacks");
        assert!(hits[0].1.lacks.contains(&"zorblat".to_string()), "{:?}", hits[0].1.lacks);
        // keyword form keeps the old rule: an unknown word abstains
        assert!(fts5_only(&store, "zorblat cat luna").is_empty());
    }

    #[test]
    fn a_confident_answer_is_not_a_closest_match() {
        let _guard = env_lock();
        let store = small_memory_store();
        let hits = fts5_sourced(&store, "What is the name of my cat luna?");
        assert_eq!(hits.len(), 1);
        assert!(!hits[0].1.closest_match);
    }

    #[test]
    fn a_question_mostly_made_of_unknown_words_still_abstains() {
        let _guard = env_lock();
        let store = small_memory_store();
        assert!(fts5_only(&store, "What did zorblat quexim vandrel say about luna?").is_empty());
    }

    #[test]
    fn a_failed_question_with_little_evidence_mass_still_abstains() {
        let _guard = env_lock();
        let store = small_memory_store();
        // one matched word (~7.4 nats) < CLOSEST_MIN_MASS
        assert!(fts5_only(&store, "What did the zorblat do with luna?").is_empty());
    }

    #[test]
    fn a_closest_match_names_an_unmatched_typo_instead_of_dropping_the_hit() {
        let _guard = env_lock();
        let store = small_memory_store();
        filler(&store, &["pasted"]); // "passed" is one edit from it, so it reads as a typo
        let hits = fts5_sourced(&store, "How much did the kestra upgrade cost before it passed?");
        let s1 = hits.iter().find(|(h, _)| h.path == "s1.md").expect("not filtered away");
        assert!(s1.1.closest_match && s1.1.lacks.contains(&"passed".to_string()), "{:?}", s1.1);
    }

    /// Codex lme r2 HIGH: the confident judge is unchanged. A keyword query whose best note lacks a
    /// rare GENERIC word ("without") still abstains in a tiny store; it is never answered unlabelled.
    #[test]
    fn a_missing_rare_generic_word_still_refuses_a_keyword_query() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("a.md", "a", "kestra upgrade downtime runbook. Downtime is mandatory.", "vault").unwrap();
        store.insert("b.md", "b", "without", "vault").unwrap();
        assert!(never_confident(&store, "kestra upgrade without downtime runbook"));
    }

    #[test]
    fn a_closest_match_keeps_the_number_filter() {
        let _guard = env_lock();
        let store = small_memory_store();
        // feat/abstain-calibrated: 2024 is in NO note, so no hit could ever satisfy it; it is
        // labelled on the closest match instead of emptying the list (it used to return nothing).
        let h = fts5_sourced(&store, "What did the kestra upgrade cost in 2024 for zorblat?");
        assert!(h.len() == 1 && h[0].1.closest_match && h[0].1.lacks.contains(&"2024".to_string()), "{h:?}");
        assert_eq!(fts5_only(&store, "What did the kestra upgrade cost in 400 for zorblat?").len(), 1);
        // 2026-11-06 (abstain never drops a hit): a number SOME note holds is no longer a filter either;
        // the kestra note, which lacks 1999, comes back LABELLED with it.
        filler(&store, &["1999"]);
        let h = fts5_sourced(&store, "What did the kestra upgrade cost in 1999 for zorblat?");
        let s1 = h.iter().find(|(x, _)| x.path == "s1.md").expect("returned, not dropped");
        assert!(s1.1.lacks.contains(&"1999".to_string()), "{:?}", s1.1);
    }

    #[test]
    fn fts5_separates_v5_012_from_purple_elephant() {
        let _guard = env_lock();
        let store = in_memory_store();
        store
            .insert("pe/agent-fitness-report.md", "Report", "the agent fitness formula", "vault")
            .unwrap();
        store.insert("misc/moon.md", "Moon", "moon tax receipts", "vault").unwrap();
        filler(&store, &["purple", "elephant", "cheese", "9931"]);
        // Every word of the question exists in the live vault; mirror that here.
        // v3 r6: "PE" is a short subject; one in no document would abstain. The live vault holds it.
        filler(&store, &["weighting", "use", "across", "three", "component", "scores", "pe"]);
        // Live (loop2 r3 abstain-json, cal-all): fts5 hits 3-5 hold 9/9 and 8/9 of the terms, so
        // the 9-term query clears the long-query share floor and the 3-of-9 answer still returns.
        store
            .insert(
                "pe/scoring-overview.md",
                "Scoring overview",
                "overview: weighting of agent fitness formula use across three component scores",
                "vault",
            )
            .unwrap();
        let q = "What weighting does the PE agent fitness formula use across its three component scores?";
        let out = fts5_only(&store, q);
        assert!(
            out.iter().any(|h| h.path == "pe/agent-fitness-report.md"),
            "the 3-of-9 answer still comes back: {out:?}"
        );
        assert!(
            never_confident(&store, "purple elephant tax on moon cheese 9931"),
            "2 of 6 terms must abstain"
        );
    }

    // ── G2 round 3: inflection padding, acronym+typo, zero-df weight ──────────

    #[test]
    fn dedup_stems_counts_one_porter_stem_once() {
        let t = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            dedup_stems(t(&["cheese", "cheeses", "moon", "moons"])).unwrap(),
            t(&["cheese", "moon"])
        );
        assert_eq!(dedup_stems(t(&["kestra", "password"])).unwrap(), t(&["kestra", "password"]));
    }

    #[test]
    fn an_inflected_duplicate_cannot_buy_a_nonsense_query_a_match() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("shots/ocr.md", "OCR", "moon cheese cheeses receipts", "vault").unwrap();
        filler(&store, &["purple", "elephant", "tax", "9931"]);
        for q in [
            "purple elephant tax on moon cheese 9931",
            "purple elephant tax on moon cheese cheeses 9931",
            "purple elephant tax on moons cheese cheeses 9931",
        ] {
            assert!(never_confident(&store, q), "{q}: still 2 distinct stems of 6");
        }
    }

    #[test]
    fn acronym_plus_typo_answers_through_the_acronym() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("AI_SETUP_GUIDE.md", "AI setup", "AI setup guide for Go v1.2", "vault").unwrap();
        for q in ["AI setpu", "Go setpu", "AI setup"] {
            assert_eq!(fts5_only(&store, q).len(), 1, "{q} must not abstain");
        }
        assert_eq!(judged_terms(&store, "AI setpu").unwrap(), vec!["ai", "setpu"]);
        assert_eq!(judged_terms(&store, "AI setup").unwrap(), vec!["setup"], "content word present");
        // v3 r6: "AI" is still a short subject: labelled when a hit lacks it, never counted.
        assert_eq!(Judge::build(&store, "AI setup").unwrap().short, vec!["ai"]);
    }

    #[test]
    fn a_term_no_document_holds_weighs_nothing() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("k.md", "Kestra", "kestra password rotation", "vault").unwrap();
        let (terms, weights, zero_df) = query_term_weights(&store, "kestra passwrod").unwrap();
        assert_eq!(terms, vec!["kestra", "passwrod"]);
        assert_eq!(zero_df, 1);
        assert_eq!(weights[1], 0.0, "a typo cannot outweigh the real term");
        assert!(weights[0] > 0.0);
    }

    #[test]
    fn doc_freq_counts_only_searchable_documents() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("live.md", "Live", "kestra notes", "vault").unwrap();
        let gone = store.insert("gone.md", "Gone", "vanishedterm notes", "vault").unwrap();
        assert_eq!(store.term_searchable_df("vanishedterm", 100).unwrap(), 1);
        store.supersede_id(gone).unwrap();
        assert_eq!(
            store.term_searchable_df("vanishedterm", 100).unwrap(),
            0,
            "tombstoned text must not count toward df"
        );
        assert_eq!(store.term_searchable_df("notes", 100).unwrap(), 1);
    }

    #[test]
    fn bge_query_text_adds_the_baai_instruction_unless_switched_off() {
        let _guard = env_lock();
        std::env::remove_var("RECALL_BGE_QUERY_INSTRUCTION");
        let q = "Which port does the kestra daemon use?";
        assert_eq!(
            crate::bge::bge_query_text(q),
            format!("Represent this sentence for searching relevant passages: {q}")
        );
        // v3 r3: a statement embeds raw (BEIR SciFact claims lost 0.014 nDCG@10); keyword
        // queries keep the instruction.
        let claim = "0-dimensional biomaterials show inductive properties.";
        assert_eq!(crate::bge::bge_query_text(claim), claim);
        assert_eq!(
            crate::bge::bge_query_text("kestra password"),
            "Represent this sentence for searching relevant passages: kestra password"
        );
        std::env::set_var("RECALL_BGE_QUERY_INSTRUCTION", "always");
        let forced = crate::bge::bge_query_text(claim);
        std::env::set_var("RECALL_BGE_QUERY_INSTRUCTION", "off");
        let raw = crate::bge::bge_query_text(q);
        std::env::remove_var("RECALL_BGE_QUERY_INSTRUCTION");
        assert_eq!(forced, format!("Represent this sentence for searching relevant passages: {claim}"));
        assert_eq!(raw, q);
    }

    // ── loop2: Codex round-3 issues ──────────────────────────────────────────

    #[test]
    fn a_weaker_first_hit_cannot_veto_the_complete_answer_below_it() {
        let _guard = env_lock();
        let store = in_memory_store();
        let short = store.insert("k.md", "Kestra", "kestra", "vault").unwrap();
        let full = store
            .insert("runbook.md", "Runbook", "kestra password rotation runbook", "vault")
            .unwrap();
        filler(&store, &["rotation"]);
        // fts5 order as Codex reproduced it: the one-term note first, the full answer second.
        let v = lexical_rule(&store, "kestra password", &[short, full]).unwrap();
        assert_eq!(v, None, "the runbook at rank 2 holds both terms");
        let v = lexical_rule(&store, "kestra password", &[short]).unwrap();
        assert!(v.is_some(), "with only the partial note there is no confident answer");
    }

    #[test]
    fn excluded_documents_cannot_veto_a_query() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("k.md", "Kestra", "kestra password rotation", "vault").unwrap();
        assert_eq!(fts5_only(&store, "kestra passwrod").len(), 1);
        // Text fts5 can never return: a path-excluded doc and a SKILLS-tier doc.
        store.insert("disabled-loader-roots/x.md", "X", "passwrod", "vault").unwrap();
        store.insert("skills/x/SKILL.md", "S", "passwrod", "vault").unwrap();
        assert!(!store.term_searchable("passwrod").unwrap());
        assert_eq!(
            fts5_only(&store, "kestra passwrod").len(),
            1,
            "excluded text must not make the typo 'findable' and veto the answer"
        );
        let (_, w, zero) = query_term_weights(&store, "kestra passwrod").unwrap();
        assert_eq!((zero, w[1]), (1, 0.0), "excluded text carries no weight either");
    }

    #[test]
    fn numbers_alone_are_never_evidence() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("ports.md", "Ports", "daemon port 2219 and 8784", "vault").unwrap();
        for q in ["plimbex trondulate vasquor 2219", "2219", "2219 8784"] {
            assert!(fts5_only(&store, q).is_empty(), "{q}: a number match is not an answer");
        }
        assert_eq!(fts5_only(&store, "daemon port 2219").len(), 1, "real words + a number answer");
        assert!(is_number_token("2219") && !is_number_token("kmg8") && !is_number_token("v1"));
    }

    #[test]
    fn an_absent_non_typo_term_abstains_but_a_typo_does_not() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("p.md", "Plan", "project rollout plan for kestra password", "vault").unwrap();
        assert!(
            fts5_only(&store, "Project Zenthora rollout plan").is_empty(),
            "a project the vault has never heard of must not get a confident answer"
        );
        assert!(fts5_only(&store, "c2efb086dfede1589caba552").is_empty(), "absent hash");
        assert_eq!(fts5_only(&store, "project rollout plan").len(), 1);
        assert_eq!(fts5_only(&store, "kestra passwrod").len(), 1, "one edit from 'password'");
        assert!(edit1_candidates("passwrod").contains(&"password".to_string()));
        assert!(edit1_candidates("setpu").contains(&"setup".to_string()));
        assert!(!is_plausible_typo(&store, "zent").unwrap(), "too short to excuse");
    }

    #[test]
    fn bge_abstain_decision_covers_every_combination() {
        let _guard = env_lock();
        let store = in_memory_store();
        let good = store.insert("k.md", "Kestra", "kestra password rotation", "vault").unwrap();
        let bad = store.insert("m.md", "Moon", "moon cheese", "vault").unwrap();
        // Partial overlap: holds the common term, not the requested entity (Codex loop2-r1 #4).
        let decoy = store.insert("g.md", "Gmail", "gmail password reset", "vault").unwrap();
        let judge = Judge::build(&store, "kestra password").unwrap();
        // bge's top 5 holds the answer -> no bge veto.
        let (s, v) = bge_rule(&store, &judge, &[(bad, 0.9), (good, 0.8)], 0.19).unwrap();
        assert_eq!((s, v), (1.0, None));
        // bge's top 5 holds none of the query -> bge vetoes, whatever its cosine.
        let (s, v) = bge_rule(&store, &judge, &[(bad, 0.99)], 0.19).unwrap();
        assert_eq!(s, 0.0);
        assert!(v.is_some());
        // Five password-only decoys, the answer at bge rank 6: each decoy alone would score a
        // weighted share well above 0.19, but none supports the query, so bge vetoes.
        let far: Vec<(i64, f32)> = std::iter::repeat_n((decoy, 0.9), 5).chain([(good, 0.5)]).collect();
        let ev = judge.evidence(&store, decoy).unwrap();
        assert!(weighted_share(&ev.held, &judge.weights) > 0.19, "the decoy's raw share passes");
        assert!(!judge.supports(&ev));
        let (s, v) = bge_rule(&store, &judge, &far, 0.19).unwrap();
        assert_eq!(s, 0.0);
        assert!(v.is_some(), "partial-overlap decoys cannot lend support");
        // Same decoys with the answer inside the top 5 -> answers.
        let near: Vec<(i64, f32)> = std::iter::repeat_n((decoy, 0.9), 4).chain([(good, 0.5)]).collect();
        assert_eq!(bge_rule(&store, &judge, &near, 0.19).unwrap().1, None);
        assert_eq!(bge_verdict(0.19, 0.19), None, "the floor itself passes");
        assert!(bge_verdict(0.1899, 0.19).is_some());
    }

    /// The four lexical x bge combinations, through `query_reasons` — the exact function
    /// `fused_sourced` decides with — over REAL evidence from a store (Codex loop2-r1 #8).
    #[test]
    fn query_reasons_is_a_conjunction_over_real_evidence() {
        let _guard = env_lock();
        let store = in_memory_store();
        let good = store.insert("k.md", "Kestra", "kestra password rotation", "vault").unwrap();
        let bad = store.insert("m.md", "Moon", "moon cheese", "vault").unwrap();
        let j = Judge::build(&store, "kestra password").unwrap();
        let ok = j.evidences(&store, &[good]).unwrap();
        let no = j.evidences(&store, &[bad]).unwrap();
        for (fts, bge, want) in [(&ok, &ok, 0), (&ok, &no, 1), (&no, &ok, 1), (&no, &no, 2)] {
            let r = query_reasons(&j, Some(fts), Some(bge), 0.19);
            assert_eq!(r.len(), want, "{r:?}");
            assert_eq!(query_abstains(true, &r), want > 0);
            assert!(!query_abstains(false, &r), "RECALL_ABSTAIN=off never abstains");
        }
        // A rule that did not run cannot fail.
        assert!(query_reasons(&j, Some(&ok), None, 0.19).is_empty());
        assert_eq!(query_reasons(&j, None, Some(&no), 0.19).len(), 1);
    }

    #[test]
    fn another_channel_cannot_buy_back_a_rejected_query() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("shots/ocr.md", "OCR", "port 7741 screenshot", "vault").unwrap();
        store.insert("shots/other.md", "Other", "screenshot of port", "vault").unwrap();
        filler(&store, &["nonexistent", "term"]);
        let q = "zzqx-nonexistent-term-7741";
        // Precondition (Codex loop2-r1 #8): fts5 and graph DO return candidates for this query,
        // so an empty answer below is the abstention, not two empty channels.
        let seeds = store.fts_search(q, 80).unwrap();
        assert!(!seeds.is_empty(), "fts5 must have candidates");
        let mut id_hit = HashMap::new();
        assert!(!graph_ids_from_seeds(&store, &seeds, 80, &mut id_hit).unwrap().is_empty());
        std::env::set_var(ABSTAIN_ENV, "off");
        let unabstained = fused(&store, None, q, 5, &["fts5".into(), "graph".into()]).unwrap();
        std::env::remove_var(ABSTAIN_ENV);
        assert!(!unabstained.is_empty(), "without abstain, fts5+graph answer");
        for ch in [vec!["fts5", "graph"], vec!["fts5", "rg"]] {
            let ch: Vec<String> = ch.into_iter().map(String::from).collect();
            let out = fused(&store, None, q, 5, &ch).unwrap();
            assert!(out.is_empty(), "{ch:?}: {out:?}");
        }
    }

    // ── loop2 round 2: Codex loop2-r1 findings ───────────────────────────────

    #[test]
    fn a_wrong_number_is_a_constraint_not_a_free_pass() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("d.md", "Daemon", "recall daemon port 2219 runbook", "vault").unwrap();
        store.insert("i.md", "Invoice", "invoice 8784 paid", "vault").unwrap();
        store.insert("r.md", "Release", "release version 3412 of the launcher", "vault").unwrap();
        store.insert("o.md", "Order", "shopify order 55120 refund for the launcher", "vault").unwrap();
        store.insert("y.md", "Budget", "marketing budget 2025 review", "vault").unwrap();
        // Positive controls: the right number answers.
        for q in ["daemon port 2219", "release version 3412", "order 55120 refund", "marketing budget 2025"] {
            assert_eq!(fts5_only(&store, q).len(), 1, "{q} must answer");
        }
        // Wrong port, version, order number, year: the words match, the number does not.
        for q in ["daemon port 8784", "release version 9999", "order 55121 refund", "marketing budget 2031"] {
            assert!(never_confident(&store, q), "{q}: a different number is a different question");
        }
        assert_eq!(number_constraints("v1.2 on 2026-08-21 port 8784"), vec!["2026", "08", "21", "8784"]);
    }

    #[test]
    fn a_typo_correction_must_be_in_the_supporting_document() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("k.md", "Kestra", "kestra architecture overview", "vault").unwrap();
        store.insert("g.md", "Gmail", "gmail password reset", "vault").unwrap();
        // "kestrb" is one edit from "kestra", but only the architecture note holds "kestra".
        assert!(never_confident(&store, "kestrb password"), "the Gmail note is not about kestra");
        let j = Judge::build(&store, "kestrb password").unwrap();
        assert_eq!(j.typos.len(), 1);
        assert_eq!(fts5_only(&store, "kestrb architecture").len(), 1, "the correction is in the answer");
        assert_eq!(fts5_only(&store, "gmail passwrod").len(), 1);
    }

    #[test]
    fn searchable_counts_are_exact_however_many_excluded_rows_come_first() {
        let _guard = env_lock();
        let store = in_memory_store();
        for i in 0..60 {
            store.insert(&format!("disabled-loader-roots/x{i}.md"), "X", "quokkaterm", "vault").unwrap();
        }
        let t = vec!["quokkaterm".to_string()];
        assert_eq!(store.searchable_rows(&t, 1).unwrap(), 0, "only excluded rows: absent");
        assert_eq!(store.term_searchable_df("quokkaterm", 5000).unwrap(), 0);
        store.insert("notes/q.md", "Q", "quokkaterm", "vault").unwrap();
        assert_eq!(store.searchable_rows(&t, 1).unwrap(), 1);
        assert!(store.term_searchable("quokkaterm").unwrap());
    }

    /// Codex r2 HIGH #2: excluded text must change no VERDICT, in either direction.
    #[test]
    fn excluded_documents_can_neither_admit_nor_veto_a_query() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("a.md", "A", "rotation runbook password vault", "vault").unwrap();
        store.insert("k.md", "Kestra", "kestra password rotation", "vault").unwrap();
        filler(&store, &["rotation", "runbook", "password", "vault", "kestra"]);
        let state = |q: &str| -> Vec<(String, bool)> {
            fts5_sourced(&store, q).into_iter().map(|(h, p)| (h.path, p.closest_match)).collect()
        };
        // Admit: an entity held only by excluded notes stays unknown.
        let q = "zorbleflux rotation runbook password";
        assert!(fts5_only(&store, q).is_empty());
        // Veto: a typo held only by excluded notes stays an excusable typo (same verdict, same list).
        let before = state("kestrb password");
        assert!(before.iter().any(|x| x.0 == "k.md"), "{before:?}");
        for i in 0..60 {
            let body = "zorbleflux kestrb";
            store.insert(&format!("disabled-loader-roots/z{i}.md"), "Z", body, "vault").unwrap();
            store.insert(&format!("skills/s{i}/SKILL.md"), "S", body, "vault").unwrap();
        }
        assert!(fts5_only(&store, q).is_empty(), "excluded copies must not make an entity findable");
        let j = Judge::build(&store, q).unwrap();
        assert_eq!(j.unknown, vec!["zorbleflux".to_string()]);
        assert_eq!(state("kestrb password"), before, "excluded copies must not veto or change the answer");
    }

    /// Codex r2 #6: the ranking limit is applied AFTER the exclusion filter, however many excluded
    /// notes outrank the eligible one, so the bge-only lexical judge (depth 5) still sees it.
    #[test]
    fn fts_search_keeps_reading_past_excluded_hits() {
        let _guard = env_lock();
        let store = in_memory_store();
        for i in 0..80 {
            store
                .insert(&format!("disabled-loader-roots/r{i}.md"), "kestra rotation", "kestra rotation", "vault")
                .unwrap();
        }
        store
            .insert("notes/runbook.md", "Runbook", "kestra rotation runbook with many other words in it", "vault")
            .unwrap();
        let hits = store.fts_search("kestra rotation", LEXICAL_JUDGE_DEPTH).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "notes/runbook.md");
    }

    /// Codex r2 HIGH #4 / G3 "numbers-only matches": an unavailable bge (no model, no vectors) must
    /// not hand a nonsense query to the ungated fts5 rescue.
    #[test]
    fn bge_unavailable_rescue_still_answers_to_the_lexical_rule() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("p.md", "Port", "daemon port 2219 notes", "vault").unwrap();
        store.insert("k.md", "Kestra", "kestra password rotation runbook", "vault").unwrap();
        let bge_only = |q: &str| fused(&store, None, q, 5, &["bge".into()]).unwrap();
        assert!(bge_only("plimbex trondulate vasquor 2219").is_empty(), "nonsense plus a number");
        assert!(bge_only("2219").is_empty(), "a number alone is never evidence");
        assert_eq!(bge_only("kestra password").len(), 1, "a real query still gets the rescue");
        // The rescue gate is the same rule when the judge is built only at rescue time.
        let other = fused(&store, None, "plimbex trondulate vasquor 2219", 5, &["qwen3".into()]).unwrap();
        assert!(other.is_empty(), "rescue for any failed channel is gated");
    }

    /// Codex r2 HIGH #1: the gate proves SOME hit holds the number; every RETURNED hit must too,
    /// including under --top 1 and after fusion.
    #[test]
    fn every_returned_hit_satisfies_the_number_and_typo_constraints() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("short.md", "daemon port", "daemon port 2219", "vault").unwrap();
        store.insert("inv.md", "Invoice", "invoice 8784 daemon", "vault").unwrap();
        store
            .insert(
                "runbook.md",
                "Runbook",
                "the recall daemon listens on port 8784; restart the daemon if the port is busy",
                "vault",
            )
            .unwrap();
        filler(&store, &["daemon", "port"]);
        // 2026-11-06 (abstain labels, never drops): every returned hit either holds the number or says it
        // does not (`lacks [8784]`), under --top 1 and after fusion.
        for top in [1, 3, 5] {
            for ch in [vec!["fts5".to_string()], vec!["fts5".into(), "graph".into()]] {
                let hits = fused_sourced(&store, None, None, "daemon port 8784", top, &ch).unwrap();
                assert!(!hits.is_empty(), "top {top} {ch:?}");
                assert!(hits.len() <= top);
                for (h, p) in &hits {
                    assert!(store.doc_matches_term(h.id, "8784").unwrap() || p.lacks.contains(&"8784".to_string()),
                        "{} lacks 8784 unlabelled: {p:?}", h.path);
                }
            }
        }
        // Typo constraint: a returned hit without a correction of "kestrb" is labelled with it.
        store.insert("k1.md", "Kestra", "kestra password rotation", "vault").unwrap();
        store.insert("g1.md", "Gmail", "gmail password password password", "vault").unwrap();
        let typo_hits = fused_sourced(&store, None, None, "kestrb password", 5, &["fts5".into()]).unwrap();
        assert!(typo_hits.iter().any(|(h, _)| h.path == "k1.md"), "{typo_hits:?}");
        for (h, p) in typo_hits {
            assert!(h.path == "k1.md" || p.lacks.contains(&"kestrb".to_string()), "{}: {p:?}", h.path);
        }
    }

    /// v3, Codex r2 HIGH #3: three of four findable words are not an answer when the missing one
    /// is the subject. A Kestra rotation runbook must not answer a Gmail question.
    #[test]
    fn a_short_query_needs_its_subject() {
        let _guard = env_lock();
        let store = in_memory_store();
        store
            .insert("kestra.md", "Kestra runbook", "kestra password rotation runbook steps", "vault")
            .unwrap();
        store.insert("gmail.md", "Gmail", "gmail smtp configuration", "vault").unwrap();
        filler(&store, &["password", "rotation", "runbook", "kestra", "smtp"]);
        let q = "gmail password rotation runbook";
        let j = Judge::build(&store, q).unwrap();
        assert_eq!(j.findable_count(), 4);
        for ch in [vec!["fts5".to_string()], vec!["bge".into()], vec!["fts5".into(), "bge".into()]] {
            assert!(fused_sourced(&store, None, None, q, 5, &ch).unwrap().iter().all(|(_, p)| p.closest_match), "{ch:?}: wrong subject");
        }
        let out = fts5_only(&store, "kestra password rotation runbook");
        assert_eq!(out.first().map(|h| h.path.as_str()), Some("kestra.md"));
        store
            .insert("gmail-rot.md", "Gmail rotation", "gmail password rotation runbook", "vault")
            .unwrap();
        let out = fts5_only(&store, q);
        assert_eq!(out.first().map(|h| h.path.as_str()), Some("gmail-rot.md"));
        // Pure: a long query is exempt; a tie on the top weight is satisfied by either term.
        let mut tie = j.clone();
        tie.weights = vec![2.0, 2.0, 1.0, 1.0];
        assert!(tie.holds_subject(&Evidence { held: vec![false, true, true, false], constraints_ok: true }));
        assert!(!tie.holds_subject(&Evidence { held: vec![false, false, true, true], constraints_ok: true }));
    }

    /// feat/abstain-calibrated: this store (21+ notes) calibrates itself, so a keyword query the r2
    /// rules refuse may come back as an explicit closest match. Either way it is never a confident
    /// answer: refused, or every hit `closest_match`, each one lacking the word labelled `lacks [missing]`.
    fn never_confident_without(hits: &[(Hit, Provenance)], missing: &str) -> bool {
        // (a hit that holds the word is named after it in these stores: gmail3.md)
        hits.iter().all(|(h, p)| p.closest_match && (h.path.contains(missing) || p.lacks.iter().any(|l| l == missing)))
    }

    /// Codex v3 r1 HIGH #2: the subject word is the COMMONEST one (twenty Gmail SMTP notes), so
    /// "rarest term" points at password/rotation/runbook, which the Kestra runbook holds. Four
    /// findable terms now need all four; five need four.
    #[test]
    fn a_common_subject_word_still_has_to_be_in_the_answer() {
        let _guard = env_lock();
        let store = in_memory_store();
        store
            .insert("kestra.md", "Kestra runbook", "kestra password rotation runbook steps", "vault")
            .unwrap();
        for i in 0..20 {
            store.insert(&format!("gmail{i}.md"), "Gmail", "gmail smtp configuration", "vault").unwrap();
        }
        let q = "gmail password rotation runbook";
        let j = Judge::build(&store, q).unwrap();
        assert_eq!(j.findable_count(), 4);
        for ch in [vec!["fts5".to_string()], vec!["bge".into()], vec!["fts5".into(), "bge".into()]] {
            assert!(never_confident_without(&fused_sourced(&store, None, None, q, 5, &ch).unwrap(), "gmail"), "{ch:?}: gmail is missing");
        }
        // v3 r2 (Codex r1b HIGH #2): gmail (df 20) is rare, so the runbook lacking it never
        // answers, at five findable terms (4 of 5 held) or at six (5 of 6 held).
        store.insert("k2.md", "K", "kestra password rotation runbook steps smtp", "vault").unwrap();
        for q in ["gmail password rotation runbook steps", "gmail password rotation runbook steps smtp"] {
            assert!(Judge::build(&store, q).unwrap().findable_count() >= 5, "{q}");
            for ch in [vec!["fts5".to_string()], vec!["bge".into()], vec!["fts5".into(), "bge".into()]] {
                assert!(never_confident_without(&fused_sourced(&store, None, None, q, 5, &ch).unwrap(), "gmail"), "{ch:?} {q}: gmail is missing");
            }
        }
        // A COMMON missing word is phrasing, not the subject: all-but-one still answers.
        for i in 0..=RARE_TERM_DF {
            store.insert(&format!("n{i}.md"), "N", "assorted notes", "vault").unwrap();
        }
        let q5 = "kestra password rotation runbook notes";
        assert_eq!(Judge::build(&store, q5).unwrap().findable_count(), 5);
        assert!(!fts5_only(&store, q5).is_empty(), "4 of 5 answers when the missing word is common");
        // ...but three of five does not.
        assert!(never_confident_without(&fts5_sourced(&store, "kestra password rotation notes gmail"), "gmail"));
    }

    /// v3 r4 (the owner's decision on Codex r3 HIGH #1): a hit lacking a subject term is never refused
    /// for it and never silent. With gmail common (201 notes) the Kestra runbook is the closest
    /// match for every form of the query, keyword or question, 5 or 7 terms: it comes back LABELLED
    /// `lacks: ["gmail"]` and the whole set is flagged partial. The assertions run on the hits that
    /// actually come back (r3's version looped over an abstained, empty result).
    #[test]
    fn a_missing_subject_term_is_labelled_on_every_returned_hit() {
        let _guard = env_lock();
        let store = in_memory_store();
        let body = "kestra password rotation runbook steps emergency recovery checklist";
        store.insert("kestra.md", "Kestra runbook", body, "vault").unwrap();
        for i in 0..=RARE_TERM_DF {
            store.insert(&format!("gmail{i}.md"), "Gmail", "gmail smtp configuration", "vault").unwrap();
        }
        filler(&store, &["password", "rotation", "runbook", "steps", "emergency", "recovery", "checklist"]);
        assert!(!Judge::build(&store, "gmail x").unwrap().weights[0].ge(&rare_term_min_weight()), "gmail is common");
        for q in [
            "gmail password rotation runbook steps",
            "gmail password rotation runbook emergency recovery checklist",
            "how gmail password rotation runbook steps",
            "Which gmail password rotation runbook steps?",
        ] {
            for ch in [vec!["fts5".to_string()], vec!["bge".into()], vec!["fts5".into(), "bge".into()]] {
                let out = fused_sourced(&store, None, None, q, 5, &ch).unwrap();
                let kestra = out.iter().find(|(h, _)| h.path == "kestra.md");
                let (_, p) = kestra.unwrap_or_else(|| panic!("{ch:?} {q}: the closest match comes back: {out:?}"));
                assert_eq!(p.lacks, vec!["gmail".to_string()], "{ch:?} {q}");
                for (h, p) in &out {
                    let holds = store.doc_matches_term(h.id, "gmail").unwrap();
                    assert_eq!(p.lacks.contains(&"gmail".to_string()), !holds, "{ch:?} {q}: {}", h.path);
                }
                if out.iter().all(|(h, _)| !store.doc_matches_term(h.id, "gmail").unwrap()) {
                    let n = partial_notice(out.iter().map(|(_, p)| p.lacks.as_slice()));
                    assert!(n.is_some_and(|n| n.contains("[gmail]")), "{ch:?} {q}");
                }
            }
        }
        // The real subject answers and its hit lacks nothing, so nothing is flagged.
        let out = fused_sourced(&store, None, None, "kestra password rotation runbook steps", 5, &["fts5".into()]).unwrap();
        assert_eq!(out.first().map(|(h, _)| h.path.as_str()), Some("kestra.md"));
        assert!(out[0].1.lacks.is_empty(), "{:?}", out[0].1);
        assert!(partial_notice(out.iter().map(|(_, p)| p.lacks.as_slice())).is_none());
        assert_eq!(lacks_label(&["gmail".into()]).as_deref(), Some("closest match \u{2014} does not mention gmail"));
        assert!(lacks_label(&[]).is_none());
    }

    /// Codex v3 r3 HIGH #2: a channel that is unavailable (qwen3 with no model) falls back to the
    /// fts5 rescue. A short gmail/password note ranks first, the complete runbook second. The rescue
    /// judged the query, so `--top 1` must return the supporting runbook, never the note alone.
    #[test]
    fn a_short_rescue_query_returns_a_supporting_hit() {
        let _guard = env_lock();
        std::env::remove_var(ABSTAIN_POLICY_ENV);
        let store = in_memory_store();
        store
            .insert("short.md", "gmail password gmail password", "gmail password gmail password gmail password", "vault")
            .unwrap();
        let body = format!("gmail password rotation {}", "filler words of a long note ".repeat(60));
        store.insert("runbook.md", "Runbook", &body, "vault").unwrap();
        filler(&store, &["gmail", "password", "banana"]);
        for i in 0..40 {
            store.insert(&format!("rot{i}.md"), "R", "rotation schedule", "vault").unwrap();
        }
        let q = "gmail password rotation";
        let first = store.fts_search(q, 10).unwrap()[0].path.clone();
        assert_eq!(first, "short.md", "fixture: the unsupported note ranks first");
        // 2026-11-06: --top 1 returns the rescue ranking as ranked; the unsupported note is LABELLED
        // (closest match, lacks rotation), never a confident answer, and never swapped for a deeper hit.
        let out = fused_sourced(&store, None, None, q, 1, &["qwen3".into()]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0.path, "short.md", "{out:?}");
        assert!(out[0].1.closest_match && out[0].1.lacks.contains(&"rotation".to_string()), "{:?}", out[0].1);
        assert_eq!(out[0].1.source, "fts5-rescue");
    }

    /// Codex v3 r4 HIGH #1: a SUCCESSFUL non-fts5/bge channel (rg) used to build no judge, so its
    /// hits carried no `lacks` and nonsense plus one real word answered. rg ORs the words, so the
    /// Gmail-only and password-only notes come back next to the real answer: each is labelled with
    /// the word it lacks, and "qxzvbnwplkjh password" abstains like it does on fts5.
    #[test]
    fn every_channel_labels_its_hits_and_answers_to_the_lexical_rule() {
        let _guard = env_lock();
        let store = in_memory_store();
        let dir = std::env::temp_dir().join(format!("recall-rg-label-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (path, body) in [
            ("gmail.md", "gmail smtp configuration"),
            ("pw.md", "password reset policy"),
            ("both.md", "gmail password rotation"),
        ] {
            std::fs::write(dir.join(path), body).unwrap();
            store.insert(path, path, body, "vault").unwrap();
        }
        filler(&store, &["banana", "telescope"]);
        std::env::set_var("RECALL_VAULT", &dir);
        let labelled = fused_sourced(&store, None, None, "gmail password", 5, &["rg".into()]);
        let nonsense = fused_sourced(&store, None, None, "qxzvbnwplkjh password", 5, &["rg".into()]);
        std::env::remove_var("RECALL_VAULT");
        let _ = std::fs::remove_dir_all(&dir);
        let out = labelled.unwrap();
        let lacks: HashMap<&str, Vec<String>> = out.iter().map(|(h, p)| (h.path.as_str(), p.lacks.clone())).collect();
        assert_eq!(lacks.len(), 3, "rg returns all three notes: {out:?}");
        assert!(out.iter().all(|(_, p)| p.source == "rg"), "{out:?}");
        assert_eq!(lacks["gmail.md"], vec!["password".to_string()]);
        assert_eq!(lacks["pw.md"], vec!["gmail".to_string()]);
        assert!(lacks["both.md"].is_empty());
        assert!(nonsense.unwrap().is_empty(), "an unknown word abstains on rg too");
    }

    /// Codex v3 r7 HIGH: a word typed in capitals is a subject even when it is also a generic /
    /// judged term ("MAKE", "TIME", "GUIDE"); typed in lowercase it stays phrasing. MEDIUM: label
    /// subjects are filtered BEFORE the 64 cap, and a cap that still bites is reported.
    #[test]
    fn capitalised_generic_words_are_labelled_and_the_cap_counts_subjects() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("smtp.md", "SMTP", "smtp relay settings", "vault").unwrap();
        for w in ["make", "time", "guide"] {
            store.insert(&format!("{w}.md"), w, &format!("{w} smtp relay"), "vault").unwrap();
        }
        store.insert("r.md", "R", "R plotting with ggplot", "vault").unwrap();
        store.insert("pine.md", "Pine", "pine script bar plotting", "vault").unwrap();
        let letters: Vec<String> = ('b'..='z').flat_map(|a| ['q', 'x', 'y'].map(|b| format!("{b}{a}"))).collect();
        store.insert("many.md", "Many", &format!("smtp {}", letters.join(" ")), "vault").unwrap();
        filler(&store, &["banana", "telescope"]);
        for (q, word) in [("MAKE smtp", "make"), ("TIME smtp", "time"), ("GUIDE smtp", "guide")] {
            let out = fused_sourced(&store, None, None, q, 8, &["fts5".into()]).unwrap();
            let lacks = |path: &str| out.iter().find(|(h, _)| h.path == path).map(|(_, p)| p.lacks.clone());
            assert_eq!(lacks(&format!("{word}.md")), Some(vec![]), "{q}: {out:?}");
            assert_eq!(lacks("smtp.md"), Some(vec![word.to_string()]), "{q}: {out:?}");
        }
        let out = fused_sourced(&store, None, None, "make smtp", 8, &["fts5".into()]).unwrap();
        assert!(out.iter().any(|(h, p)| h.path == "smtp.md" && p.lacks.is_empty()), "lowercase is phrasing: {out:?}");
        // 64 ignored "a"s do not use up the label budget before "R".
        let q = format!("{}R plotting", "a ".repeat(64));
        let out = fused_sourced(&store, None, None, &q, 5, &["fts5".into()]).unwrap();
        assert_eq!(out.iter().find(|(h, _)| h.path == "pine.md").map(|(_, p)| p.lacks.clone()), Some(vec!["r".into()]));
        assert!(out.iter().all(|(_, p)| !p.labels_truncated));
        // 75 real label subjects: the first 64 are checked and every hit says so.
        let q = format!("smtp {}", letters.join(" "));
        let j = Judge::build(&store, &q).unwrap();
        assert!(letters.len() > LABEL_TOKEN_CAP && j.short.len() == LABEL_TOKEN_CAP && j.labels_truncated);
        let out = fused_sourced(&store, None, None, &q, 5, &["fts5".into()]).unwrap();
        assert!(!out.is_empty() && out.iter().all(|(_, p)| p.labels_truncated), "{out:?}");
        assert!(truncation_note(true).is_some() && truncation_note(false).is_none());
    }

    /// Codex v3 r6 HIGH: one-letter subjects ("R", "C", "q z x v j") and capitalised acronyms that
    /// spell a function word ("IT") are labelled; "it" / "or" typed in lowercase are not.
    #[test]
    fn one_letter_and_acronym_subjects_are_labelled() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("r.md", "R", "R plotting with ggplot", "vault").unwrap();
        store.insert("pine.md", "Pine", "pine script bar plotting", "vault").unwrap();
        store.insert("c.md", "C", "C smtp client library", "vault").unwrap();
        store.insert("it.md", "IT", "IT department smtp relay", "vault").unwrap();
        store.insert("smtp.md", "SMTP", "smtp relay settings", "vault").unwrap();
        store.insert("pw.md", "Passwords", "password rotation policy", "vault").unwrap();
        for w in ["q", "z", "x", "v"] {
            store.insert(&format!("{w}.md"), w, &format!("letter {w} appears here"), "vault").unwrap();
        }
        filler(&store, &["banana", "telescope"]);
        assert_eq!(short_subject_terms("R plotting", &["plotting".into()]), vec!["r"]);
        assert_eq!(short_subject_terms("IT smtp", &["smtp".into()]), vec!["it"]);
        assert!(short_subject_terms("is it smtp or not", &["smtp".into()]).is_empty(), "lowercase function words");
        assert!(short_subject_terms("How do I use what's A smtp", &["smtp".into()]).is_empty(), "I, A, 's");
        assert_eq!(short_subject_terms("OR US 7 smtp", &["smtp".into()]), vec!["or", "us", "7"]);
        for (q, word, holder, other) in [
            ("R plotting", "r", "r.md", "pine.md"),
            ("C smtp", "c", "c.md", "smtp.md"),
            ("IT smtp", "it", "it.md", "smtp.md"),
        ] {
            let out = fused_sourced(&store, None, None, q, 5, &["fts5".into()]).unwrap();
            let lacks = |path: &str| out.iter().find(|(h, _)| h.path == path).map(|(_, p)| p.lacks.clone());
            assert_eq!(lacks(holder), Some(vec![]), "{q}: {out:?}");
            assert_eq!(lacks(other), Some(vec![word.to_string()]), "{q}: {out:?}");
        }
        // Lowercase "it" / "or" stay function words: the smtp-only hit carries no label.
        let out = fused_sourced(&store, None, None, "is it smtp or relay", 5, &["fts5".into()]).unwrap();
        assert!(out.iter().any(|(h, p)| h.path == "smtp.md" && p.lacks.is_empty()), "{out:?}");
        // Every letter is held by some note, so the letters label the password hit; "j" is held by
        // none, so the query abstains as an unknown subject.
        let out = fused_sourced(&store, None, None, "q z x v password", 5, &["fts5".into()]).unwrap();
        let pw = out.iter().find(|(h, _)| h.path == "pw.md").expect("closest match returned");
        assert_eq!(pw.1.lacks, vec!["q", "z", "x", "v"]);
        assert!(partial_notice(out.iter().map(|(_, p)| p.lacks.as_slice())).is_some());
        assert!(fused_sourced(&store, None, None, "q z x v j password", 5, &["fts5".into()]).unwrap().is_empty());
    }

    /// Codex v3 r5 HIGH #1: two-character subjects ("AI", "qx") and non-ASCII ones ("日本") are
    /// judged and labelled like any other word; only the fixed SHORT_STOPWORDS list is dropped.
    /// Numbers stay hard constraints: a port-9000 note never answers "service port 8784".
    #[test]
    fn short_numeric_and_non_ascii_subjects_are_judged() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("smtp.md", "SMTP", "smtp relay settings for the mail server", "vault").unwrap();
        store.insert("ai-smtp.md", "AI mail", "AI agent smtp relay notes", "vault").unwrap();
        store.insert("pw.md", "Passwords", "password rotation policy", "vault").unwrap();
        store.insert("svc.md", "Service", "service port 9000 for the daemon", "vault").unwrap();
        store.insert("jp.md", "Japan", "日本 travel notes", "vault").unwrap();
        filler(&store, &["banana", "telescope", "symphony"]);
        assert_eq!(short_subject_terms("AI smtp", &confidence_terms("AI smtp")), vec!["ai"]);
        assert!(short_subject_terms("go to it", &confidence_terms("go to it")).is_empty(), "judged already");
        assert!(short_subject_terms("the of it smtp", &["smtp".into()]).is_empty(), "lowercase stopwords");
        assert_eq!(confidence_terms("日本 password"), vec!["日本", "password"]);
        for ch in [vec!["fts5".to_string()], vec!["fts5".into(), "bge".into()]] {
            let out = fused_sourced(&store, None, None, "AI smtp", 5, &ch).unwrap();
            assert!(out.iter().any(|(h, p)| h.path == "ai-smtp.md" && p.lacks.is_empty()), "{ch:?} {out:?}");
            assert!(partial_notice(out.iter().map(|(_, p)| p.lacks.as_slice())).is_none());
            for (h, p) in &out {
                let holds = store.doc_matches_term(h.id, "ai").unwrap();
                assert_eq!(p.lacks.contains(&"ai".to_string()), !holds, "{ch:?} {}: {:?}", h.path, p.lacks);
            }
            assert!(out.iter().any(|(h, p)| h.path == "smtp.md" && p.lacks == vec!["ai".to_string()]), "{ch:?} {out:?}");
            // "qx" is in no document and too short to be a typo: unknown, so the query abstains.
            assert!(fused_sourced(&store, None, None, "qx password", 5, &ch).unwrap().is_empty(), "{ch:?}");
            // "日本" IS indexed (jp.md), but no document holds it with "password": no support.
            assert!(fused_sourced(&store, None, None, "日本 password", 5, &ch).unwrap().iter().all(|(_, p)| p.closest_match), "{ch:?}");
            assert!(fused_sourced(&store, None, None, "service port 8784", 5, &ch).unwrap().is_empty(), "{ch:?}");
            let out = fused_sourced(&store, None, None, "service port 9000", 5, &ch).unwrap();
            assert_eq!(out.first().map(|(h, _)| h.path.as_str()), Some("svc.md"), "{ch:?}");
        }
    }

    #[test]
    fn question_form_and_generic_terms() {
        for q in [
            "Which port does the daemon use?",
            "In the Critic agent's review checklist, which category comes first?",
            "Does recall abstain on nonsense",
            "how do I rotate the gmail password",
        ] {
            assert!(is_question_form(q), "{q}");
        }
        for q in ["kestra password", "how gmail password rotation runbook steps", "notes about Phone Number Decision", "0-dimensional biomaterials show inductive properties."] {
            assert!(!is_question_form(q), "{q}");
        }
        assert!(is_generic_term("notes") && is_generic_term("steps") && !is_generic_term("gmail"));
    }

    /// v3 r3, Codex r2 HIGH #2: two snapshots of one file (agents.md / AGENTS.md). The copy that
    /// ranks first holds only gmail + password; the other holds all three terms and is what the
    /// judge approved. Deduplication keeps the approved copy, and a short query's returned set must
    /// hold a supporting hit, so `--top 1` can never return only the unsupported snapshot.
    #[test]
    fn dedup_keeps_the_approved_snapshot_and_short_queries_check_the_returned_set() {
        let _guard = env_lock();
        std::env::remove_var(ABSTAIN_POLICY_ENV);
        let store = in_memory_store();
        let weak = store
            .insert("lib/agents.md", "gmail password gmail password", "gmail password gmail password gmail password", "vault")
            .unwrap();
        let long_body = format!("gmail password rotation {}", "filler words of a long note ".repeat(60));
        let strong = store.insert("lib/AGENTS.md", "Agents", &long_body, "vault").unwrap();
        filler(&store, &["gmail", "password", "banana"]);
        // "rotation" is common, so the weak snapshot's title and term frequency rank it first.
        for i in 0..40 {
            store.insert(&format!("rot{i}.md"), "R", "rotation schedule", "vault").unwrap();
        }
        let q = "gmail password rotation";
        let raw_order: Vec<i64> = store.fts_search(q, 10).unwrap().iter().map(|h| h.id).collect();
        assert_eq!(raw_order.first(), Some(&weak), "fixture: the unsupported snapshot ranks first");
        // 2026-11-06: the kept copy is the best-RANKED one with abstain on or off (it used to be the
        // judge-approved one, so abstain changed the returned id); the collapse is disclosed, and the
        // kept copy says which judged term it lacks.
        for top in [1, 5] {
            let out = fused_sourced(&store, None, None, q, top, &["fts5".into()]).unwrap();
            let lib: Vec<_> = out.iter().filter(|(h, _)| h.path.to_lowercase() == "lib/agents.md").collect();
            assert_eq!(lib.len(), 1, "top {top}: one file, one slot");
            assert_eq!(lib[0].0.id, weak, "top {top}: the best-ranked snapshot is kept");
            assert_eq!(lib[0].1.merged, vec![strong], "top {top}: the collapse is disclosed");
            assert!(lib[0].1.lacks.contains(&"rotation".to_string()), "top {top}: {:?}", lib[0].1);
        }
        // Unit: the preferred copy is kept even when another copy ranks first everywhere.
        let mut ranked = vec![Ranked { channel: "fts5".into(), ids: vec![1, 2, 3] }];
        let id_hit: HashMap<i64, Hit> =
            [(1, "a/X.md"), (2, "a/x.md"), (3, "b.md")].into_iter().map(|(i, p)| (i, test_hit(i, p))).collect();
        let mut raw: HashMap<String, HashMap<i64, f64>> = HashMap::new();
        raw.insert("fts5".into(), HashMap::from([(1, 9.0), (2, 4.0), (3, 1.0)]));
        let merged = collapse_case_duplicates(&mut ranked, &mut raw, &id_hit, &|id| id == 2);
        assert_eq!(ranked[0].ids, vec![2, 3], "kept at the best copy's position");
        assert_eq!(raw["fts5"][&2], 4.0, "the kept copy keeps its OWN score");
        assert_eq!(merged[&2], vec![1]);
    }

    /// Codex r2 #5: a two-digit port is a constraint like any other number.
    #[test]
    fn two_digit_numbers_constrain_the_answer() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("ssh.md", "SSH", "the daemon port 22 is ssh", "vault").unwrap();
        filler(&store, &["daemon", "port", "ssh"]);
        // 2026-11-06: a returned note without 22 is labelled with it (it used to be dropped).
        let h = fts5_sourced(&store, "daemon port 22");
        assert_eq!(h.first().map(|(x, _)| x.path.as_str()), Some("ssh.md"), "{h:?}");
        assert!(h.iter().all(|(x, p)| x.path == "ssh.md" || p.lacks.contains(&"22".to_string())), "{h:?}");
        assert!(fts5_only(&store, "daemon port 23").is_empty(), "23 is in no note: refused (refuse policy)");
    }

    /// G2a (loop2 r3): a long sentence of ordinary words matched by a long note through three
    /// incidental words must abstain; a long real question its note answers must not.
    #[test]
    fn long_queries_need_a_weighted_majority_of_their_terms() {
        let _guard = env_lock();
        let store = in_memory_store();
        let words = [
            "volcano", "borrows", "rusty", "carpet", "argues", "golden", "pigeon", "compass", "lantern",
        ];
        filler(&store, &words);
        store.insert("book.md", "Book", "a long book about a volcano, a carpet and a pigeon", "vault").unwrap();
        let nonsense =
            "the volcano borrows the rusty carpet, argues with a golden pigeon, compass and lantern";
        let j = Judge::build(&store, nonsense).unwrap();
        assert!(j.findable_count() >= LONG_QUERY_FINDABLE, "{:?}", j.terms);
        assert!(never_confident(&store, nonsense), "3 incidental words of 9 is not an answer");
        store
            .insert("real.md", "Volcano notes", "the volcano borrows rusty carpet; argues golden pigeon", "vault")
            .unwrap();
        // v3 r4: keyword or question, the share floor decides and 7 of 9 answers, labelled with
        // what it lacks (compass, lantern).
        for q in [nonsense.to_string(), format!("which {nonsense}?")] {
            let out = fused_sourced(&store, None, None, &q, 5, &["fts5".into()]).unwrap();
            let real = out.iter().find(|(h, _)| h.path == "real.md");
            assert_eq!(real.map(|(_, p)| p.lacks.clone()), Some(vec!["compass".to_string(), "lantern".into()]), "{q}: {out:?}");
        }
        assert!(j.unknown.is_empty() && j.typos.is_empty(), "abstains on the share, not a veto");
        // Pure: below LONG_QUERY_FINDABLE terms the share floor does not apply.
        let short = Judge {
            terms: vec!["a".into(), "b".into(), "c".into()],
            weights: vec![1.0, 1.0, 1.0],
            findable: vec![true; 3],
            typos: vec![],
            unknown: vec![],
            numbers: vec![],
            subject: vec![true; 3],
            short: vec![],
            labels_truncated: false,
        };
        let e = Evidence { held: vec![true, true, true], constraints_ok: true };
        assert!(lexically_supports(&short, &e));
        let mut long = short.clone();
        long.terms = (0..10).map(|i| format!("t{i}")).collect();
        long.weights = vec![1.0; 10];
        long.findable = vec![true; 10];
        let mut held = vec![false; 10];
        held[..5].fill(true);
        assert!(!lexically_supports(&long, &Evidence { held: held.clone(), constraints_ok: true }));
        held[5] = true;
        assert!(lexically_supports(&long, &Evidence { held, constraints_ok: true }));
    }

    #[test]
    fn excluded_occurrences_cannot_move_a_weight_or_a_verdict() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("a.md", "A", "alpha bravo charlie delta echo", "vault").unwrap();
        store.insert("b.md", "B", "alpha notes", "vault").unwrap();
        filler(&store, &["bravo", "charlie", "delta", "echo"]);
        let q = "alpha bravo charlie delta echo";
        let before = Judge::build(&store, q).unwrap();
        let verdict = fts5_only(&store, q).len();
        for i in 0..40 {
            store.insert(&format!("disabled-loader-roots/a{i}.md"), "X", "alpha", "vault").unwrap();
            store.insert(&format!("skills/s{i}/SKILL.md"), "S", "alpha", "vault").unwrap();
        }
        let after = Judge::build(&store, q).unwrap();
        assert_eq!(before.weights, after.weights, "excluded copies must not reweight a term");
        assert_eq!(fts5_only(&store, q).len(), verdict);
    }

    #[test]
    fn an_oversized_absent_token_is_unknown_without_expansion() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("k.md", "Kestra", "kestra password", "vault").unwrap();
        let long = "q".repeat(5000);
        assert!(!is_plausible_typo(&store, &long).unwrap());
        assert!(!is_plausible_typo(&store, &"k".repeat(TYPO_MAX_LEN + 1)).unwrap());
        let t = std::time::Instant::now();
        assert!(fts5_only(&store, &format!("kestra {long}")).is_empty());
        assert!(t.elapsed() < std::time::Duration::from_secs(5), "{:?}", t.elapsed());
    }

    /// feat/abstain-calibrated: a 30-note chat store, enough to calibrate itself.
    fn chat_store() -> Store {
        let store = in_memory_store();
        add_chat_notes(&store);
        store
    }

    /// Threads that computed a phantom null, and a delay each computation takes (both test-only),
    /// so a daemon test can see WHERE calibration ran and make it slow enough to block a loop.
    pub(crate) static CALIBRATED_ON: Mutex<Vec<String>> = Mutex::new(Vec::new());
    pub(crate) static CALIBRATION_DELAY_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    pub(super) fn note_calibration_thread() {
        let name = std::thread::current().name().unwrap_or("?").to_string();
        CALIBRATED_ON.lock().unwrap_or_else(|e| e.into_inner()).push(name);
        let ms = CALIBRATION_DELAY_MS.load(std::sync::atomic::Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }

    #[test]
    fn a_pending_view_is_queued_once_until_the_calibrator_finishes_it() {
        let (q, rx) = CalibrationQueue::new();
        let w1 = crate::scope::resolve(Some("w1".into()), None, None).unwrap();
        for _ in 0..3 {
            q.submit(w1.clone());
            q.submit(crate::scope::Reader::Operator);
        }
        assert_eq!(rx.try_iter().count(), 2, "one job per view, however many queries miss");
        // a long-running job never expires into a twin (Codex calib r2)
        q.age(CALIBRATION_RETRY * 3);
        q.submit(w1.clone());
        assert_eq!(rx.try_iter().count(), 0, "still pending: no second job");
        q.status().finished(&w1, false); // a failed attempt waits out CALIBRATION_RETRY...
        q.submit(w1.clone());
        assert_eq!(rx.try_iter().count(), 0);
        q.age(CALIBRATION_RETRY);
        q.submit(w1.clone()); // ...then queues once more
        q.submit(w1.clone());
        assert_eq!(rx.try_iter().count(), 1);
        q.status().finished(&w1, true); // stored: the next miss (a later corpus change) queues again
        q.submit(w1);
        assert_eq!(rx.try_iter().count(), 1);
    }

    /// The daemon's stale stand-in (Codex calib r2): an evolved corpus is served the view's last
    /// null while it recalibrates; a rewritten corpus under the same prefix is not.
    #[test]
    fn a_stale_null_stands_in_only_for_an_evolved_corpus() {
        let _guard = env_lock();
        let (q, rx) = CalibrationQueue::new();
        // a thread of its own, so the thread-local calibrator cannot leak into other tests
        std::thread::scope(|s| {
            s.spawn(move || {
                let a = chat_store();
                let n1 = phantom_null(&a).unwrap();
                let _defer = defer_calibration_to(q.clone());
                // (a JSON round trip may move the last ulp)
                let close = |v: &[f64]| v.len() == n1.len() && v.iter().zip(&n1).all(|(x, y)| (x - y).abs() < 1e-12);
                assert!(close(&phantom_null_or_defer(&a).unwrap().unwrap()), "fresh: the stored null");
                assert_eq!(rx.try_iter().count(), 0);
                for i in 0..4 {
                    // past PHANTOM_REUSE_MIN_SIMILARITY (see phantom_probes_...), within the stale bound
                    a.insert(&format!("more{i}.md"), "m", &format!("Melanie: kayak lesson {i}"), "vault").unwrap();
                }
                let stale = phantom_null_or_defer(&a).unwrap().expect("evolved corpus: stale null served");
                assert!(close(&stale));
                assert_eq!(rx.try_iter().count(), 1, "and a recalibration queued");
                // a disjoint corpus carrying that null under its own prefix gets none
                let b = in_memory_store();
                for i in 0..30 {
                    b.insert(&format!("o{i:02}.md"), "o", &format!("Carol: chess opening {i} rook"), "vault").unwrap();
                }
                let row = a.calibration_latest(&phantom_prefix(&crate::scope::Reader::Operator)).unwrap().unwrap();
                b.calibration_put(&format!("v{PHANTOM_VERSION}:0:0"), &row).unwrap();
                assert_eq!(phantom_null_or_defer(&b).unwrap(), None, "rewritten corpus: strict until recalibrated");
            });
        });
    }

    /// The 30 notes of [`chat_store`], into any store (the daemon's calibrator test needs a file).
    pub(crate) fn add_chat_notes(store: &Store) {
        let w = [
            "camping", "painting", "pottery", "beach", "guitar", "adoption", "museum", "garden",
            "violin", "marathon", "bakery", "library", "festival", "puppy", "workshop",
        ];
        for i in 0..30 {
            let body = format!("Melanie: {} {} {} today", w[i % 15], w[(i * 7 + 3) % 15], w[(i * 11 + 5) % 15]);
            store.insert(&format!("t{i:02}.md"), &format!("t{i}"), &body, "vault").unwrap();
        }
    }

    #[test]
    fn phantom_probes_are_unanswerable_and_deterministic() {
        let _guard = env_lock();
        let store = chat_store();
        // a corpus state no other test has, so the in-process cache cannot answer for it
        store.insert("own.md", "own", "Melanie: harmonica lessons", "vault").unwrap();
        let ids = store.searchable_snapshot().unwrap().1;
        let a = phantom_queries(&store, &ids, 7).unwrap();
        assert!(a.len() >= PHANTOM_MIN_PROBES, "{}", a.len());
        assert_eq!(a, phantom_queries(&store, &ids, 7).unwrap());
        for q in &a {
            let words: Vec<String> = q.trim_start_matches("What did ").trim_end_matches('?')
                .split(' ').map(String::from).collect();
            assert!(!store.some_doc_holds_all(&words).unwrap(), "{q}");
        }
        // the null is cached in the store, keyed by corpus state
        let n1 = phantom_null(&store).unwrap();
        assert_eq!(n1.len(), a.len());
        let key = format!("v{PHANTOM_VERSION}:{}", store.searchable_snapshot().unwrap().0);
        assert!(store.calibration_get(&key).unwrap().is_some());
        store.insert("new.md", "new", "brand new note", "vault").unwrap();
        let key2 = format!("v{PHANTOM_VERSION}:{}", store.searchable_snapshot().unwrap().0);
        assert_ne!(key, key2);
        // one new note in 32 (similarity ~0.97) reuses the stored null instead of recalibrating
        let reused = phantom_null(&store).unwrap(); // (a JSON round trip may move the last ulp)
        assert!(reused.len() == n1.len() && reused.iter().zip(&n1).all(|(a, b)| (a - b).abs() < 1e-12));
        assert!(store.calibration_get(&key2).unwrap().is_none());
        // a bigger change recalibrates and stores under the new fingerprint
        for i in 0..4 {
            store.insert(&format!("more{i}.md"), "m", &format!("Melanie: kayak lesson {i}"), "vault").unwrap();
        }
        phantom_null(&store).unwrap();
        let key3 = format!("v{PHANTOM_VERSION}:{}", store.searchable_snapshot().unwrap().0);
        assert!(store.calibration_get(&key3).unwrap().is_some());
    }

    /// Codex abstain-cal r1 #1/#2: the same ids with different text are a different corpus, and each
    /// store keeps its own null (there is no process-wide cache to leak one into the other).
    #[test]
    fn same_ids_different_text_is_a_different_calibration() {
        let _guard = env_lock();
        let a = chat_store();
        let b = in_memory_store();
        for i in 0..30 {
            b.insert(&format!("t{i:02}.md"), &format!("t{i}"), &format!("ledger entry {i} invoice audit payroll"), "vault").unwrap();
        }
        let (fa, ia, _) = a.searchable_snapshot().unwrap();
        let (fb, ib, _) = b.searchable_snapshot().unwrap();
        assert_eq!(ia, ib, "same ids");
        assert_ne!(fa, fb, "different content, different fingerprint");
        let (na, nb) = (phantom_null(&b).unwrap(), phantom_null(&a).unwrap());
        assert!(a.calibration_get(&format!("v{PHANTOM_VERSION}:{fa}")).unwrap().is_some());
        assert!(b.calibration_get(&format!("v{PHANTOM_VERSION}:{fb}")).unwrap().is_some());
        assert_ne!(na, nb);
        // any corpus change moves the fingerprint, so a stored null is never reused for new text
        let before = a.searchable_snapshot().unwrap().0;
        a.insert("t00.md", "t0", "Melanie: rewritten note", "vault").unwrap();
        assert_ne!(before, a.searchable_snapshot().unwrap().0);
    }

    /// Codex abstain-cal r2 HIGH: a stored null from a corpus of the SAME size but different
    /// content (a full rewrite, or a disjoint tier / exclude policy) must not be reused.
    #[test]
    fn a_same_size_different_corpus_recalibrates() {
        let _guard = env_lock();
        let a = chat_store();
        phantom_null(&a).unwrap();
        let (ka, _, _) = a.searchable_snapshot().unwrap();
        let stolen = a.calibration_get(&format!("v{PHANTOM_VERSION}:{ka}")).unwrap().unwrap();
        let b = in_memory_store();
        for i in 0..30 {
            b.insert(&format!("t{i:02}.md"), "t", &format!("ledger entry {i} invoice audit payroll"), "vault").unwrap();
        }
        b.calibration_put(&format!("v{PHANTOM_VERSION}:{ka}"), &stolen).unwrap();
        let (kb, _, _) = b.searchable_snapshot().unwrap();
        phantom_null(&b).unwrap();
        assert!(b.calibration_get(&format!("v{PHANTOM_VERSION}:{kb}")).unwrap().is_some(), "recalibrated");
        // the same corpus under an exclude policy that hides half of it is a different state too
        std::env::set_var("RECALL_EXCLUDE_EXTRA", "t0,t1");
        let (kx, ix, _) = a.searchable_snapshot().unwrap();
        std::env::remove_var("RECALL_EXCLUDE_EXTRA");
        assert!(ix.len() < 30 && kx != ka, "{}", ix.len());
    }

    #[test]
    fn sketch_similarity_estimates_jaccard() {
        assert_eq!(crate::store::sketch_similarity(&[], &[]), 1.0);
        assert_eq!(crate::store::sketch_similarity(&[1, 2, 3, 4], &[1, 2, 3, 4]), 1.0);
        assert_eq!(crate::store::sketch_similarity(&[1, 2, 3, 4], &[5, 6, 7, 8]), 0.0);
        assert!((crate::store::sketch_similarity(&[1, 2, 3], &[1, 2, 3, 4]) - 0.75).abs() < 1e-9);
    }

    /// Lead 2026-11-05 (packaging clean-room): a fresh 2-note install answered a plain query with
    /// nothing. A tiny store labels its closest matches instead; only a word no note holds refuses.
    #[test]
    fn a_two_note_store_answers_plain_queries_with_labelled_matches() {
        let _guard = env_lock();
        let store = in_memory_store();
        store.insert("a.md", "a", "tomato soup recipe with basil", "vault").unwrap();
        store.insert("b.md", "b", "garden notes: watering schedule", "vault").unwrap();
        assert!(!fts5_sourced(&store, "tomato").is_empty(), "plain one-word query");
        let h = fts5_sourced(&store, "tomato watering");
        assert!(!h.is_empty() && h.iter().all(|(_, p)| p.closest_match), "{h:?}");
        assert!(h.iter().any(|(_, p)| !p.lacks.is_empty()), "the gap is labelled: {h:?}");
        assert!(fts5_sourced(&store, "tomato qxzvbnwplkjh").is_empty(), "a word no note holds still refuses");
    }

    /// Policy decision 2026-11-06: the default answers a query the judge rejects with labelled hits, never silence.
    #[test]
    fn label_policy_returns_labelled_hits_for_a_rejected_query() {
        let _guard = env_lock();
        std::env::remove_var(ABSTAIN_POLICY_ENV);
        let store = small_memory_store();
        // r9 refused this one (one matched word, mass < 12, and keyword form)
        let h = fts5_sourced(&store, "zorblat cat luna");
        assert!(!h.is_empty() && h.iter().all(|(_, p)| p.closest_match), "{h:?}");
        assert!(h[0].1.lacks.contains(&"zorblat".to_string()), "{:?}", h[0].1);
        // a confident answer is untouched
        let c = fts5_sourced(&store, "What is the name of my cat luna?");
        assert!(!c.is_empty() && !c[0].1.closest_match);
        // nothing to show is still nothing
        assert!(fts5_sourced(&store, "qxzvbnwplkjh").is_empty());
        std::env::set_var(ABSTAIN_POLICY_ENV, "refuse");
        assert!(fts5_sourced(&store, "zorblat cat luna").is_empty(), "refuse mode keeps r9");
    }

    /// RECALL_JUDGE=haiku: a confident set the judge rejects is labelled; a judge that fails labels too.
    /// The CLI is faked with a script so no model is called.
    #[cfg(unix)]
    #[test]
    fn judge_labels_rejected_and_failed_results_and_keeps_accepted_ones() {
        let _guard = env_lock();
        std::env::remove_var(ABSTAIN_POLICY_ENV);
        let store = small_memory_store();
        let dir = std::env::temp_dir().join(format!("recall-judge-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fake = |name: &str, body: &str| {
            let p = dir.join(name);
            std::fs::write(&p, format!("#!/bin/sh\ncat > /dev/null\n{body}\n")).unwrap();
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
            p
        };
        std::env::set_var("RECALL_JUDGE", "haiku");
        let q = "What is the name of my cat luna?";
        for (script, labelled) in [("echo 2", true), ("echo 90", false), ("exit 3", true), ("echo maybe", true)] {
            let exe = fake(&format!("c{}.sh", script.len()), script);
            std::env::set_var("RECALL_JUDGE_CLAUDE", &exe);
            let h = fts5_sourced(&store, q);
            assert!(!h.is_empty(), "{script}: never emptied");
            assert_eq!(h.iter().all(|(_, p)| p.closest_match), labelled, "{script}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The same on Windows, through a fake `.cmd` launcher (npm installs claude as claude.CMD).
    #[cfg(windows)]
    #[test]
    fn judge_runs_a_cmd_launcher_on_windows() {
        let _guard = env_lock();
        std::env::remove_var(ABSTAIN_POLICY_ENV);
        let store = small_memory_store();
        let dir = std::env::temp_dir().join(format!("recall-judge-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("RECALL_JUDGE", "haiku");
        for (body, labelled) in [("@echo 2", true), ("@echo 90", false), ("@exit /b 3", true)] {
            let p = dir.join(format!("fake{}.cmd", body.len()));
            std::fs::write(&p, format!("@more > nul\r\n{body}\r\n")).unwrap();
            std::env::set_var("RECALL_JUDGE_CLAUDE", &p);
            let h = fts5_sourced(&store, "What is the name of my cat luna?");
            assert!(!h.is_empty(), "{body}: never emptied");
            assert_eq!(h.iter().all(|(_, p)| p.closest_match), labelled, "{body}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// integrate-rc1 2026-11-06: two fresh indexes of the SAME corpus, built in a different order (so every doc
    /// id differs), must calibrate identically.
    #[test]
    fn the_calibration_does_not_depend_on_index_order() {
        let _guard = env_lock();
        let w = [
            "camping", "painting", "pottery", "beach", "guitar", "adoption", "museum", "garden",
            "violin", "marathon", "bakery", "library", "festival", "puppy", "workshop",
        ];
        let notes: Vec<(String, String)> = (0..30)
            .map(|i| (format!("t{i:02}.md"), format!("Melanie: {} {} {} today", w[i % 15], w[(i * 7 + 3) % 15], w[(i * 11 + 5) % 15])))
            .collect();
        let build = |order: &[usize]| {
            let st = in_memory_store();
            for &i in order {
                st.insert(&notes[i].0, &notes[i].0, &notes[i].1, "vault").unwrap();
            }
            st
        };
        let fwd: Vec<usize> = (0..30).collect();
        let shuffled: Vec<usize> = (0..30).map(|i| (i * 7 + 11) % 30).collect();
        let (a, b) = (build(&fwd), build(&shuffled));
        assert_ne!(a.searchable_snapshot().unwrap().1, b.searchable_by_path().unwrap().iter().map(|d| d.0).collect::<Vec<_>>(),
            "the two indexes really do number the notes differently");
        let (na, nb) = (phantom_null(&a).unwrap(), phantom_null(&b).unwrap());
        assert!(na.len() >= PHANTOM_MIN_PROBES);
        assert_eq!(na, nb, "same corpus, same calibration");
    }

    /// Codex det1 #1: equal-BM25 ties at the top-5 cut must not be broken by doc id. 20 notes of equal length,
    /// nine "camping painting", six "camping pottery", five "painting pottery": a three-term probe ties them all.
    #[test]
    fn fts5_ties_break_by_path_not_insertion_order() {
        let _guard = env_lock();
        let notes: Vec<(String, &str)> = (0..20)
            .map(|i| (format!("n{i:02}.md"), if i < 9 { "camping painting" } else if i < 15 { "camping pottery" } else { "painting pottery" }))
            .collect();
        let build = |rev: bool| {
            let st = in_memory_store();
            let order: Vec<usize> = if rev { (0..20).rev().collect() } else { (0..20).collect() };
            for i in order {
                st.insert(&notes[i].0, &notes[i].0, notes[i].1, "vault").unwrap();
            }
            st
        };
        let (a, b) = (build(false), build(true));
        let q = "camping painting pottery";
        let paths = |st: &Store| -> Vec<String> {
            st.fts_judge_ids(q, LEXICAL_JUDGE_DEPTH).unwrap().iter().map(|id| lookup_path_of(st, *id)).collect()
        };
        assert_eq!(paths(&a), paths(&b));
        let score = |st: &Store| {
            let j = Judge::build(st, q).unwrap();
            closest_score(&j, &j.evidences(st, &st.fts_judge_ids(q, LEXICAL_JUDGE_DEPTH).unwrap()).unwrap())
        };
        assert_eq!(score(&a), score(&b));
    }

    fn lookup_path_of(st: &Store, id: i64) -> String {
        st.conn.query_row("SELECT path FROM documents WHERE id = ?1", [id], |r| r.get(0)).unwrap()
    }

    /// Codex det1 #2: a legacy row without content_sha256 is identified by its body, never by valid_from.
    #[test]
    fn a_missing_content_hash_falls_back_to_the_body_not_the_timestamp() {
        let _guard = env_lock();
        let mk = |stamp: &str| {
            let st = chat_store();
            st.conn.execute("UPDATE documents SET content_sha256 = NULL, valid_from = ?1", [stamp]).unwrap();
            st
        };
        let (a, b) = (mk("2026-01-01T00:00:00Z"), mk("2026-11-06T12:00:00Z"));
        assert_eq!(a.searchable_snapshot().unwrap().0, b.searchable_snapshot().unwrap().0);
        assert_eq!(phantom_null(&a).unwrap(), phantom_null(&b).unwrap());
    }

    /// Lead 2026-11-06 (VibeMem): abstain only LABELS or REFUSES. For every query that returns hits both with
    /// abstain off and on (label or refuse policy), the ranked id list is identical; only the labels differ.
    /// Covers the paths that used to reorder: number and typo constraints (hits used to be dropped), the
    /// supporting-hit replacement, a long constrained query (judged ranking), and case-duplicate files.
    #[test]
    fn abstain_never_changes_the_ranked_ids() {
        let _guard = env_lock();
        let store = chat_store();
        store.insert("ops/port.md", "port", "daemon port 2219 runbook for the gateway", "vault").unwrap();
        store.insert("ops/port-real.md", "port", "recall daemon listens on port 8784", "vault").unwrap();
        store.insert("ops/Kestra.md", "Kestra", "kestra password rotation runbook", "vault").unwrap();
        store.insert("ops/kestra.md", "kestra", "kestra password rotation runbook steps", "vault").unwrap();
        store.insert("misc/gmail.md", "gmail", "gmail smtp configuration and password", "vault").unwrap();
        // a LONG CONSTRAINED query: the number sits inside the judge's first 64 tokens, then 200 filler words
        let long = format!("daemon port 8784 {}", vec!["melanie camping beach museum garden violin"; 34].join(" "));
        assert!(crate::store::exceeds_judge_cap(&long));
        // conflicting case-duplicate snapshots: the first-ranked copy lacks a term the other holds
        store.insert("lib/agents.md", "gmail password gmail password", "gmail password gmail password", "vault").unwrap();
        store.insert("lib/AGENTS.md", "Agents", &format!("gmail password rotation {}", "filler ".repeat(80)), "vault").unwrap();
        let queries = [
            "daemon port 8784", "recall daemon port 2219", "kestrb password", "kestra password rotation runbook",
            "gmail password rotation runbook", "What did Melanie paint at the museum?", "melanie camping zorblat",
            "camping beach adoption", long.as_str(),
        ];
        let ids = |q: &str, abstain: &str, policy: &str, top: usize, ch: &str| {
            std::env::set_var(ABSTAIN_ENV, abstain);
            std::env::set_var(ABSTAIN_POLICY_ENV, policy);
            let h = fused_sourced(&store, None, None, q, top, &[ch.to_string()]).unwrap();
            h.into_iter().map(|(h, _)| h.id).collect::<Vec<i64>>()
        };
        let mut cases: Vec<(&str, usize, &str)> = queries.iter().flat_map(|q| [(*q, 5, "fts5"), (*q, 1, "fts5")]).collect();
        cases.extend([("gmail password rotation", 1, "fts5"), ("zorblat cat luna", 5, "qwen3"), ("daemon port 8784", 3, "qwen3")]);
        // tenancy v1: the same invariant for a SCOPED reader (the scope filter runs before fusion and the
        // judge). Two notes are moved out of alice's view; she must get identical ids with abstain on and off.
        // bob's private copies of two notes (scope is immutable once written, so they are inserted scoped)
        store.insert_scoped("bob/port-real.md", "port", "recall daemon listens on port 8784 bob", "vault", "agent:bob", Some("bob")).unwrap();
        store.insert_scoped("bob/agents.md", "Agents", &format!("gmail password rotation {}", "bob ".repeat(80)), "vault", "agent:bob", Some("bob")).unwrap();
        let readers = [crate::scope::Reader::Operator, crate::scope::resolve(Some("alice".into()), None, None).unwrap()];
        let mut checked = 0;
        for reader in readers {
            let prev = crate::scope::set(reader.clone());
        for (q, top, ch) in cases.clone() {
            let off = ids(q, "off", "label", top, ch);
            if !reader.is_operator() {
                checked += 1;
            }
            for policy in ["label", "refuse"] {
                let on = ids(q, "on", policy, top, ch);
                if !on.is_empty() && !off.is_empty() {
                    assert_eq!(on, off, "{q:?} top {top} {ch} ({policy}, {reader:?}): abstain changed the ranked ids");
                }
                if policy == "label" {
                    assert_eq!(on.is_empty(), off.is_empty(), "{q:?}: label policy must not empty a non-empty list");
                }
            }
        }
            crate::scope::set(prev);
        }
        assert!(checked > 0, "the scoped reader was exercised");
        // fixture check: the scope filter really hides bob's notes from alice (else this proves nothing)
        let bob_seen = |reader: crate::scope::Reader| {
            let prev = crate::scope::set(reader);
            std::env::set_var(ABSTAIN_ENV, "off");
            let h = fused_sourced(&store, None, None, "daemon port 8784", 10, &["fts5".into()]).unwrap();
            crate::scope::set(prev);
            h.iter().any(|(h, _)| h.path.starts_with("bob/"))
        };
        assert!(bob_seen(crate::scope::Reader::Operator));
        assert!(!bob_seen(crate::scope::resolve(Some("alice".into()), None, None).unwrap()));
        // PR #29 cold view: on a daemon thread a scoped reader with no stored calibration is judged strict
        // (refuse) or labelled (label). Either way the ids that do come back equal abstain-off.
        let (q, _rx) = CalibrationQueue::new();
        std::thread::scope(|sc| {
            sc.spawn(move || {
                let st = chat_store();
                st.insert("ops/port.md", "port", "daemon port 2219 runbook for the gateway", "vault").unwrap();
                let _defer = defer_calibration_to(q.clone());
                crate::scope::set(crate::scope::resolve(Some("carol".into()), None, None).unwrap());
                let ids = |qq: &str, abstain: &str, policy: &str| {
                    std::env::set_var(ABSTAIN_ENV, abstain);
                    std::env::set_var(ABSTAIN_POLICY_ENV, policy);
                    fused_sourced(&st, None, None, qq, 5, &["fts5".into()]).unwrap().into_iter().map(|(h, _)| h.id).collect::<Vec<i64>>()
                };
                for qq in ["melanie camping beach adoption zorblat", "daemon port 8784", "camping beach adoption"] {
                    let off = ids(qq, "off", "label");
                    assert_eq!(ids(qq, "on", "label"), off, "{qq:?}: cold view, label policy");
                    let refuse = ids(qq, "on", "refuse");
                    assert!(refuse.is_empty() || refuse == off, "{qq:?}: cold view, refuse policy");
                }
            });
        });
    }

    #[test]
    fn a_tiny_store_never_calibrates() {
        let _guard = env_lock();
        let store = small_memory_store();
        assert!(phantom_queries(&store, &store.searchable_snapshot().unwrap().1, 1).unwrap().is_empty());
        assert!(phantom_threshold(&phantom_null(&store).unwrap(), 0.3).is_none());
    }

    #[test]
    fn phantom_threshold_is_the_quantile_of_the_null() {
        let null: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        assert_eq!(phantom_threshold(&null, 0.3), Some(0.3));
        assert_eq!(phantom_threshold(&null, 1.0), Some(0.99));
        assert_eq!(phantom_threshold(&null[..PHANTOM_MIN_PROBES - 1], 0.3), None);
    }

    #[test]
    fn a_calibrated_keyword_query_is_a_closest_match_and_phantom_nonsense_is_not() {
        let _guard = env_lock();
        let store = chat_store();
        // t00 = "Melanie: camping beach adoption today". An unknown word fails the lexical rule; the
        // rest is one note's whole content, so its score is mass / (mass + ln 5001) ~ 0.74, above the
        // threshold here (asserted, not assumed).
        let q = "melanie camping beach adoption zorblat";
        let thr = phantom_threshold(&phantom_null(&store).unwrap(), closest_phantom_q()).unwrap();
        let j = Judge::build(&store, q).unwrap();
        let ev = j.evidences(&store, &store.fts_judge_ids(q, LEXICAL_JUDGE_DEPTH).unwrap()).unwrap();
        assert!(closest_score(&j, &ev) > thr && closest_score(&j, &ev) < 1.0, "{} vs {thr}", closest_score(&j, &ev));
        let hits = fts5_sourced(&store, q);
        assert!(!hits.is_empty() && hits.iter().all(|(_, p)| p.closest_match), "{hits:?}");
        assert!(hits[0].1.lacks.contains(&"zorblat".to_string()), "{:?}", hits[0].1);
        // a keyword phantom at or below the threshold abstains, as before
        let ids = store.searchable_snapshot().unwrap().1;
        let weak = phantom_queries(&store, &ids, 7).unwrap().into_iter().find(|q| {
            let j = Judge::build(&store, q).unwrap();
            let ev = j.evidences(&store, &store.fts_judge_ids(q, LEXICAL_JUDGE_DEPTH).unwrap()).unwrap();
            !q.starts_with("What") && q.split(' ').count() <= 4 && closest_score(&j, &ev) <= thr
        });
        let weak = weak.expect("some keyword phantom sits at or below the threshold");
        assert!(fts5_only(&store, &weak).is_empty(), "{weak}");
        assert!(fts5_only(&store, "qxzvbnwplkjh vandrelq quexim").is_empty(), "unknown words still abstain");
    }

    #[test]
    fn a_closest_match_labels_a_number_no_document_holds() {
        let _guard = env_lock();
        let store = small_memory_store();
        let hits = fts5_sourced(&store, "What did my cat luna do in 2023?");
        assert_eq!(hits.len(), 1, "a date no note holds labels the hit instead of emptying the list");
        assert!(hits[0].1.closest_match && hits[0].1.lacks.contains(&"2023".to_string()), "{:?}", hits[0].1);
        // a number SOME note holds still filters
        assert!(fts5_sourced(&store, "What did the kestra upgrade cost in 400 or 999 dollars?")
            .iter().all(|(h, _)| h.path == "s1.md"));
    }

    #[test]
    fn convex_merge_weights_normalised_raw_scores() {
        let ranked = vec![
            Ranked { channel: "fts5".into(), ids: vec![1, 2, 3] },
            Ranked { channel: "bge".into(), ids: vec![3, 2, 4] },
        ];
        let mut raw: HashMap<String, HashMap<i64, f64>> = HashMap::new();
        raw.insert("fts5".into(), [(1, 12.0), (2, 11.0), (3, 2.0)].into_iter().collect());
        raw.insert("bge".into(), [(3, 0.80), (2, 0.79), (4, 0.60)].into_iter().collect());
        // fts5 norm: 1 -> 1, 2 -> 0.9, 3 -> 0; bge norm: 3 -> 1, 2 -> 0.95, 4 -> 0
        // w=0.6: 1 = .4, 2 = .36 + .57 = .93, 3 = .6, 4 = 0
        assert_eq!(convex_merge(&ranked, &raw, 4, 0.6).unwrap(), vec![2, 3, 1, 4]);
        // w=0: fts5 order, the bge-only doc last; w=1: bge order
        assert_eq!(convex_merge(&ranked, &raw, 4, 0.0).unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(convex_merge(&ranked, &raw, 2, 1.0).unwrap(), vec![3, 2]);
        // a channel set other than exactly fts5+bge, or a missing raw score, falls back (None)
        assert!(convex_merge(&ranked[..1], &raw, 4, 0.6).is_none());
        raw.get_mut("bge").unwrap().remove(&4);
        assert!(convex_merge(&ranked, &raw, 4, 0.6).is_none());
        // a finite but overflowing score range (hi - lo = inf) also falls back instead of sorting NaNs
        raw.get_mut("bge").unwrap().insert(4, 0.6);
        raw.get_mut("fts5").unwrap().insert(3, -1e308);
        raw.get_mut("fts5").unwrap().insert(1, 1e308);
        assert!(convex_merge(&ranked, &raw, 4, 0.6).is_none());
    }

    /// The UI and the CLI both sort fts5+bge by convex fusion by default, NOT by the RRF sum they
    /// also report (`Provenance.rrf`): the ids below rank 3 first under RRF but 2 first under convex.
    #[test]
    fn default_fusion_orders_by_convex_not_by_the_reported_rrf_sum() {
        let _guard = env_lock();
        std::env::remove_var(FUSION_MODE_ENV);
        std::env::remove_var(DENOISE_MODE_ENV);
        let ranked = vec![
            Ranked { channel: "fts5".into(), ids: vec![1, 2, 3] },
            Ranked { channel: "bge".into(), ids: vec![3, 2, 4] },
        ];
        let mut raw: HashMap<String, HashMap<i64, f64>> = HashMap::new();
        raw.insert("fts5".into(), [(1, 12.0), (2, 11.0), (3, 2.0)].into_iter().collect());
        raw.insert("bge".into(), [(3, 0.80), (2, 0.79), (4, 0.60)].into_iter().collect());
        let by_rrf = rrf_merge_with_k(&ranked, 4, 60.0);
        assert_eq!(by_rrf[0], 3, "RRF sum puts 3 first");
        assert_eq!(fuse_with_scores(&ranked, &raw, 4, &HashMap::new(), None)[0], 2, "default fusion is convex");
        assert_eq!(effective_fusion_mode(), "convex");
    }

    /// The 3-note guided test (cli-quality, 2026-11-12): api / backup / keys notes.
    fn three_note_store() -> Store {
        let store = in_memory_store();
        store.insert("api.md", "API notes", "# API notes
The API port is 8080.", "vault").unwrap();
        store.insert("backup.md", "Backups", "# Backups
We back up the database nightly to B2.", "vault").unwrap();
        store.insert("keys.md", "Keys", "# Keys
My email is test.person@example.com", "vault").unwrap();
        store
    }

    /// Bug 1: function words are not subjects. Over notes that hold none of keep / copies / data, the judge
    /// fell back to every raw token and called "how", "do", "of", "our" missing subjects.
    #[test]
    fn function_words_are_never_judged_or_labelled_as_missing_subjects() {
        let _guard = env_lock();
        std::env::remove_var(ABSTAIN_POLICY_ENV);
        let store = three_note_store();
        let q = "how do we keep copies of our data";
        assert_eq!(Judge::build(&store, q).unwrap().unknown, vec!["keep", "copies", "data"]);
        let out = fts5_sourced(&store, q);
        assert_eq!(out.first().map(|(h, _)| h.path.as_str()), Some("backup.md"), "{out:?}");
        for (_, p) in &out {
            for w in ["how", "do", "we", "of", "our"] {
                assert!(!p.lacks.iter().any(|l| l == w), "{w} labelled: {:?}", p.lacks);
            }
        }
        // the raw-token fallback still judges a short subject when every content word is absent ("AI setpu")
        assert!(judged_terms(&store, "AI setpu").unwrap().contains(&"ai".to_string()));
    }

    /// Bug 3: an unknown-term reason must not read as "nothing matched" next to a hit that came via fts5.
    #[test]
    fn an_unknown_term_reason_says_the_keyword_hits_matched_other_words() {
        let _guard = env_lock();
        let store = three_note_store();
        let q = "how do we keep copies of our data";
        let j = Judge::build(&store, q).unwrap();
        let ids = store.fts_judge_ids(q, LEXICAL_JUDGE_DEPTH).unwrap();
        assert!(!ids.is_empty(), "fts5 does return a hit (it shares \"we\")");
        let why = lexical_judgement(&j, &j.evidences(&store, &ids).unwrap()).unwrap();
        assert!(why.contains("[keep, copies, data]") && why.contains("keyword hits match only other words"), "{why}");
        assert!(!why.contains("no searchable document"), "{why}");
    }

    /// Bugs 2 and 6: a CONFIDENT list never calls a hit a closest match; its top hit is never tagged, a lower
    /// hit lacking a subject term is muted "(weak)". Only a weak list says "closest match".
    #[test]
    fn only_a_weak_list_says_closest_match_and_the_top_of_a_confident_list_is_untagged() {
        let _guard = env_lock();
        std::env::remove_var(ABSTAIN_POLICY_ENV);
        let store = three_note_store();
        let out = fts5_sourced(&store, "API port");
        assert_eq!(out[0].0.path, "api.md");
        assert!(!out[0].1.closest_match && out[0].1.lacks.is_empty(), "{:?}", out[0].1);
        let lacks = vec!["api".to_string(), "port".to_string()];
        assert_eq!(hit_note(0, false, &[]), (false, None));
        assert_eq!(hit_note(0, false, &lacks), (false, Some("does not mention api, port".into())));
        assert_eq!(hit_note(1, false, &lacks), (true, Some("does not mention api, port".into())));
        assert_eq!(hit_note(1, false, &[]), (false, None));
        let (weak, note) = hit_note(0, true, &lacks);
        assert!(!weak && note.unwrap().contains("closest match"));
    }

    /// Addendum B: the plain line names the main words NO note holds (df 0), never a function word, and is not
    /// printed when every main word occurs somewhere.
    #[test]
    fn the_absent_line_names_words_no_note_holds_and_never_a_function_word() {
        let _guard = env_lock();
        std::env::remove_var(ABSTAIN_POLICY_ENV);
        let store = three_note_store();
        let out = fts5_sourced(&store, "how do we keep copies of our data");
        assert!(!out.is_empty() && out.iter().all(|(_, p)| p.absent == ["keep", "copies", "data"]), "{out:?}");
        assert_eq!(
            absent_line(&out[0].1.absent).unwrap(),
            "Not found in the indexed text: keep, copies, data. These are the nearest matches."
        );
        // a main word some note holds is never listed: only the absent one is
        let out = fts5_sourced(&store, "database kubernetes");
        assert!(!out.is_empty() && out.iter().all(|(_, p)| p.absent == ["kubernetes"]), "{out:?}");
        // every main word occurs somewhere: no line
        let out = fts5_sourced(&store, "API port");
        assert!(out.iter().all(|(_, p)| p.absent.is_empty()) && absent_line(&[]).is_none(), "{out:?}");
        // "how do": a function-only query falls back to judging function words, but never labels or lists them
        store.insert("how.md", "Howto", "# Howto
how it works", "vault").unwrap();
        let fo = fts5_sourced(&store, "how do");
        assert!(!fo.is_empty(), "fts5 answers through \"how\"");
        for (_, p) in fo {
            assert!(!p.lacks.iter().any(|l| l == "how" || l == "do"), "{:?}", p.lacks);
            assert!(p.absent.is_empty(), "{:?}", p.absent);
        }
    }

    /// The pre-registered `agree` rule: first in BOTH channels, nothing else counts.
    #[test]
    fn agree_needs_rank_one_in_both_fts5_and_bge() {
        let r = |v: &[(&str, usize)]| v.iter().map(|(k, n)| (k.to_string(), *n)).collect::<std::collections::BTreeMap<_, _>>();
        assert!(top_hit_agrees(&r(&[("fts5", 0), ("bge", 0)])));
        assert!(!top_hit_agrees(&r(&[("fts5", 0), ("bge", 1)])));
        assert!(!top_hit_agrees(&r(&[("bge", 0)])));
        assert!(!top_hit_agrees(&r(&[("fts5-rescue", 0), ("bge", 0)])));
    }
}
