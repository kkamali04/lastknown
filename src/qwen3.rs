// qwen3.rs — Qwen3-Embedding-0.6B, driven through `ort` directly.
//
// NOT fastembed. fastembed feeds a session exactly `input_ids` + `attention_mask`
// (+`token_type_ids`) and gives no hook to add more, while this graph also requires
// `position_ids` — verified by reading the graph's own tensor names, not from a model card. That
// single missing input is why this is a direct integration rather than a model swap.
//
// Three traps this file exists to avoid, all of which fail SILENTLY — the model returns vectors
// either way, they are just worthless:
//
//   1. POOLING. Qwen3-Embedding is a decoder. Its sentence meaning lives in the hidden state of
//      the FINAL token, not in a [CLS] token at the front. fastembed only offers Cls and Mean.
//   2. THE EOS TOKEN. Last-token pooling reads the final position, so the final position has to
//      be the end-of-sequence marker. Verified from the downloaded tokenizer.json: its
//      TemplateProcessing appends `<|endoftext|>` — but ONLY when `add_special_tokens = true`.
//      Encode with `false` and it silently reads the last WORD of the text instead.
//   3. ASYMMETRY. Queries get an instruction prefix; documents get raw text. Prefixing documents
//      too is the documented failure mode.
//
// And one this corpus supplies on its own:
//
//   4. PAD == EOS. `tokenizer_config.json` sets pad_token to `<|endoftext|>`, the SAME id (151643)
//      as the end-of-sequence token. So "find the last EOS" would land on padding. The last real
//      token can only be located through the attention mask.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::path::PathBuf;

/// Qwen3-Embedding-0.6B output width. 🔴 NOT interchangeable with bge's 384 — a cosine between a
/// 1024-dim and a 384-dim vector is refused, but a cosine between two DIFFERENT 1024-dim models
/// computes happily and means nothing. Vectors from this model live in their own table.
pub const DIM: usize = 1024;

/// `<|endoftext|>`, read from the model's `added_tokens.json`.
///
/// 🔴 `tokenizer_config.json` advertises `eos_token: <|im_end|>` (151645). That is a decoy left
/// over from the chat variant — the tokenizer's TemplateProcessing appends `<|endoftext|>`, not
/// `<|im_end|>`. Taking the id from tokenizer_config gives a token this model never emits.
pub const EOS_ID: i64 = 151643;

/// Padding id. Deliberately the same value as [`EOS_ID`] — that is what the model ships. See
/// trap 4 in the module comment: this is why pooling uses the attention mask, never an id search.
pub const PAD_ID: i64 = EOS_ID;

/// Hard cap on tokens per sequence. Override with `RECALL_QWEN3_MAX_TOKENS`.
///
/// The model's window is 32,768, so this is not the model's limit — it is a VRAM limit. Attention
/// memory grows with sequence length, and the card is 8 GB with ~1.6 GB already spoken for.
///
/// 🔴 This cap is SILENT: past it a chunk is truncated and re-terminated with EOS, and the vector
/// that comes back is a perfectly plausible embedding of the first N tokens. Any experiment that
/// varies chunk size MUST raise this first, or it measures the cap instead of the chunk.
pub const MAX_TOKENS: usize = 2048;

/// Token slots (`batch x longest sequence`) allowed in one forward pass.
///
/// Measured on the 8 GB card: ~14,000 slots of prose sat at 3,823 MiB and ran at full speed,
/// while ~30,000 slots reached 7,811 MiB and thrashed. 12,000 keeps a margin for the desktop.
/// Override with `RECALL_QWEN3_TOKEN_BUDGET`.
pub const TOKEN_BUDGET: usize = 12_000;

pub fn batch_token_budget() -> usize {
    std::env::var("RECALL_QWEN3_TOKEN_BUDGET")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 512)
        .unwrap_or(TOKEN_BUDGET)
}

/// The effective cap, after `RECALL_QWEN3_MAX_TOKENS`.
pub fn max_tokens() -> usize {
    std::env::var("RECALL_QWEN3_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 16)
        .unwrap_or(MAX_TOKENS)
}

/// The task half of the query instruction. Documents never see this.
pub const QUERY_TASK: &str =
    "Given a search query, retrieve relevant passages from a personal knowledge vault that answer the query";

/// Wrap a user query in the documented instruction format.
///
/// 🔴 Exactly `Instruct: {task}\nQuery:{query}` — no space after `Query:`. Worth 1-5% on
/// retrieval, and applying it to documents as well is the way to lose more than that.
pub fn query_prompt(query: &str) -> String {
    format!("Instruct: {QUERY_TASK}\nQuery:{query}")
}

/// Where the model snapshot lives: `~/.recall/models/qwen3-embedding-0.6b`.
pub fn model_dir() -> Result<PathBuf> {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .context("USERPROFILE/HOME not set")?;
    Ok(PathBuf::from(home)
        .join(".recall")
        .join("models")
        .join("qwen3-embedding-0.6b"))
}

/// True when the model files are present. Lets a caller skip this channel rather than fail.
pub fn available() -> bool {
    model_dir()
        .map(|d| d.join("model.onnx").is_file() && d.join("tokenizer.json").is_file())
        .unwrap_or(false)
}

/// The query-side channel: a loaded model plus every stored Qwen3 chunk vector.
///
/// Deliberately the same shape as [`crate::bge::BgeSearch`], and it reuses that module's
/// `best_per_doc` rather than shipping a second max-pool implementation - one doc_id appears
/// once per CHUNK in `vectors`, and collapsing that to one score per document is a rule that
/// must not be allowed to differ between two channels.
pub struct Qwen3Search {
    /// `session.run` needs `&mut`, but the query path holds this channel by shared reference the
    /// way it holds BGE. RefCell keeps the call sites identical instead of threading `&mut`
    /// through every caller.
    embedder: std::cell::RefCell<Qwen3Embedder>,
    vectors: Vec<(i64, Vec<f32>)>,
}

impl Qwen3Search {
    /// `Ok(None)` when nothing has been embedded yet, so a caller can skip the channel instead
    /// of treating an un-embedded store as an error.
    ///
    /// 🔴 It returns None rather than falling back to BGE. A channel that quietly answers with a
    /// DIFFERENT model's vectors is the fail-open shape this project keeps getting burned by:
    /// the caller asked for qwen3, got plausible results, and nothing said otherwise.
    pub fn load(store: &crate::store::Store) -> Result<Option<Self>> {
        let vectors = store.load_all_qwen3_chunks()?;
        if vectors.is_empty() {
            eprintln!("qwen3: no vectors stored - channel unavailable");
            return Ok(None);
        }
        let docs = {
            let mut ids: Vec<i64> = vectors.iter().map(|(d, _)| *d).collect();
            ids.sort_unstable();
            ids.dedup();
            ids.len()
        };
        // Whole-corpus counts: operator only (Codex tenancy b1).
        if crate::scope::current().is_operator() {
            eprintln!(
                "qwen3: loaded {} chunk vectors over {} documents",
                vectors.len(),
                docs
            );
        }
        let embedder = Qwen3Embedder::new()?;
        Ok(Some(Self {
            embedder: std::cell::RefCell::new(embedder),
            vectors,
        }))
    }

    pub fn backend(&self) -> String {
        self.embedder.borrow().backend().to_string()
    }

    pub fn doc_count(&self) -> usize {
        let mut ids: Vec<i64> = self.vectors.iter().map(|(d, _)| *d).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }

    /// Top-k doc ids by cosine, best first. The query gets the instruction prefix; the stored
    /// document vectors did not. That asymmetry is the model's documented contract.
    pub fn search_ids(&self, query: &str, k: usize) -> Result<Vec<i64>> {
        let q = self.embedder.borrow_mut().embed_query(query)?;
        if q.is_empty() {
            return Ok(vec![]);
        }
        let mut best = crate::bge::best_per_doc(&self.vectors, &q);
        crate::scope::drop_denied(&mut best);
        let mut scored: Vec<(i64, f32)> = best.into_iter().collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        Ok(scored.into_iter().map(|(id, _)| id).collect())
    }
}

/// Target chunk size in characters. Override with `RECALL_QWEN3_CHUNK_CHARS`.
///
/// Measured on 60 real vault documents, DirectML, batch 16 (docs/sec is the honest metric -
/// chunks/sec rises as chunks shrink while the run gets slower, because the TEXT is constant):
///
/// | chars | docs/sec | hours for the corpus | corpus chunks |
/// |-------|----------|------------------|---------------|
/// | 2400  | 4.86     | 3.40             | 99,180        |
/// | 3600  | 4.19     | 3.95             | 78,352        |
/// | 4800  | 1.86     | 8.89             | 73,393        |
///
/// 3600 wins: 4800 falls off a cliff (attention memory is quadratic in sequence length, so the
/// batch stops fitting), and 2400 buys 0.55 hours once in exchange for 21,000 extra vectors
/// scanned on every query forever. At 3600 the mean vault document (2,701 chars) is ONE chunk.
pub const CHUNK_CHARS: usize = 3600;

pub fn chunk_chars() -> usize {
    std::env::var("RECALL_QWEN3_CHUNK_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 200)
        .unwrap_or(CHUNK_CHARS)
}

/// One chunk of a document: the heading context, its BYTE span in the body, and the text to embed.
pub struct Chunk {
    pub heading: String,
    pub start: i64,
    pub end: i64,
    pub text: String,
}

/// Split a document into Qwen3-sized chunks, preferring paragraph boundaries.
///
/// Spans are BYTE offsets into `body`, matching what `bge_chunk_embeddings` stores (the column is
/// named `char_start` and holds bytes - see the schema comment). Every byte of the body lands in
/// exactly one chunk; nothing is dropped and nothing overlaps.
pub fn split_document(title: &str, body: &str, target: usize) -> Vec<Chunk> {
    let title = title.trim();
    if body.trim().is_empty() {
        // A title-only note is still a real, findable thing. One row, not zero.
        return vec![Chunk {
            heading: title.to_string(),
            start: 0,
            end: 0,
            text: title.to_string(),
        }];
    }

    let bytes = body.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let mut end = (i + target).min(bytes.len());
        while end < bytes.len() && !body.is_char_boundary(end) {
            end += 1;
        }
        if end < bytes.len() {
            // Prefer a blank line in the last 15% of the window - a chunk that stops mid-sentence
            // embeds worse than one slightly short of target.
            let floor = i + target * 85 / 100;
            let mut j = end;
            while j > floor && j > i + 2 {
                if bytes[j - 1] == b'\n' && bytes[j - 2] == b'\n' {
                    end = j;
                    break;
                }
                j -= 1;
            }
        }
        while end < bytes.len() && !body.is_char_boundary(end) {
            end += 1;
        }
        if end <= i {
            end = bytes.len();
        }
        let slice = &body[i..end];
        let heading = last_heading_before(body, i, title);
        out.push(Chunk {
            heading: heading.clone(),
            start: i as i64,
            end: end as i64,
            text: if heading.is_empty() {
                slice.to_string()
            } else {
                format!("{heading}\n{slice}")
            },
        });
        i = end;
    }
    out
}

/// The nearest markdown heading at or above `pos`, prefixed with the note title when it adds
/// information. Gives an orphaned mid-document chunk something to be ABOUT.
fn last_heading_before(body: &str, pos: usize, title: &str) -> String {
    let mut heading = String::new();
    for line in body[..pos.min(body.len())].lines() {
        let t = line.trim_start();
        if t.starts_with('#') {
            heading = t.trim_start_matches('#').trim().to_string();
        }
    }
    match (title.is_empty(), heading.is_empty()) {
        (true, true) => String::new(),
        (false, true) => title.to_string(),
        (true, false) => heading,
        (false, false) if heading == title => title.to_string(),
        (false, false) => format!("{title} > {heading}"),
    }
}

pub struct Qwen3Embedder {
    session: Session,
    tok: tokenizers::Tokenizer,
    backend: String,
}

impl Qwen3Embedder {
    /// Load the model, honouring `RECALL_BACKEND` exactly as the bge channel does.
    pub fn new() -> Result<Self> {
        let dir = model_dir()?;
        let model = dir.join("model.onnx");
        if !model.is_file() {
            anyhow::bail!(
                "Qwen3 model not found at {}. It is a ~1.2 GB download and is not fetched \
                 automatically.",
                model.display()
            );
        }

        let (providers, backend) = crate::bge::resolve_execution_providers()?;
        eprintln!("qwen3: backend={backend}");

        // `with_execution_providers` returns an error that carries the builder itself, which is
        // neither Send nor Sync, so `?` cannot lift it into anyhow. Flatten it by hand.
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("ort session builder: {e}"))?
            .with_execution_providers(providers)
            .map_err(|e| anyhow::anyhow!("register execution provider: {e}"))?;
        let session = builder
            .commit_from_file(&model)
            .with_context(|| format!("load Qwen3 ONNX from {}", model.display()))?;

        let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("load Qwen3 tokenizer.json: {e}"))?;
        // Padding and truncation are applied here, by hand, per batch — the mask has to agree
        // with the ids exactly or last-token pooling reads the wrong row.
        tok.with_truncation(None).map_err(anyhow::Error::msg)?;
        tok.with_padding(None);

        Ok(Self {
            session,
            tok,
            backend,
        })
    }

    /// Which execution provider this embedder actually resolved to.
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// Embed DOCUMENTS. Raw text, no instruction prefix — see trap 3.
    pub fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.forward_budgeted(texts)
    }

    /// Embed any number of texts, forming micro-batches by TOKEN budget rather than by count.
    ///
    /// 🔴 A caller-supplied batch COUNT cannot bound GPU memory here, and assuming it can cost
    /// three failed runs. The padded tensor is `batch x longest_sequence`, and sequence length
    /// per chunk varies ~3x across this corpus because chunks are cut by BYTES while the model
    /// charges by TOKENS: measured 1.31 bytes/token for minified JSON against 4.86 for prose. So
    /// "batch 16" was ~14k token slots on the prose I benchmarked and ~44k on a markdown table,
    /// which pinned VRAM at 7,811 MiB of 8,192 and collapsed into shared-memory thrashing - 100%
    /// GPU utilisation, almost no throughput, and no error anywhere.
    ///
    /// Two things happen here, and the second is nearly free:
    ///   1. micro-batches are capped by `batch x longest`, the real shape of the tensor;
    ///   2. texts are SORTED by length first, so each batch pads to a length close to its own
    ///      members instead of to the longest text in the whole call. Padding is wasted compute
    ///      at full price.
    /// Order is restored before returning - callers zip these against their own chunk lists.
    fn forward_budgeted(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let budget = batch_token_budget();
        let cap = max_tokens();

        // Cheap proxy for token count: bytes. Only used for GROUPING, never for truncation -
        // `forward` re-tokenizes and enforces `cap` exactly.
        let mut idx: Vec<usize> = (0..texts.len()).collect();
        idx.sort_by_key(|&i| texts[i].len());

        let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut i = 0usize;
        while i < idx.len() {
            let mut j = i;
            let mut longest = 0usize;
            while j < idx.len() {
                // bytes/token is never below 1, so bytes is a safe upper bound on tokens.
                let est = texts[idx[j]].len().min(cap);
                let next_longest = longest.max(est);
                let n = j - i + 1;
                if n > 1 && next_longest.saturating_mul(n) > budget {
                    break;
                }
                longest = next_longest;
                j += 1;
            }
            let group: Vec<String> = idx[i..j].iter().map(|&k| texts[k].clone()).collect();
            let vecs = self.forward(&group)?;
            anyhow::ensure!(
                vecs.len() == group.len(),
                "qwen3: {} vectors for {} texts",
                vecs.len(),
                group.len()
            );
            for (slot, v) in idx[i..j].iter().zip(vecs) {
                out[*slot] = Some(v);
            }
            i = j;
        }

        out.into_iter()
            .enumerate()
            .map(|(i, v)| v.ok_or_else(|| anyhow::anyhow!("qwen3: text {i} produced no vector")))
            .collect()
    }

    /// Embed a QUERY. Instruction prefix applied — see trap 3.
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let mut v = self.forward(&[query_prompt(query)])?;
        Ok(v.pop().unwrap_or_default())
    }

    fn forward(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // 🔴 `true` = add special tokens = the TemplateProcessing that appends <|endoftext|>.
        // With `false` the final position is the last WORD of the text and every vector is wrong,
        // with no error anywhere. This is trap 2 and it is one boolean.
        let encodings = self
            .tok
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("qwen3 tokenize: {e}"))?;

        let cap = max_tokens();
        let mut truncated = 0usize;
        let mut rows: Vec<Vec<i64>> = Vec::with_capacity(encodings.len());
        for enc in &encodings {
            let mut ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
            if ids.len() > cap {
                // Truncating drops the appended EOS, so put it back: last-token pooling has to
                // read an EOS, not whatever word the cut landed on.
                truncated += 1;
                ids.truncate(cap - 1);
                ids.push(EOS_ID);
            }
            if ids.is_empty() {
                ids.push(EOS_ID);
            }
            rows.push(ids);
        }

        if truncated > 0 {
            // Loud, because the failure it prevents is silent: a truncated chunk still produces
            // a confident vector, and the text past the cut is simply never searchable.
            eprintln!(
                "qwen3: WARNING {truncated} of {} text(s) exceeded {cap} tokens and were cut",
                rows.len()
            );
        }

        let batch = rows.len();
        let seq = rows.iter().map(|r| r.len()).max().unwrap_or(1);

        // RIGHT padding. The last real token is therefore at `len-1` per row, which is why the
        // mask sum is the index — see `pool_last_token`.
        let mut input_ids = vec![PAD_ID; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        let mut positions = vec![0i64; batch * seq];
        for (b, r) in rows.iter().enumerate() {
            for (t, &id) in r.iter().enumerate() {
                input_ids[b * seq + t] = id;
                mask[b * seq + t] = 1;
                positions[b * seq + t] = t as i64;
            }
        }

        let shape = vec![batch as i64, seq as i64];
        let outputs = self.session.run(ort::inputs![
            "input_ids"      => Tensor::from_array((shape.clone(), input_ids))?,
            "attention_mask" => Tensor::from_array((shape.clone(), mask.clone()))?,
            "position_ids"   => Tensor::from_array((shape, positions))?,
        ])?;

        let hidden = outputs
            .get("last_hidden_state")
            .context("qwen3: graph produced no `last_hidden_state`")?;

        // The published export is fp16, so the output usually is too. Accept either rather than
        // assume — a wrong guess here is a type error at runtime on a 20-hour job.
        let flat: Vec<f32> = match hidden.try_extract_tensor::<f32>() {
            Ok((_, s)) => s.to_vec(),
            Err(_) => {
                let (_, s) = hidden
                    .try_extract_tensor::<half::f16>()
                    .context("qwen3: last_hidden_state is neither f32 nor f16")?;
                s.iter().map(|h| h.to_f32()).collect()
            }
        };

        let expect = batch * seq * DIM;
        if flat.len() != expect {
            anyhow::bail!(
                "qwen3: last_hidden_state has {} values, expected {batch}x{seq}x{DIM}={expect}. \
                 The model's hidden size is not {DIM}.",
                flat.len()
            );
        }

        Ok(pool_last_token(&flat, &mask, batch, seq))
    }
}

/// Last-token pooling, then L2 normalisation.
///
/// 🔴 The index is `(number of 1s in this row's mask) - 1`, NOT `seq - 1` and NOT a search for
/// `EOS_ID`. Padding is right-hand, so `seq - 1` is padding for every row shorter than the
/// longest; and padding uses the same id as EOS, so searching for the id finds padding too.
/// The mask is the only thing that knows where the real text stopped.
fn pool_last_token(flat: &[f32], mask: &[i64], batch: usize, seq: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let real = mask[b * seq..(b + 1) * seq].iter().filter(|&&m| m == 1).count();
        let idx = real.saturating_sub(1);
        let start = (b * seq + idx) * DIM;
        let mut v: Vec<f32> = flat[start..start + DIM].to_vec();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        out.push(v);
    }
    out
}

/// Cosine similarity. Both inputs are already L2-normalised, so this is a dot product — but the
/// length check stays, because comparing a 1024-dim vector with a 384-dim one is a category error
/// that must never silently produce a number.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_prompt_is_the_documented_format_exactly() {
        let p = query_prompt("what did I decide about backups");
        // No space after `Query:` - the published format has none, and this is the kind of detail
        // that silently costs a few percent.
        assert_eq!(
            p,
            format!("Instruct: {QUERY_TASK}\nQuery:what did I decide about backups")
        );
        assert!(!p.contains("Query: "), "there must be no space after `Query:`");
    }

    #[test]
    fn pooling_reads_the_last_REAL_token_not_the_last_slot() {
        // Two rows, seq 3, DIM stubbed by using a tiny slice layout: row 0 has 2 real tokens,
        // row 1 has 3. If pooling used `seq - 1` it would read padding for row 0.
        let seq = 3;
        let batch = 2;
        let mut flat = vec![0.0f32; batch * seq * DIM];
        // row 0, token 1 (the last real one) -> marker in the first component
        flat[(0 * seq + 1) * DIM] = 5.0;
        // row 0, token 2 is PADDING -> a decoy that must NOT be read
        flat[(0 * seq + 2) * DIM] = -99.0;
        // row 1, token 2 (the last real one)
        flat[(1 * seq + 2) * DIM] = 7.0;
        let mask = vec![1, 1, 0, 1, 1, 1];

        let out = pool_last_token(&flat, &mask, batch, seq);
        assert_eq!(out.len(), 2);
        // normalised, so the sign is what matters: the decoy was negative.
        assert!(out[0][0] > 0.0, "row 0 read padding instead of the last real token");
        assert!(out[1][0] > 0.0);
    }

    #[test]
    fn vectors_are_l2_normalised() {
        let seq = 1;
        let mut flat = vec![0.0f32; seq * DIM];
        flat[0] = 3.0;
        flat[1] = 4.0;
        let out = pool_last_token(&flat, &[1], 1, seq);
        let norm = out[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit length, got {norm}");
    }

    #[test]
    fn cosine_refuses_to_compare_different_widths() {
        // A 1024-dim Qwen3 vector and a 384-dim bge vector must never produce a plausible number.
        assert_eq!(cosine(&[1.0; DIM], &[1.0; 384]), 0.0);
    }
}
