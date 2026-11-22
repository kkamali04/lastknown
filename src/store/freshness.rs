// store/freshness.rs — "does the file on disk still match what recall indexed?"
//
// A child module of `store` on purpose: it needs the same private helpers `ingest_dir` uses
// (`find_moved_document`, `revive_or_insert`, `path_is_absent`, `read_markdown_text`), and
// reusing them is the point — a second move-detection or tombstone implementation would drift.
//
// Columns (all on `documents`, all added by migration, all NULL on legacy rows):
//   modified_at  file mtime in unix NANOSECONDS, captured BEFORE the body was read (so a write that
//                lands during the read leaves a stale-looking stat, never a fresh-looking one)
//   file_size    bytes on disk at that moment
//   indexed_at   unix secs when recall last read this file's content
//   changed_at   unix secs a change was first noticed; NULL when fresh
//   stale        0 fresh | 1 file changed | 2 file deleted      (a HINT — see below)
// Table `fs_events(path, kind='new', seen_at)` holds created files that have no row yet.
//
// 🔴 The stored `stale` flag is a hint for the refresh queue, never the answer a query reports.
// A query re-checks the disk itself (`check_row`): stat first, SHA-256 only to confirm. A flag
// written by a watcher that was down for a day would otherwise be wrong in both directions.

use super::*;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

pub const STALE_NONE: i64 = 0;
pub const STALE_CHANGED: i64 = 1;
pub const STALE_DELETED: i64 = 2;

/// Documents tombstoned in one refresh may not exceed this share of live vault docs (floor 50).
/// A mounted-wrong or half-synced vault looks exactly like mass deletion, so a large batch is
/// left flagged and handed to `recall index --prune-missing`, which has the enumeration gate.
pub const REFRESH_MAX_DELETE_FRACTION: f64 = 0.02;
pub const REFRESH_MIN_DELETE_ALLOWANCE: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatSig {
    /// Nanoseconds since the epoch. Millisecond truncation would let two same-size edits inside
    /// one millisecond (or a tool that restores mtime to the ms) pass the stat fast path.
    pub mtime_ns: i64,
    pub size: i64,
}

const NS_PER_SEC: i64 = 1_000_000_000;

pub fn stat_sig(md: &std::fs::Metadata) -> StatSig {
    let mtime_ns = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos().min(i64::MAX as u128) as i64)
        .unwrap_or(0);
    StatSig { mtime_ns, size: md.len() as i64 }
}

pub fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// `2026-11-05 14:03 UTC` — civil-from-days (Hinnant), so no chrono dependency.
pub fn fmt_utc(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02} UTC", rem / 3_600, rem % 3_600 / 60)
}

// ── Query-time check ───────────────────────────────────────────────────────────

/// What a search result reports about its file. `stale: None` means "could not check" and carries
/// a note — absence of a flag must never read as "fresh".
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct FreshInfo {
    pub stale: Option<bool>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub deleted: bool,
    /// The file's mtime when recall indexed it (recency signal). Absent on legacy rows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexed_at: Option<String>,
    /// The file's mtime NOW; present only when the file changed after indexing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_modified_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stale_note: Option<String>,
    /// mtime (unix secs) to rank recency by: the live one when known, else the indexed one.
    #[serde(skip)]
    pub mtime_secs: Option<i64>,
}

impl FreshInfo {
    pub fn is_stale(&self) -> bool {
        self.stale == Some(true)
    }
    pub fn unchecked(why: &str) -> Self {
        Self { stale: None, stale_note: Some(format!("freshness not checked: {why}")), ..Self::default() }
    }
}

/// `modified_at` is nanoseconds since the epoch. An earlier revision of this branch stored
/// milliseconds; any value below 1e15 ns (1970-01-12) is from that era, and is treated as "no
/// recorded stat" so the row is re-hashed and re-stamped instead of being shown as a 1970 date.
fn plausible_ns(v: Option<i64>) -> Option<i64> {
    v.filter(|n| *n >= 1_000_000_000_000_000)
}

#[derive(Debug, Clone)]
pub struct FreshRow {
    pub id: i64,
    pub path: String,
    pub source: String,
    pub modified_at: Option<i64>,
    pub file_size: Option<i64>,
    pub indexed_at: Option<i64>,
    pub valid_from: String,
    pub content_sha: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum Verdict {
    /// Stat identical to the recorded one.
    Fresh,
    /// Different stat, identical bytes (a `touch`, a re-save). Carries the stat taken BEFORE the
    /// bytes were read: stamping a second, later stat would certify bytes we never hashed.
    Touched(StatSig),
    /// The file differs from the index; carries its current mtime (ns).
    Changed(i64),
    /// The file differs, but the excerpt the reader was shown still sits in it verbatim (after
    /// normalisation, see [`passage_check`]): the passage is not stale. Only produced when the
    /// caller passed the excerpt, and only from bytes the hash check already read.
    ChangedPassageKept(i64),
    /// The file differs and a line the shown excerpt sat on has changed (the excerpt's own characters may
    /// all still be there: lines are compared whole).
    ChangedPassageGone(i64),
    Deleted,
    /// Not file-backed (ad-hoc save): nothing on disk to drift from.
    NoFile,
    Unknown(String),
}

/// Compare one row to the disk.
///
/// `verify_content = false` (the bulk scan over ~70k documents): stat first, and read + SHA-256 only
/// when the stat differs. That cannot see an edit that keeps both size and mtime to the nanosecond
/// (a tool that restores timestamps); `recall index` hashes every body and does.
///
/// `verify_content = true` (a query's best hits up to [`QUERY_HASH_BUDGET_BYTES`], and single
/// watcher events): always read and hash, so what the reader is shown first is exact, not a stat
/// heuristic. Hits past the budget get the stat-first check.
pub fn check_row(root: &Path, row: &FreshRow, verify_content: bool) -> Verdict {
    check_row_budgeted(root, row, verify_content, &mut u64::MAX)
}

/// [`check_row`] that reads at most `read_budget` bytes (charged by the file's size on disk right
/// now, not the recorded one), and never reads past it: a file it may not read is judged by its stat
/// alone, an unchanged stat as fresh and a changed one as changed (the cautious answer: go read the
/// file). Codex conc r1 #5: charging the recorded size let grown or unsized files, and stat-changed
/// ones after the budget, read without limit.
pub fn check_row_budgeted(root: &Path, row: &FreshRow, verify_content: bool, read_budget: &mut u64) -> Verdict {
    check_row_excerpt(root, row, verify_content, read_budget, None, &[], &|| None)
}

/// [`check_row_budgeted`] that, when the file's hash differs and `excerpt` is given, also decides
/// whether that excerpt still exists in the current text. No extra I/O: it reuses the bytes the
/// hash check read, and only runs for a file that already failed the hash.
pub fn check_row_excerpt(
    root: &Path,
    row: &FreshRow,
    verify_content: bool,
    read_budget: &mut u64,
    excerpt: Option<&str>,
    terms: &[String],
    old_body: &dyn Fn() -> Option<String>,
) -> Verdict {
    if row.path.is_empty() || row.source != "vault" {
        return Verdict::NoFile;
    }
    let abs = match resolve_ingest_abs_path(root, &row.path) {
        Ok(p) => p,
        Err(e) => return Verdict::Unknown(format!("{e:#}")),
    };
    let md = match std::fs::metadata(&abs) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Verdict::Deleted,
        Err(e) => return Verdict::Unknown(format!("stat {}: {e}", abs.display())),
    };
    let sig = stat_sig(&md);
    let same_stat = row.modified_at == Some(sig.mtime_ns) && row.file_size == Some(sig.size);
    if same_stat && !verify_content {
        return Verdict::Fresh;
    }
    // A file it may not read is judged by its stat alone: unchanged = fresh, changed = changed.
    let unread = || if same_stat { Verdict::Fresh } else { Verdict::Changed(sig.mtime_ns) };
    if u64::try_from(sig.size).unwrap_or(0) > *read_budget {
        return unread();
    }
    let Some(recorded) = row.content_sha.as_deref() else {
        return Verdict::Unknown("row has no content hash to compare against".into());
    };
    // The stat above can be stale by the time the file is read (it grew, or was replaced), so the
    // read itself is capped and charged by what it consumed (Codex conc r2 #3).
    match read_capped(&abs, *read_budget) {
        Ok(Some(bytes)) => {
            *read_budget -= bytes.len() as u64;
            let text = super::stored_form(bytes);
            if hash_matches_modulo_line_endings(&text, recorded) {
                if same_stat { Verdict::Fresh } else { Verdict::Touched(sig) }
            } else {
                // The indexed body is fetched only now: this file already failed the hash and the
                // caller gave an excerpt. It must itself hash to the recorded value, or it proves nothing.
                let verdict = excerpt.map(|ex| match old_body() {
                    Some(old) if sha256_hex(old.as_bytes()) == recorded => passage_check(ex, &old, &text, terms),
                    _ => PassageCheck::Unusable,
                });
                match verdict {
                    Some(PassageCheck::Present) => Verdict::ChangedPassageKept(sig.mtime_ns),
                    Some(PassageCheck::Absent) => Verdict::ChangedPassageGone(sig.mtime_ns),
                    Some(PassageCheck::Unusable) | None => Verdict::Changed(sig.mtime_ns),
                }
            }
        }
        Ok(None) => {
            *read_budget = 0;
            unread()
        }
        Err(e) => Verdict::Unknown(format!("read {}: {e}", abs.display())),
    }
}

/// `s` with CRLF folded to LF.
pub(crate) fn lf_form(s: &str) -> Cow<'_, str> {
    if s.contains("\r\n") { Cow::Owned(s.replace("\r\n", "\n")) } else { Cow::Borrowed(s) }
}

/// True when every line ending in `s` is the same kind: all LF, or all CRLF, and no lone CR. A flip
/// between two such files is what git's autocrlf does; anything else (an added stray CR, a file
/// that mixes both) is an edit and must be reported as one.
pub(crate) fn eol_is_uniform(s: &str) -> bool {
    let (cr, crlf, lf) = (s.matches('\r').count(), s.matches("\r\n").count(), s.matches('\n').count());
    cr == crlf && (crlf == 0 || crlf == lf)
}

/// Same text up to a uniform line-ending flip. Git's autocrlf (or an editor's save settings)
/// rewrites every line ending in a file without the author touching a word; that is not a change.
pub(crate) fn same_text_modulo_line_endings(a: &str, b: &str) -> bool {
    a == b || (eol_is_uniform(a) && eol_is_uniform(b) && lf_form(a) == lf_form(b))
}

/// Does `text` carry the hash recall recorded, as it is or with its line endings flipped to the
/// other uniform form (all-LF <-> all-CRLF)? The recorded hash stays the hash of the text as
/// indexed, so no stored hash changes and no existing store needs a re-hash or a re-embed; only the
/// comparison is tolerant. Same policy as [`same_text_modulo_line_endings`]: a current file with a
/// stray CR or mixed endings matches only if it is byte-identical to what was indexed.
pub(crate) fn hash_matches_modulo_line_endings(text: &str, recorded: &str) -> bool {
    if sha256_hex(text.as_bytes()) == recorded {
        return true;
    }
    if !eol_is_uniform(text) {
        return false;
    }
    let lf = lf_form(text);
    if lf.as_ref() != text && sha256_hex(lf.as_bytes()) == recorded {
        return true;
    }
    sha256_hex(lf.replace('\n', "\r\n").as_bytes()) == recorded
}

/// Could FTS5 have highlighted `word` for the query term `term`? The index is porter-stemmed, so
/// "feeding" highlights `Feed` and "city" highlights `cities`. Not a stemmer: the two must be the
/// same word, or agree on all but the last letter of the shorter one (at least three letters) and
/// differ only in an ending of at most four letters. "station" is not a form of "starter". A miss
/// (organize / organization) costs only precision: the excerpt is then not placed and the whole
/// file is reported changed. It must stay narrow, because a wrong yes removes brackets that may
/// be the note's own text.
fn same_stem(word: &str, term: &str) -> bool {
    if word == term {
        return true;
    }
    let (w, t): (Vec<char>, Vec<char>) = (word.chars().collect(), term.chars().collect());
    let common = w.iter().zip(&t).take_while(|(a, b)| a == b).count();
    let shorter = w.len().min(t.len());
    common >= 3 && common + 1 >= shorter && w.len() - common <= 4 && t.len() - common <= 4
}

/// A regex that finds, in the indexed text, every place the shown excerpt `s` could have come from.
///
/// FTS5 `snippet()` wraps matched tokens in `[word]`. A pair may be such a highlight only when (1) it
/// encloses a bare word that is a form of one of the query's own terms ([`same_stem`]; the only
/// words snippet() highlights), and (2) it sits on token boundaries: the char before `[` and after
/// `]` is not alphanumeric, so `cache[key]` is source text. But a `[word]` that passes both tests
/// can still be the note's own text (a checkbox, a link label, the inside of a `[[wikilink]]`), and
/// nothing in the excerpt says which. So each such pair matches EITHER way, with or without its
/// brackets, and everything else matches literally. One pattern then stands for every reading of
/// the excerpt at once, and [`passage_check`] accepts a placement only when the whole pattern fits
/// exactly one place: choosing one reading (all pairs are highlights) once cleared an edited line
/// because that reading happened to fit a different, untouched line.
fn placement_pattern(s: &str, terms: &[String]) -> String {
    let c: Vec<char> = s.chars().collect();
    let mut pat = String::with_capacity(s.len() * 2);
    let mut lit = String::new();
    let mut i = 0;
    while i < c.len() {
        if c[i] == '[' {
            let mut j = i + 1;
            while j < c.len() && (c[j].is_alphanumeric() || c[j] == '_') {
                j += 1;
            }
            let before_ok = i == 0 || !c[i - 1].is_alphanumeric();
            let after_ok = j + 1 >= c.len() || !c[j + 1].is_alphanumeric();
            if j > i + 1 && j < c.len() && c[j] == ']' && before_ok && after_ok {
                let word: String = c[i + 1..j].iter().collect();
                if terms.iter().any(|t| same_stem(&word.to_lowercase(), t)) {
                    let w = regex::escape(&word);
                    pat.push_str(&regex::escape(&lit));
                    lit.clear();
                    pat.push_str(&format!(r"(?:\[{w}\]|{w})"));
                    i = j + 1;
                    continue;
                }
            }
        }
        lit.push(c[i]);
        i += 1;
    }
    pat.push_str(&regex::escape(&lit));
    pat
}

/// The whole line(s) of `text` that the byte range `start..end` touches.
fn line_span(text: &str, start: usize, end: usize) -> &str {
    let ls = text[..start].rfind('\n').map_or(0, |i| i + 1);
    let le = text[end..].find('\n').map_or(text.len(), |i| end + i);
    &text[ls..le]
}

/// The line(s) a match of `needle` at `at` sits on. Newlines at the needle's two ends belong to the
/// match (they must be there) but not to the lines it sits on: a needle ending in `\n` would
/// otherwise claim the NEXT line, and a line appended after the passage would read as a change to it.
fn passage_lines<'a>(text: &'a str, at: usize, needle: &str) -> &'a str {
    let lead = needle.len() - needle.trim_start_matches('\n').len();
    let trail = needle.len() - needle.trim_end_matches('\n').len();
    line_span(text, at + lead, (at + needle.len() - trail).max(at + lead))
}

/// Where the next match starting at or after byte `from` may begin: one char further on, so that
/// overlapping matches are seen (`match_indices` and `find_iter` skip them, and "run\nrun" sits
/// twice in "run\nrun\nrun").
fn step_past(text: &str, from: usize) -> usize {
    from + text[from..].chars().next().map_or(1, char::len_utf8)
}

/// Shorter than this an excerpt proves nothing.
const MIN_PASSAGE_CHARS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PassageCheck {
    Present,
    /// The lines the passage sat on are not in the file any more.
    Absent,
    /// Empty, too short, or not placeable at exactly one spot in the indexed text: nothing is known
    /// about the passage, so the caller keeps the whole-file answer ("this file changed").
    Unusable,
}

/// Lowercased alphanumeric runs of at least 2 chars: the terms FTS5 can highlight for `query`.
pub(crate) fn query_terms(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Is the passage a search showed still, unchanged, where it was? Judged against the text recall
/// INDEXED (`old`, the stored body whose hash is the recorded one) and the file now (`new`), so the
/// answer rests on provenance, not on "this text exists somewhere":
/// - the excerpt (outer `…` dropped) must fit exactly ONE place in `old`, overlaps counted, over
///   every reading of its `[word]` pairs at once ([`placement_pattern`]). An excerpt that fits no
///   place or more than one cannot be placed, so nothing is known about it: `Unusable`, and the
///   caller reports the whole file as changed;
/// - `new` must hold the same text, every character of it, on a line span that is byte-for-byte the
///   line span(s) the passage sat on in `old` ([`passage_lines`]). An edit anywhere on those lines (a
///   `# ` put in front of a call, a `[`/`]` added, a number continued, an indent moved, a space
///   doubled) is a change: `Absent`. The lines may sit elsewhere in the file; only their content is
///   compared. Only CRLF is folded to LF.
/// Edits on OTHER lines are exactly what this forgives. A doubt never answers "unchanged", and never
/// claims the passage is gone either.
pub(crate) fn passage_check(excerpt: &str, old: &str, new: &str, terms: &[String]) -> PassageCheck {
    let ex = excerpt.replace("\r\n", "\n");
    let ex = ex.trim_matches('…');
    if ex.trim().chars().count() < MIN_PASSAGE_CHARS {
        return PassageCheck::Unusable;
    }
    let (old, new) = (old.replace("\r\n", "\n"), new.replace("\r\n", "\n"));
    let Ok(re) = regex::Regex::new(&placement_pattern(ex, terms)) else { return PassageCheck::Unusable };
    // Exactly one place in the indexed text, whichever reading fits there. The needle is then the
    // indexed text itself at that place, the note's own brackets included.
    let Some(m) = re.find(&old) else { return PassageCheck::Unusable };
    let next = step_past(&old, m.start());
    if next <= old.len() && re.find_at(&old, next).is_some() {
        return PassageCheck::Unusable;
    }
    let (at, needle) = (m.start(), m.as_str());
    let want = passage_lines(&old, at, needle);
    let mut from = 0;
    while let Some(i) = new.get(from..).and_then(|rest| rest.find(needle)).map(|i| from + i) {
        if passage_lines(&new, i, needle) == want {
            return PassageCheck::Present;
        }
        from = step_past(&new, i);
    }
    PassageCheck::Absent
}

/// The file's bytes if it holds at most `cap` of them, else `None` (having read at most `cap` + 1).
pub(crate) fn read_capped(path: &Path, cap: u64) -> std::io::Result<Option<Vec<u8>>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.take(cap.saturating_add(1)).read_to_end(&mut bytes)?;
    Ok((bytes.len() as u64 <= cap).then_some(bytes))
}

/// Turn a verdict into the user-facing fields. The wording is the contract: it must tell the
/// reader to go to the file.
pub fn describe(row: &FreshRow, v: &Verdict) -> FreshInfo {
    let indexed = row
        .indexed_at
        .or_else(|| row.valid_from.parse::<i64>().ok());
    let mut info = FreshInfo {
        modified_at: row.modified_at.map(|ns| fmt_utc(ns / NS_PER_SEC)),
        indexed_at: indexed.map(fmt_utc),
        mtime_secs: row.modified_at.map(|ns| ns / NS_PER_SEC),
        ..FreshInfo::default()
    };
    let indexed_txt = info.indexed_at.clone().unwrap_or_else(|| "an unknown date".into());
    match v {
        Verdict::Fresh | Verdict::Touched(_) | Verdict::NoFile => info.stale = Some(false),
        Verdict::Changed(ns) => {
            let when = fmt_utc(ns / NS_PER_SEC);
            info.stale = Some(true);
            info.file_modified_at = Some(when.clone());
            info.mtime_secs = Some(ns / NS_PER_SEC);
            info.stale_note = Some(format!(
                "this file changed on {when} after recall indexed it on {indexed_txt}; read the actual file."
            ));
        }
        Verdict::ChangedPassageKept(ns) => {
            let when = fmt_utc(ns / NS_PER_SEC);
            info.stale = Some(false);
            info.file_modified_at = Some(when.clone());
            info.mtime_secs = Some(ns / NS_PER_SEC);
            info.stale_note = Some(format!(
                "file changed, this passage unchanged: the file changed on {when} after recall indexed it on {indexed_txt}, but the excerpt shown is still in it."
            ));
        }
        Verdict::ChangedPassageGone(ns) => {
            let when = fmt_utc(ns / NS_PER_SEC);
            info.stale = Some(true);
            info.file_modified_at = Some(when.clone());
            info.mtime_secs = Some(ns / NS_PER_SEC);
            info.stale_note = Some(format!(
                "this passage changed: the file changed on {when} after recall indexed it on {indexed_txt}, and a line the excerpt shown is on has changed; read the actual file."
            ));
        }
        Verdict::Deleted => {
            info.stale = Some(true);
            info.deleted = true;
            info.stale_note = Some(format!(
                "this file was deleted from disk after recall indexed it on {indexed_txt}; the indexed text is a copy of a file that no longer exists."
            ));
        }
        Verdict::Unknown(why) => {
            info.stale = None;
            info.stale_note = Some(format!("freshness not checked: {why}"));
        }
    }
    info
}

const FRESH_SELECT: &str = "SELECT id, path, source, modified_at, file_size, indexed_at, valid_from, content_sha256 FROM documents";
// Same shape for a database a pre-freshness binary last migrated: the three stat columns are
// simply unknown, and `check_row` falls back to hashing.
const LEGACY_SELECT: &str = "SELECT id, path, source, NULL, NULL, NULL, valid_from, content_sha256 FROM documents";

fn load_rows(conn: &Connection, ids: &[i64]) -> Result<Vec<FreshRow>> {
    let marks = vec!["?"; ids.len()].join(",");
    let run = |select: &str| -> rusqlite::Result<Vec<FreshRow>> {
        let mut stmt = conn.prepare(&format!("{select} WHERE id IN ({marks})"))?;
        let rows = stmt.query_map(rusqlite::params_from_iter(ids.iter()), |r| {
            Ok(FreshRow {
                id: r.get(0)?,
                path: r.get(1)?,
                source: r.get(2)?,
                modified_at: plausible_ns(r.get(3)?),
                file_size: r.get(4)?,
                indexed_at: r.get(5)?,
                valid_from: r.get(6)?,
                content_sha: r.get(7)?,
            })
        })?;
        rows.collect()
    };
    match run(FRESH_SELECT) {
        Ok(rows) => Ok(rows),
        Err(e) if e.to_string().contains("no such column") => Ok(run(LEGACY_SELECT)?),
        Err(e) => Err(e.into()),
    }
}

/// Freshness for a result set, in the order of `ids`. Opens its own READ-ONLY connection: the CLI
/// answers from the warm daemon without a `Store`, and a query must never write.
pub fn annotate_ids(ids: &[i64]) -> Vec<FreshInfo> {
    annotate_ids_for(ids, &[], "")
}

/// [`annotate_ids`] that also judges, for a file that changed, whether the excerpt shown for that
/// hit (`excerpts[i]` for `ids[i]`; a shorter slice means no excerpt) is still in it.
pub fn annotate_ids_for(ids: &[i64], excerpts: &[&str], query: &str) -> Vec<FreshInfo> {
    if ids.is_empty() {
        return vec![];
    }
    let conn = match default_db_path().and_then(|p| {
        Connection::open_with_flags(
            &p,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("open {}", p.display()))
    }) {
        Ok(c) => c,
        Err(e) => return vec![FreshInfo::unchecked(&format!("{e:#}")); ids.len()],
    };
    // A missing root would report every hit as deleted; "could not check" is the honest answer.
    match crate::store::check_vault_root(vault_root()) {
        Ok(root) => annotate_with_budget_excerpts(&conn, &root, ids, QUERY_HASH_BUDGET_BYTES, excerpts, &query_terms(query)),
        Err(e) => vec![FreshInfo::unchecked(&format!("{e:#}")); ids.len()],
    }
}

pub fn annotate_with(conn: &Connection, root: &Path, ids: &[i64]) -> Vec<FreshInfo> {
    annotate_with_budget(conn, root, ids, QUERY_HASH_BUDGET_BYTES)
}

/// Bytes of hit files one query may read and hash, best hits first ([`check_row_budgeted`]); past
/// it hits are judged by size + mtime alone. Measured 2026-11-06 on the live vault: hashing all 50
/// hits of a --top 50 query took up to 816 ms.
pub const QUERY_HASH_BUDGET_BYTES: u64 = 2 * 1024 * 1024;

pub fn annotate_with_budget(conn: &Connection, root: &Path, ids: &[i64], budget: u64) -> Vec<FreshInfo> {
    annotate_with_budget_excerpts(conn, root, ids, budget, &[], &[])
}

pub fn annotate_with_budget_excerpts(conn: &Connection, root: &Path, ids: &[i64], budget: u64, excerpts: &[&str], terms: &[String]) -> Vec<FreshInfo> {
    let rows = match load_rows(conn, ids) {
        Ok(r) => r,
        Err(e) => return vec![FreshInfo::unchecked(&format!("{e:#}")); ids.len()],
    };
    let by_id: HashMap<i64, FreshRow> = rows.into_iter().map(|r| (r.id, r)).collect();
    let mut left = budget;
    ids.iter()
        .enumerate()
        .map(|(i, id)| match by_id.get(id) {
            Some(row) => describe(row, &check_row_excerpt(root, row, true, &mut left, excerpts.get(i).copied(), terms, &|| {
                conn.query_row("SELECT body FROM documents WHERE id = ?1", [row.id], |r| r.get::<_, String>(0)).ok()
            })),
            None => FreshInfo::unchecked("document row not found"),
        })
        .collect()
}

/// Result order after the optional stale penalty and recency boost, as indices into the input.
/// Both knobs are in RANK POSITIONS and default to 0 (identity): they are unmeasured, so the
/// default must not move anything. Re-orders within the returned set only — it cannot pull in a
/// document that missed the top-k.
pub fn rerank_order(infos: &[FreshInfo], stale_penalty: f64, recency_boost: f64, now: i64) -> Vec<usize> {
    let mut keyed: Vec<(f64, usize)> = infos
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut key = i as f64;
            if f.is_stale() {
                key += stale_penalty;
            }
            if recency_boost > 0.0 {
                if let Some(m) = f.mtime_secs {
                    let age_days = ((now - m).max(0) as f64) / 86_400.0;
                    key -= recency_boost * 0.5f64.powf(age_days / 30.0); // 30-day half-life
                }
            }
            (key, i)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    keyed.into_iter().map(|(_, i)| i).collect()
}

// ── Path normalisation for watcher events ───────────────────────────────────────

/// Vault-relative, forward-slash path for an event path, or `None` when it is outside the root.
/// `case_insensitive` is true on Windows and macOS: the OS may report `c:\users\alice\Vault\a.md`
/// for a root configured as `C:\Users\alice\vault`. Never lower-cases the returned path.
pub fn rel_from_event(root: &Path, event: &Path, case_insensitive: bool) -> Option<String> {
    let norm = |p: &Path| p.to_string_lossy().replace('\\', "/");
    let (r, e) = (norm(root), norm(event));
    let r = r.trim_end_matches('/');
    let rest = if case_insensitive { strip_prefix_ci(&e, r)? } else { e.strip_prefix(r)? };
    if !rest.starts_with('/') {
        return None;
    }
    let rel = rest.trim_matches('/');
    if rel.is_empty() || rel.split('/').any(|c| c == "..") {
        return None;
    }
    Some(rel.to_string())
}

/// `s` minus `prefix`, comparing char by char with Unicode lower-casing, and slicing `s` by the
/// bytes it actually consumed. (Lower-casing a whole string can change its byte length, so slicing
/// the original by the lower-cased prefix length would cut mid-name or return the wrong tail.)
fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let mut it = s.char_indices();
    for pc in prefix.chars() {
        let (_, sc) = it.next()?;
        if sc != pc && !sc.to_lowercase().eq(pc.to_lowercase()) {
            return None;
        }
    }
    Some(it.next().map_or("", |(i, _)| &s[i..]))
}

pub fn is_indexable_name(rel: &str) -> bool {
    let ext = Path::new(rel).extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    ext == "md" || ext == "qmd"
}

// ── Writers ───────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ScanReport {
    pub checked: usize,
    pub changed: usize,
    pub deleted: usize,
    pub new_files: usize,
    pub cleared: usize,
    pub legacy_stamped: usize,
    pub legacy_remaining: usize,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ApplyReport {
    pub updated: usize,
    pub touched: usize,
    pub moved: usize,
    pub inserted: usize,
    pub revived: usize,
    pub tombstoned: usize,
    pub deletes_deferred: usize,
    /// Vault-relative paths whose text changed: the re-embed work list.
    pub reembed: Vec<String>,
    /// Ids of those documents (for the warm daemon's in-RAM vector patch).
    pub reembed_ids: Vec<i64>,
    /// Ids just tombstoned: the daemon must drop their vectors from the matrix too.
    pub tombstoned_ids: Vec<i64>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PathChange {
    Unchanged,
    Changed,
    Deleted,
    NewFile,
    Ignored,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FreshCounts {
    pub changed: i64,
    pub deleted: i64,
    pub new_pending: i64,
}

impl Store {
    /// Record a file's stat on its live row. `force` writes unconditionally (the body was just
    /// written); otherwise only when something actually differs, so an unchanged re-index of
    /// 70k files stays a read.
    pub(super) fn stamp_stat(tx: &Connection, rel: &str, sig: StatSig, now: i64, force: bool) -> Result<()> {
        tx.execute(
            "UPDATE documents SET modified_at = ?1, file_size = ?2, indexed_at = ?3,
                                  stale = 0, changed_at = NULL
              WHERE path = ?4 AND source = 'vault' AND valid_to IS NULL
                AND (?5 = 1 OR modified_at IS NOT ?1 OR file_size IS NOT ?2
                     OR indexed_at IS NULL OR stale <> 0)",
            params![sig.mtime_ns, sig.size, now, rel, i64::from(force)],
        )?;
        Ok(())
    }

    /// Ids of the live vault documents at these exact paths.
    pub fn live_ids_for_paths(&self, paths: &[String]) -> Result<Vec<i64>> {
        let mut out = Vec::new();
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM documents WHERE path = ?1 AND source = 'vault' AND valid_to IS NULL")?;
        for p in paths {
            if let Some(id) = stmt.query_row(params![p], |r| r.get::<_, i64>(0)).optional()? {
                out.push(id);
            }
        }
        Ok(out)
    }

    pub fn freshness_counts(&self) -> Result<FreshCounts> {
        let n = |sql: &str| -> Result<i64> { Ok(self.conn.query_row(sql, [], |r| r.get(0))?) };
        Ok(FreshCounts {
            changed: n("SELECT count(*) FROM documents WHERE valid_to IS NULL AND stale = 1")?,
            deleted: n("SELECT count(*) FROM documents WHERE valid_to IS NULL AND stale = 2")?,
            new_pending: n("SELECT count(*) FROM fs_events WHERE kind = 'new'")?,
        })
    }

    fn live_row_for(&self, rel: &str, case_insensitive: bool) -> Result<Option<FreshRow>> {
        let cols = "id, path, source, modified_at, file_size, indexed_at, valid_from, content_sha256";
        let map = |r: &rusqlite::Row| -> rusqlite::Result<FreshRow> {
            Ok(FreshRow {
                id: r.get(0)?,
                path: r.get(1)?,
                source: r.get(2)?,
                modified_at: plausible_ns(r.get(3)?),
                file_size: r.get(4)?,
                indexed_at: r.get(5)?,
                valid_from: r.get(6)?,
                content_sha: r.get(7)?,
            })
        };
        let exact = self
            .conn
            .query_row(
                &format!("SELECT {cols} FROM documents WHERE path = ?1 AND source = 'vault' AND valid_to IS NULL"),
                params![rel],
                map,
            )
            .optional()?;
        if exact.is_some() || !case_insensitive {
            return Ok(exact);
        }
        Ok(self
            .conn
            .query_row(
                &format!(
                    "SELECT {cols} FROM documents WHERE lower(path) = lower(?1)
                       AND source = 'vault' AND valid_to IS NULL ORDER BY id LIMIT 1"
                ),
                params![rel],
                map,
            )
            .optional()?)
    }

    /// One watcher event. Marks the document stale/deleted, or queues a brand-new file; never
    /// embeds and never rewrites a body. `rel` is vault-relative with forward slashes.
    pub fn note_path_event(&self, root: &Path, rel: &str, case_insensitive: bool) -> Result<PathChange> {
        if !is_indexable_name(rel) || PathExcludeList::load().is_excluded(rel) {
            return Ok(PathChange::Ignored);
        }
        let now = now_secs();
        if let Some(row) = self.live_row_for(rel, case_insensitive)? {
            let verdict = check_row(root, &row, true);
            return self.apply_verdict_in(&self.conn, root, &row, &verdict, now);
        }
        // No live row: a created file (or a renamed-in one). Remember it until a refresh.
        let abs = resolve_ingest_abs_path(root, rel)?;
        match std::fs::metadata(&abs) {
            Ok(md) if md.is_file() && (md.len() as usize) <= 512 * 1024 => {
                self.conn.execute(
                    "INSERT INTO fs_events (path, kind, seen_at) VALUES (?1, 'new', ?2)
                     ON CONFLICT(path) DO UPDATE SET seen_at = excluded.seen_at",
                    params![rel, now],
                )?;
                Ok(PathChange::NewFile)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Created then removed before anyone looked: forget it. Any OTHER error (a file
                // briefly locked by the writer) says nothing about whether it exists, so the
                // queue entry, if there is one, stays.
                self.conn.execute("DELETE FROM fs_events WHERE path = ?1", params![rel])?;
                Ok(PathChange::Ignored)
            }
            _ => Ok(PathChange::Ignored),
        }
    }

    /// A directory event (moved, deleted, or created): re-check every document underneath it and
    /// queue every indexable file found there.
    pub fn note_dir_event(&self, root: &Path, rel_dir: &str, case_insensitive: bool) -> Result<usize> {
        let mut touched = 0usize;
        let like = format!("{}/%", rel_dir.replace('\\', "/").replace('%', "\\%").replace('_', "\\_"));
        let sql = if case_insensitive {
            "SELECT path FROM documents WHERE valid_to IS NULL AND source = 'vault' AND lower(path) LIKE lower(?1) ESCAPE '\\'"
        } else {
            "SELECT path FROM documents WHERE valid_to IS NULL AND source = 'vault' AND path LIKE ?1 ESCAPE '\\'"
        };
        let mut paths: Vec<String> = {
            let mut stmt = self.conn.prepare(sql)?;
            let rows = stmt.query_map(params![like], |r| r.get::<_, String>(0))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        // New files inside a directory that appeared whole (a folder moved in): walk it.
        if let Ok(abs) = resolve_ingest_abs_path(root, rel_dir) {
            let mut stack = vec![abs];
            let mut seen = 0usize;
            while let Some(d) = stack.pop() {
                let Ok(rd) = std::fs::read_dir(&d) else { continue };
                for e in rd.flatten() {
                    seen += 1;
                    if seen > 50_000 {
                        break; // a pathological tree is the scan's job
                    }
                    let p = e.path();
                    match e.file_type() {
                        Ok(t) if t.is_dir() => stack.push(p),
                        Ok(t) if t.is_file() => {
                            if let Some(rel) = rel_from_event(root, &p, case_insensitive) {
                                paths.push(rel);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        paths.sort();
        paths.dedup();
        for rel in paths {
            if self.note_path_event(root, &rel, case_insensitive)? != PathChange::Ignored {
                touched += 1;
            }
        }
        Ok(touched)
    }

    /// The watcher-down fallback and the refresh's first phase: compare every live vault document
    /// to the disk by stat (cheap), hash only when the stat differs, and queue unseen files.
    ///
    /// `legacy_cap` bounds how many rows with NO recorded stat (written before this feature) get
    /// read and hashed per call, so the first scan of a 70k-document store converges over a few
    /// runs instead of reading the whole vault at once under a busy-backoff.
    pub fn scan_freshness(&self, root: &Path, find_new: bool, legacy_cap: usize) -> Result<ScanReport> {
        anyhow::ensure!(
            root.is_dir(),
            "vault root {} is not a directory — refusing to scan (an unmounted drive would flag everything deleted)",
            root.display()
        );
        let now = now_secs();
        let mut rep = ScanReport::default();
        let rows: Vec<(FreshRow, i64)> = {
            let mut stmt = self.conn.prepare(
                "SELECT id, path, source, modified_at, file_size, indexed_at, valid_from, content_sha256, stale
                   FROM documents WHERE valid_to IS NULL AND source = 'vault' AND path <> ''",
            )?;
            let it = stmt.query_map([], |r| {
                Ok((
                    FreshRow {
                        id: r.get(0)?,
                        path: r.get(1)?,
                        source: r.get(2)?,
                        modified_at: plausible_ns(r.get(3)?),
                        file_size: r.get(4)?,
                        indexed_at: r.get(5)?,
                        valid_from: r.get(6)?,
                        content_sha: r.get(7)?,
                    },
                    r.get::<_, i64>(8)?,
                ))
            })?;
            it.collect::<rusqlite::Result<_>>()?
        };
        let tx = self.conn.unchecked_transaction()?;
        let mut legacy_seen = 0usize;
        for (row, flag) in &rows {
            rep.checked += 1;
            let legacy = row.modified_at.is_none();
            if legacy {
                legacy_seen += 1;
                if legacy_seen > legacy_cap {
                    rep.legacy_remaining += 1;
                    continue;
                }
            }
            let v = check_row(root, row, false);
            match &v {
                Verdict::Fresh | Verdict::Touched(_) => {
                    self.apply_verdict_in(&tx, root, row, &v, now)?;
                    if *flag != STALE_NONE {
                        rep.cleared += 1;
                    }
                    if legacy {
                        rep.legacy_stamped += 1;
                    }
                }
                Verdict::Changed(_) | Verdict::ChangedPassageKept(_) | Verdict::ChangedPassageGone(_) => {
                    if *flag != STALE_CHANGED {
                        rep.changed += 1;
                    }
                    self.apply_verdict_in(&tx, root, row, &v, now)?;
                }
                Verdict::Deleted => {
                    if *flag != STALE_DELETED {
                        rep.deleted += 1;
                    }
                    self.apply_verdict_in(&tx, root, row, &v, now)?;
                }
                Verdict::NoFile | Verdict::Unknown(_) => {}
            }
        }
        tx.commit()?;

        if find_new {
            let exclude = PathExcludeList::load();
            let known: HashSet<String> = self
                .conn
                .prepare("SELECT path FROM documents WHERE valid_to IS NULL AND source = 'vault'")?
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<_>>()?;
            let en = enumerate_docs(root, &exclude)?;
            let tx = self.conn.unchecked_transaction()?;
            let mut present: HashSet<String> = HashSet::new();
            for f in &en.files {
                let rel = f.strip_prefix(root).unwrap_or(f).to_string_lossy().replace('\\', "/");
                if known.contains(&rel) || exclude.is_excluded(&rel) {
                    continue;
                }
                if let Ok(md) = std::fs::metadata(f) {
                    if md.len() as usize > 512 * 1024 {
                        continue;
                    }
                }
                present.insert(rel.clone());
                let inserted = tx.execute(
                    "INSERT OR IGNORE INTO fs_events (path, kind, seen_at) VALUES (?1, 'new', ?2)",
                    params![rel, now],
                )?;
                rep.new_files += inserted;
            }
            // Queue entries whose file is gone, or that a refresh already indexed, are noise.
            tx.execute(
                "DELETE FROM fs_events WHERE kind = 'new' AND path IN
                   (SELECT path FROM documents WHERE valid_to IS NULL AND source = 'vault')",
                [],
            )?;
            let queued: Vec<String> = {
                let mut stmt = tx.prepare("SELECT path FROM fs_events WHERE kind = 'new'")?;
                let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
                rows.collect::<rusqlite::Result<_>>()?
            };
            for p in queued {
                if !present.contains(&p) && path_is_absent(root, &p) {
                    tx.execute("DELETE FROM fs_events WHERE path = ?1", params![p])?;
                }
            }
            tx.commit()?;
        }
        Ok(rep)
    }

    /// Apply a verdict for one live row to its flag columns.
    fn apply_verdict_in(&self, tx: &Connection, root: &Path, row: &FreshRow, v: &Verdict, now: i64) -> Result<PathChange> {
        Ok(match v {
            Verdict::Fresh => {
                // Stat matches; only a stale flag left by an earlier event needs clearing.
                tx.execute(
                    "UPDATE documents SET stale = 0, changed_at = NULL WHERE id = ?1 AND stale <> 0",
                    params![row.id],
                )?;
                PathChange::Unchanged
            }
            Verdict::Touched(sig) => {
                // Same bytes behind a different stat: adopt the stat taken BEFORE the hash read.
                Self::stamp_stat(tx, &row.path, *sig, row.indexed_at.unwrap_or(now), false)?;
                PathChange::Unchanged
            }
            Verdict::Changed(_) | Verdict::ChangedPassageKept(_) | Verdict::ChangedPassageGone(_) => {
                tx.execute(
                    "UPDATE documents SET stale = 1, changed_at = COALESCE(changed_at, ?2) WHERE id = ?1",
                    params![row.id, now],
                )?;
                PathChange::Changed
            }
            Verdict::Deleted => {
                tx.execute(
                    "UPDATE documents SET stale = 2, changed_at = COALESCE(changed_at, ?2) WHERE id = ?1",
                    params![row.id, now],
                )?;
                PathChange::Deleted
            }
            Verdict::NoFile | Verdict::Unknown(_) => PathChange::Ignored,
        })
    }

    /// Re-index exactly what the scan/watcher flagged, keeping every document id:
    ///   * flagged-changed  → body rewritten IN PLACE (same id; `embedded_sha256` untouched, so the
    ///     existing stale-embedding queue picks the row up);
    ///   * queued-new       → a move if exactly one live document of identical content lost its
    ///     file (same rule and same function as `ingest_dir`), else revive-or-insert;
    ///   * flagged-deleted  → tombstoned (`valid_to`), unless the batch exceeds the delete cap.
    ///
    /// `queue_is_complete` says the created-file queue came from a full enumeration in this same
    /// run (`scan_freshness(find_new = true)`), not just from watcher events. A move needs it:
    /// "exactly one new file carries this content" is only knowable when EVERY new file is in the
    /// queue. After `a.md` -> `z.md` plus a copy at `b.md`, an event for `b.md` alone would
    /// otherwise look like an unambiguous rename and hand it `a.md`'s identity.
    pub fn apply_flagged(&self, root: &Path, queue_is_complete: bool) -> Result<ApplyReport> {
        anyhow::ensure!(root.is_dir(), "vault root {} is not a directory — refusing to apply", root.display());
        let now_s = now_secs();
        let now = Self::now();
        let mut rep = ApplyReport::default();
        let mut tx = self.conn.unchecked_transaction()?;

        // 1. edited files, in place.
        let changed: Vec<(i64, String, Option<String>)> = {
            let mut stmt = tx.prepare(
                "SELECT id, path, content_sha256 FROM documents
                  WHERE valid_to IS NULL AND source = 'vault' AND stale = 1",
            )?;
            let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        for (n, (id, rel, old_sha)) in changed.iter().enumerate() {
            let abs = resolve_ingest_abs_path(root, rel)?;
            // Stat BEFORE the read, same rule as ingest_dir.
            let Ok(md) = std::fs::metadata(&abs) else { continue };
            if md.len() as usize > 512 * 1024 {
                continue;
            }
            let sig = stat_sig(&md);
            let Ok(body) = read_markdown_text(&abs) else { continue };
            let digest = sha256_hex(body.as_bytes());
            if old_sha.as_deref().is_some_and(|o| o == digest || hash_matches_modulo_line_endings(&body, o)) {
                Self::stamp_stat(&tx, rel, sig, now_s, true)?;
                rep.touched += 1;
            } else {
                let title = derive_title(rel, &body);
                // Guarded on the hash we read the row at: if anything else rewrote the body since,
                // leave it flagged for the next pass instead of overwriting the newer text.
                let n = tx.execute(
                    "UPDATE documents SET title = ?2, body = ?3, valid_from = ?4, content_sha256 = ?5
                      WHERE id = ?1 AND content_sha256 IS ?6",
                    params![id, title, body, now, digest, old_sha],
                )?;
                if n == 0 {
                    continue;
                }
                Self::stamp_stat(&tx, rel, sig, now_s, true)?;
                rep.updated += 1;
                rep.reembed.push(rel.clone());
            }
            if (n + 1) % 500 == 0 {
                tx.commit()?;
                tx = self.conn.unchecked_transaction()?;
            }
        }
        tx.commit()?;
        let tx = self.conn.unchecked_transaction()?;

        // 2. created / renamed-in files.
        let queued: Vec<String> = {
            let mut stmt = tx.prepare("SELECT path FROM fs_events WHERE kind = 'new' ORDER BY path")?;
            let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        // (rel, sig, body, digest) read once; the digest count decides whether a move is unambiguous.
        let mut queue_ok = queue_is_complete;
        let mut incoming: Vec<(String, StatSig, String, String)> = Vec::new();
        let mut digest_count: HashMap<String, usize> = HashMap::new();
        for rel in &queued {
            let abs = resolve_ingest_abs_path(root, rel)?;
            let md = match std::fs::metadata(&abs) {
                Ok(m) => m,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    tx.execute("DELETE FROM fs_events WHERE path = ?1", params![rel])?;
                    continue;
                }
                // Unreadable right now (locked by the writer): keep the entry, retry next pass.
                // It may carry the same content as a file we can read, so "exactly one new file
                // has this body" is no longer knowable: no moves this pass.
                Err(_) => {
                    queue_ok = false;
                    continue;
                }
            };
            if !md.is_file() || md.len() as usize > 512 * 1024 {
                continue;
            }
            let sig = stat_sig(&md);
            let Ok(body) = read_markdown_text(&abs) else {
                queue_ok = false;
                continue;
            };
            let digest = sha256_hex(body.as_bytes());
            if !body.trim().is_empty() {
                *digest_count.entry(digest.clone()).or_insert(0) += 1;
            }
            incoming.push((rel.clone(), sig, body, digest));
        }
        for (rel, sig, body, digest) in &incoming {
            let taken: i64 = tx.query_row(
                "SELECT count(*) FROM documents WHERE path = ?1 AND source = 'vault' AND valid_to IS NULL",
                params![rel],
                |r| r.get(0),
            )?;
            if taken > 0 {
                tx.execute("DELETE FROM fs_events WHERE path = ?1", params![rel])?;
                continue;
            }
            let title = derive_title(rel, body);
            let blank = body.trim().is_empty();
            let unique = digest_count.get(digest).copied().unwrap_or(0) == 1;
            let moved_from = if unique && queue_ok {
                Self::find_moved_document(&tx, root, rel, digest, blank, -1)?
            } else {
                None
            };
            if let Some((old_id, old_title)) = moved_from {
                if title != old_title {
                    tx.execute(
                        "UPDATE documents SET path = ?2, title = ?3, valid_from = ?4, embedded_sha256 = NULL WHERE id = ?1",
                        params![old_id, rel, title, now],
                    )?;
                    rep.reembed.push(rel.clone());
                } else {
                    tx.execute(
                        "UPDATE documents SET path = ?2, valid_from = ?3 WHERE id = ?1",
                        params![old_id, rel, now],
                    )?;
                }
                rep.moved += 1;
            } else {
                Self::revive_or_insert(&tx, rel, &title, body, digest, &now, &mut rep.inserted, &mut rep.revived)?;
                rep.reembed.push(rel.clone());
            }
            Self::stamp_stat(&tx, rel, *sig, now_s, true)?;
            tx.execute("DELETE FROM fs_events WHERE path = ?1", params![rel])?;
        }
        tx.commit()?;

        // 3. deletions that were not renames (a rename's old row now carries its new path).
        let live: i64 = self.conn.query_row(
            "SELECT count(*) FROM documents WHERE valid_to IS NULL AND source = 'vault'",
            [],
            |r| r.get(0),
        )?;
        let gone: Vec<(i64, String)> = {
            let mut stmt = self.conn.prepare(
                "SELECT id, path FROM documents WHERE valid_to IS NULL AND source = 'vault' AND stale = 2",
            )?;
            let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        let cap = REFRESH_MIN_DELETE_ALLOWANCE.max((live as f64 * REFRESH_MAX_DELETE_FRACTION) as usize);
        // The floor of 50 must not let a SMALL index be wiped (a vault whose mount point is an
        // empty directory passes `is_dir`): never tombstone everything, and never more than half
        // of an index with a handful of documents.
        let wipes_index = gone.len() as i64 >= live || (live >= 4 && (gone.len() as i64) * 2 > live);
        if gone.len() > cap || wipes_index {
            rep.deletes_deferred = gone.len();
        } else {
            for (id, rel) in gone {
                // The filesystem's NotFound is the verdict (same authority as --prune-missing).
                if path_is_absent(root, &rel) {
                    self.conn.execute(
                        "UPDATE documents SET valid_to = ?2, stale = 0 WHERE id = ?1 AND valid_to IS NULL",
                        params![id, now],
                    )?;
                    rep.tombstoned += 1;
                    rep.tombstoned_ids.push(id);
                }
            }
        }
        rep.reembed_ids = self.live_ids_for_paths(&rep.reembed)?;
        Ok(rep)
    }
}

#[cfg(test)]
#[path = "freshness_tests.rs"]
mod tests;
