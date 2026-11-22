//! Tenancy v1 leak + migration tests (design: RECALL-TENANCY-DESIGN-2026-11-06.md §6).
//!
//! Every leak check pairs a NEGATIVE (other boxes absent) with a POSITIVE control (own boxes
//! present), so a dead search can never pass as a clean one.

use super::*;
use crate::scope::{self, Reader};

/// One temp DB per test thread.
fn temp_db(tag: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "recall-tenancy-{tag}-{}-{:?}.db",
        std::process::id(),
        std::thread::current().id()
    ));
    for ext in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{ext}", p.display()));
    }
    p
}

/// Restores the operator on drop, so a failing assert cannot leak a reader into the next test
/// that happens to run on this thread.
struct As(Reader);
impl As {
    fn agent(a: &str, p: Option<&str>, t: Option<&str>) -> Self {
        As(scope::set(scope::resolve(Some(a.into()), p.map(Into::into), t.map(Into::into)).unwrap()))
    }
}
impl Drop for As {
    fn drop(&mut self) {
        scope::set(self.0.clone());
    }
}

fn ins(s: &Store, path: &str, body: &str, scope_tag: &str) -> i64 {
    s.conn
        .execute(
            "INSERT INTO documents (path, title, body, source, created_at, valid_from, scope)
             VALUES (?1, ?1, ?2, 'cli', '1', '1', ?3)",
            params![path, body, scope_tag],
        )
        .unwrap();
    s.conn.last_insert_rowid()
}

/// Seven boxes, one doc each. Every body shares "wombat provisioning" (so one query matches all)
/// plus a token unique to its box.
fn seven(s: &Store) -> HashMap<&'static str, i64> {
    let mut m = HashMap::new();
    for (tag, tok) in [
        ("global", "qzglobal"),
        ("team:a", "qzteama"),
        ("project:p", "qzproj"),
        ("agent:w1", "qzwone"),
        ("agent:w2", "qzwtwo"),
        ("team:b", "qzteamb"),
        ("project:q", "qzprojq"),
    ] {
        let id = ins(s, &format!("notes/{tok}.md"), &format!("wombat provisioning runbook {tok}"), tag);
        m.insert(tag, id);
    }
    m
}

const MINE: [&str; 4] = ["global", "team:a", "project:p", "agent:w1"];
const THEIRS: [&str; 3] = ["agent:w2", "team:b", "project:q"];

fn ids_of(hits: &[Hit]) -> HashSet<i64> {
    hits.iter().map(|h| h.id).collect()
}

fn assert_matrix(what: &str, got: &HashSet<i64>, m: &HashMap<&str, i64>) {
    for t in MINE {
        assert!(got.contains(&m[t]), "{what}: own box {t} missing (positive control) — got {got:?}");
    }
    for t in THEIRS {
        assert!(!got.contains(&m[t]), "{what}: LEAK of {t} — got {got:?}");
    }
}

#[test]
fn fts_and_every_sql_read_surface_respect_the_reader() {
    let path = temp_db("matrix");
    let s = Store::open_at(&path).unwrap();
    let m = seven(&s);
    let tiers = TierFilter::from_spec("all");

    // Operator: everything, and no clause at all.
    let all = ids_of(&s.fts_search_with_tiers("wombat provisioning", 50, &tiers).unwrap());
    assert_eq!(all.len(), 7, "operator sees all seven boxes");

    let _w1 = As::agent("w1", Some("p"), Some("a"));
    let fts = ids_of(&s.fts_search_with_tiers("wombat provisioning", 50, &tiers).unwrap());
    assert_matrix("fts", &fts, &m);

    let recent = ids_of(&s.recent(50).unwrap());
    assert_matrix("recent", &recent, &m);

    let mut buf = Vec::new();
    s.export_json(&mut buf).unwrap();
    let exported: Vec<serde_json::Value> = serde_json::from_slice(&buf).unwrap();
    let exported: HashSet<i64> = exported.iter().map(|v| v["id"].as_i64().unwrap()).collect();
    assert_matrix("export", &exported, &m);

    let every: Vec<i64> = m.values().copied().collect();
    let hydrated: HashSet<i64> = s.title_body_rows(&every).unwrap().into_iter().map(|r| r.0).collect();
    assert_matrix("title_body_rows", &hydrated, &m);
    let by_id: HashSet<i64> = every.iter().filter(|id| s.hit_by_id(**id).unwrap().is_some()).copied().collect();
    assert_matrix("hit_by_id", &by_id, &m);

    let allowed = s.allowed_ids(&every).unwrap().expect("an agent always gets a set");
    assert_matrix("allowed_ids", &allowed, &m);

    // Abstain evidence: a private-only token must read as absent, own tokens as present.
    assert!(s.some_doc_holds_all(&["qzwone".into()]).unwrap(), "own token findable (control)");
    assert!(!s.some_doc_holds_all(&["qzwtwo".into()]).unwrap(), "LEAK: w2's token findable");
    assert_eq!(s.searchable_rows(&["qzteamb".into()], 10).unwrap(), 0, "LEAK: team:b df > 0");
    assert_eq!(s.searchable_rows(&["qzteama".into()], 10).unwrap(), 1, "own team df (control)");

    // Graph neighbours share the notes/ parent of every seed: only own boxes may come back.
    let seed = s.hit_by_id(m["global"]).unwrap().unwrap();
    let mut nb = ids_of(&s.graph_path_neighbors(&[seed], 50).unwrap());
    nb.insert(m["global"]);
    assert_matrix("graph_path_neighbors", &nb, &m);
    drop(s);
}

#[test]
fn fused_results_and_provenance_hold_no_foreign_id() {
    let path = temp_db("fused");
    let s = Store::open_at(&path).unwrap();
    let m = seven(&s);
    let _w1 = As::agent("w1", Some("p"), Some("a"));
    let out = crate::query::fused_sourced(&s, None, None, "wombat provisioning runbook", 20, &["fts5".to_string(), "graph".to_string()]).unwrap();
    let mut seen = HashSet::new();
    for (h, p) in &out {
        seen.insert(h.id);
        seen.extend(p.merged.iter().copied());
    }
    let theirs: HashSet<i64> = THEIRS.iter().map(|t| m[t]).collect();
    assert!(seen.is_disjoint(&theirs), "LEAK in fused hits/merged: {seen:?}");
    assert!(seen.contains(&m["agent:w1"]) && seen.contains(&m["global"]), "positive control: {seen:?}");
}

#[test]
fn opening_a_pre_tenancy_database_migrates_without_changing_operator_results() {
    let path = temp_db("pre");
    // The OLD shape (pre-tier, pre-tenancy) with rows already in it, as a restored snapshot has.
    let old = rusqlite::Connection::open(&path).unwrap();
    old.execute_batch(
        "CREATE TABLE documents (
             id INTEGER PRIMARY KEY, path TEXT NOT NULL DEFAULT '',
             title TEXT NOT NULL DEFAULT '', body TEXT NOT NULL,
             source TEXT NOT NULL DEFAULT 'cli', created_at TEXT NOT NULL,
             valid_from TEXT NOT NULL, valid_to TEXT, superseded_by INTEGER);
         INSERT INTO documents (path, title, body, created_at, valid_from)
           VALUES ('notes/a.md', 'A', 'wombat provisioning alpha', '1', '1'),
                  ('notes/b.md', 'B', 'wombat provisioning beta', '1', '1');",
    )
    .unwrap();
    drop(old);

    let s = Store::open_at(&path).unwrap();
    // FTS was created after the rows; index them the way a restore would, then search.
    s.conn.execute_batch("INSERT INTO documents_fts(documents_fts) VALUES('rebuild');").unwrap();
    let rows: Vec<(String, Option<String>)> = s
        .conn
        .prepare("SELECT scope, agent_id FROM documents ORDER BY id")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(rows, vec![("global".into(), None), ("global".into(), None)], "old rows become global, author unknown");

    // Operator: the shipped ranking SQL carries no scope text at all, and both rows come back.
    assert_eq!(scope::clause("d"), "");
    for pushdown in [false, true] {
        let sql = Store::fts_rank_sql(&TierFilter::from_spec("all").sql_clause("d"), pushdown);
        assert!(!sql.contains("scope"), "operator ranking SQL must be byte-identical to pre-tenancy: {sql}");
    }
    let hits = s.fts_search_with_tiers("wombat provisioning", 10, &TierFilter::from_spec("all")).unwrap();
    assert_eq!(hits.len(), 2);
    drop(s);

    // A second open must not take the writer lock (Codex r2): hold it elsewhere and open anyway.
    let blocker = rusqlite::Connection::open(&path).unwrap();
    blocker.execute_batch("BEGIN IMMEDIATE").unwrap();
    let t = std::time::Instant::now();
    let again = Store::open_at(&path);
    assert!(again.is_ok(), "steady-state open must not need the write lock: {:?}", again.err());
    assert!(t.elapsed() < std::time::Duration::from_secs(2), "open waited on the lock: {:?}", t.elapsed());
    blocker.execute_batch("ROLLBACK").unwrap();
}

#[test]
fn scope_triggers_enforce_grammar_and_immutability() {
    let path = temp_db("trig");
    let s = Store::open_at(&path).unwrap();
    for ok in ["global", "team:a", "project:x-1_2.3", "agent:w1"] {
        ins(&s, "ok.md", "x", ok);
    }
    let long = format!("team:{}", "x".repeat(65));
    for bad in ["", "Global", "agent:", "agent:a:b", "org:a", "agent:A", "agent:-x", "agent:a b", long.as_str()] {
        let r = s.conn.execute(
            "INSERT INTO documents (body, created_at, valid_from, scope) VALUES ('x', '1', '1', ?1)",
            params![bad],
        );
        assert!(r.is_err(), "malformed scope {bad:?} must be refused");
        assert!(!scope::valid_tag(bad), "Rust grammar must agree with the trigger on {bad:?}");
    }
    let id = ins(&s, "m.md", "x", "agent:w1");
    assert!(s.conn.execute("UPDATE documents SET scope = 'global' WHERE id = ?1", params![id]).is_err(), "scope is immutable");
    // Writing the same value (e.g. a whole-row UPDATE) is not a change and must still work.
    s.conn.execute("UPDATE documents SET scope = 'agent:w1', title = 't' WHERE id = ?1", params![id]).unwrap();
}

/// Saved observations all have `path = ''`. Fusion's case-duplicate collapse keyed on the path and
/// merged every one of them into a single hit, so an agent's second memory could never be found.
#[test]
fn saved_observations_with_empty_paths_are_not_collapsed_into_one() {
    let path = temp_db("emptypath");
    let s = Store::open_at(&path).unwrap();
    let a = s.insert("", "", "wombat provisioning first observation", "cli").unwrap();
    let b = s.insert("", "", "wombat provisioning second observation", "cli").unwrap();
    let out = crate::query::fused_sourced(&s, None, None, "wombat provisioning observation", 10, &["fts5".to_string()]).unwrap();
    let got: HashSet<i64> = out.iter().map(|(h, _)| h.id).collect();
    assert!(got.contains(&a) && got.contains(&b), "both saves must be returned separately: {got:?}");
}

/// The abstain phantom null is calibrated per reader view and cached under a key the other views
/// can never read: an agent must not reuse the operator's whole-corpus threshold, and must not
/// overwrite or seed the operator's (Codex tenancy b1).
#[test]
fn phantom_calibration_is_namespaced_per_reader() {
    let path = temp_db("cal");
    let s = Store::open_at(&path).unwrap();
    seven(&s);
    let keys = |s: &Store| -> Vec<String> {
        s.conn.prepare("SELECT key FROM abstain_calibration ORDER BY rowid").unwrap()
            .query_map([], |r| r.get(0)).unwrap().map(|r| r.unwrap()).collect()
    };
    let (op_fp, _, _) = s.searchable_snapshot().unwrap();
    crate::query::phantom_null(&s).unwrap();
    let after_op = keys(&s);
    assert!(after_op.iter().all(|k| k.starts_with('v')), "operator keys keep the pre-tenancy form: {after_op:?}");
    {
        let _w1 = As::agent("w1", Some("p"), Some("a"));
        let (ag_fp, _, _) = s.searchable_snapshot().unwrap();
        assert_ne!(op_fp, ag_fp, "the agent's searchable snapshot must differ from the operator's");
        crate::query::phantom_null(&s).unwrap();
    }
    let after = keys(&s);
    let agent_keys: Vec<&String> = after.iter().filter(|k| !after_op.contains(k)).collect();
    assert!(
        agent_keys.iter().all(|k| k.starts_with("s") && k.contains("agent:w1")),
        "agent keys must be namespaced by policy: {agent_keys:?}"
    );
    assert_eq!(after.iter().filter(|k| k.starts_with('v')).count(), after_op.len(), "agent must not touch operator keys");
}

/// Resident vector channels drop denied docs BEFORE top-k, so a scoped reader still gets k
/// candidates; the operator's map is untouched.
#[test]
fn denied_guard_keeps_vector_top_k_full_and_clears_after_the_query() {
    let path = temp_db("denied");
    let s = Store::open_at(&path).unwrap();
    let m = seven(&s);
    let mut per_doc: HashMap<i64, f32> = m.values().map(|id| (*id, 0.5)).collect();
    {
        let _w1 = As::agent("w1", Some("p"), Some("a"));
        let denied = s.denied_ids().unwrap().expect("agent gets a set");
        let want: HashSet<i64> = THEIRS.iter().map(|t| m[t]).collect();
        assert_eq!(denied, want, "exactly the other boxes are denied");
        let _g = scope::DeniedGuard::install(Some(denied));
        let mut map = per_doc.clone();
        scope::drop_denied(&mut map);
        assert_matrix("drop_denied", &map.keys().copied().collect(), &m);
    }
    assert!(s.denied_ids().unwrap().is_none(), "operator: no denied set");
    scope::drop_denied(&mut per_doc);
    assert_eq!(per_doc.len(), 7, "guard dropped: nothing is filtered afterwards");
}

/// A tombstoned private row is still denied: its vector stays in the resident matrix.
#[test]
fn denied_ids_include_tombstoned_private_rows() {
    let path = temp_db("deniedtomb");
    let s = Store::open_at(&path).unwrap();
    let m = seven(&s);
    s.conn.execute("UPDATE documents SET valid_to = '2' WHERE id = ?1", params![m["agent:w2"]]).unwrap();
    let _w1 = As::agent("w1", None, None);
    assert!(s.denied_ids().unwrap().unwrap().contains(&m["agent:w2"]));
}

/// Retire ownership: an agent may forget / supersede only what it wrote or what sits in its own
/// box, even when it can READ the row (shared project memory by someone else).
#[test]
fn an_agent_retires_only_its_own_memories() {
    let path = temp_db("retire");
    let s = Store::open_at(&path).unwrap();
    let w2_shared = s.insert_scoped("", "", "pangolin shared note by w2", "cli", "project:p", Some("w2")).unwrap();
    let w2_private = s.insert_scoped("", "", "pangolin private note by w2", "cli", "agent:w2", Some("w2")).unwrap();
    let w1_shared = s.insert_scoped("", "", "pangolin shared note by w1", "cli", "project:p", Some("w1")).unwrap();
    let w1_private = s.insert_scoped("", "", "pangolin private note by w1", "cli", "agent:w1", Some("w1")).unwrap();
    let live = |id: i64| -> bool {
        s.conn.query_row("SELECT valid_to IS NULL FROM documents WHERE id = ?1", params![id], |r| r.get(0)).unwrap()
    };
    {
        let _w1 = As::agent("w1", Some("p"), None);
        assert!(!s.supersede_id(w2_shared).unwrap(), "readable but not owned: 'not found'");
        assert!(!s.supersede_id(w2_private).unwrap(), "invisible: 'not found'");
        assert_eq!(s.forget("pangolin").unwrap(), 2, "forget reaches only w1's two notes");
    }
    assert!(live(w2_shared) && live(w2_private), "w2's memories survive w1");
    assert!(!live(w1_shared) && !live(w1_private));
}

/// `dream` never merges near-duplicates across boxes, so a private copy cannot retire a shared one.
#[test]
fn dream_consolidates_within_a_scope_only() {
    let path = temp_db("dream");
    let s = Store::open_at(&path).unwrap();
    let shared = s.insert_scoped("", "", "the deploy key rotates every ninety days on monday", "cli", "global", None).unwrap();
    let private = s.insert_scoped("", "", "the deploy key rotates every ninety days on monday", "cli", "agent:w1", Some("w1")).unwrap();
    let rep = crate::dream::consolidate(&s).unwrap();
    assert_eq!(rep.superseded, 0, "identical text in two boxes is not a duplicate");
    let a = s.insert_scoped("", "", "the deploy key rotates every ninety days on monday", "cli", "agent:w1", Some("w1")).unwrap();
    assert_eq!(crate::dream::consolidate(&s).unwrap().superseded, 1, "same box still consolidates");
    let live: i64 = s.conn.query_row(
        "SELECT count(*) FROM documents WHERE valid_to IS NULL AND id IN (?1, ?2, ?3)",
        params![shared, private, a], |r| r.get(0)).unwrap();
    assert_eq!(live, 2);
}

/// What a save records: the tag and the writer, and the operator path is unchanged.
#[test]
fn scoped_saves_record_scope_and_writer() {
    let path = temp_db("save");
    let s = Store::open_at(&path).unwrap();
    let op = s.insert("", "", "operator note", "cli").unwrap();
    let ag = s.insert_scoped("", "", "agent note", "cli", "project:p", Some("w1")).unwrap();
    let row = |id: i64| -> (String, Option<String>) {
        s.conn.query_row("SELECT scope, agent_id FROM documents WHERE id = ?1", params![id], |r| Ok((r.get(0)?, r.get(1)?))).unwrap()
    };
    assert_eq!(row(op), ("global".into(), None));
    assert_eq!(row(ag), ("project:p".into(), Some("w1".into())));
    let _w2 = As::agent("w2", Some("p"), None);
    assert_eq!(s.count_visible().unwrap(), 2, "same project: both visible");
    drop(_w2);
    let _w3 = As::agent("w3", None, None);
    assert_eq!(s.count_visible().unwrap(), 1, "other agent: only global");
}

/// Calibration retention is per reader family: agents calibrating in turn never evict the
/// operator's null (CodeRabbit, tenancy PR).
#[test]
fn agent_calibrations_do_not_evict_the_operators() {
    let path = temp_db("calkeep");
    let s = Store::open_at(&path).unwrap();
    // Real key shape: `<family><doc count>:<hash>`, with the count changing as the corpus grows.
    s.calibration_put("v9:100:00000000000000aa", "{}").unwrap();
    for i in 0..(CALIBRATION_KEEP * 3) {
        for a in ["w1", "w2", "w3", "w4"] {
            s.calibration_put(&format!("s9:global,agent:{a}:{}:{i:016x}", 100 + i), "{}").unwrap();
        }
    }
    assert!(s.calibration_get("v9:100:00000000000000aa").unwrap().is_some(), "operator null survived the agents");
    let w1: i64 = s.conn.query_row(
        "SELECT count(*) FROM abstain_calibration WHERE key LIKE 's9:global,agent:w1:%'", [], |r| r.get(0)).unwrap();
    assert_eq!(w1 as usize, CALIBRATION_KEEP, "per-family cap");
}
