use super::*;
use crate::refresh::{self, Outcome, RefreshOpts};
use crate::settings::Settings;
use std::time::Duration;

fn tmp(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!(
        "recall-fresh-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write(dir: &Path, rel: &str, body: &str) {
    let p = dir.join(rel);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&p, body).unwrap();
}

/// Push a file's mtime into the future so a same-second rewrite still looks different.
fn set_mtime_ahead(p: &Path, secs: u64) {
    let f = std::fs::OpenOptions::new().write(true).open(p).unwrap();
    f.set_modified(std::time::SystemTime::now() + Duration::from_secs(secs)).unwrap();
}

fn id_of(s: &Store, path: &str) -> i64 {
    s.conn
        .query_row(
            "SELECT id FROM documents WHERE path = ?1 AND valid_to IS NULL",
            params![path],
            |r| r.get(0),
        )
        .unwrap()
}

fn live_paths(s: &Store) -> Vec<String> {
    let mut st = s
        .conn
        .prepare("SELECT path FROM documents WHERE valid_to IS NULL ORDER BY path")
        .unwrap();
    st.query_map([], |r| r.get(0)).unwrap().map(|r| r.unwrap()).collect()
}

/// Stand-in for `recall embed`: give each live doc a vector and certify it against the body it
/// was built from, through the same write path the real embedder uses.
fn fake_embed(s: &Store, only: Option<&[i64]>) {
    let rows: Vec<(i64, String)> = {
        let mut st = s.conn.prepare("SELECT id, body FROM documents WHERE valid_to IS NULL").unwrap();
        st.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap().map(|r| r.unwrap()).collect()
    };
    for (id, body) in rows {
        if only.is_some_and(|o| !o.contains(&id)) {
            continue;
        }
        let chunk = vec![(
            vec![0u8; 8],
            ChunkPlacement { heading_path: "", char_start: 0, char_end: body.len().min(1) },
        )];
        assert!(s.replace_chunk_embeddings(id, 2, &chunk, &sha256_hex(body.as_bytes())).unwrap());
    }
}

/// The report of a refresh that ran; a skip panics WITH its reason (it used to say only "skipped").
fn ran(o: Outcome) -> refresh::RefreshReport {
    match o {
        Outcome::Ran(r) => r,
        Outcome::Skipped(why) => panic!("refresh skipped: {why}"),
    }
}

fn opts(dir: &Path) -> RefreshOpts {
    RefreshOpts {
        embed: false,
        ignore_busy: false,
        find_new: true,
        legacy_cap: 1_000,
        busy: || None,
        dir: Some(dir.to_path_buf()),
        embed_exe: None,
    }
}

fn infos(s: &Store, root: &Path, ids: &[i64]) -> Vec<FreshInfo> {
    annotate_with(&s.conn, root, ids)
}

fn queue(s: &Store) -> Vec<i64> {
    let sql = format!(
        "SELECT d.id FROM documents d WHERE {} ORDER BY d.id",
        needs_chunk_embedding_clause()
    );
    let mut st = s.conn.prepare(&sql).unwrap();
    st.query_map([], |r| r.get(0)).unwrap().map(|r| r.unwrap()).collect()
}

#[test]
fn dates_format_as_utc() {
    assert_eq!(fmt_utc(0), "1970-01-01 00:00 UTC");
    assert_eq!(fmt_utc(1_700_000_000), "2023-11-14 22:13 UTC");
    assert_eq!(fmt_utc(951_782_400), "2000-02-29 00:00 UTC"); // leap day
}

#[test]
fn ranking_knobs_default_to_identity_and_move_only_when_asked() {
    let f = |stale: bool, age_days: i64| FreshInfo {
        stale: Some(stale),
        mtime_secs: Some(1_000_000_000 - age_days * 86_400),
        ..FreshInfo::default()
    };
    let now = 1_000_000_000;
    let list = vec![f(true, 400), f(false, 400), f(false, 0)];
    assert_eq!(rerank_order(&list, 0.0, 0.0, now), vec![0, 1, 2], "both knobs off = untouched");
    assert_eq!(rerank_order(&list, 2.5, 0.0, now), vec![1, 2, 0], "penalty demotes the stale hit");
    // A hit modified today gains up to `boost` positions; one from 400 days ago gains ~0.
    assert_eq!(rerank_order(&list, 0.0, 3.0, now), vec![2, 0, 1]);
}

#[test]
fn windows_paths_match_across_case_and_separators() {
    let root = Path::new(r"C:\Users\alice\Vault");
    let ev = Path::new(r"c:\users\alice\vault\Sub\Note.md");
    assert_eq!(rel_from_event(root, ev, true).as_deref(), Some("Sub/Note.md"));
    assert_eq!(rel_from_event(root, ev, false), None, "case-sensitive fs must not fold case");
    assert_eq!(
        rel_from_event(Path::new(r"C:\Users\alice\Vault\"), Path::new(r"C:/Users/alice/Vault/a.md"), true).as_deref(),
        Some("a.md"),
        "trailing separator and forward slashes"
    );
    assert_eq!(rel_from_event(root, Path::new(r"C:\Users\alice\VaultX\a.md"), true), None, "sibling prefix");
    assert_eq!(rel_from_event(root, Path::new(r"C:\Users\alice\Vault"), true), None, "the root itself");
    assert_eq!(rel_from_event(root, Path::new(r"C:\Users\alice\Vault\..\x.md"), true), None);
    let posix = rel_from_event(Path::new("/v/notes"), Path::new("/v/notes/a/b.md"), false);
    assert_eq!(posix.as_deref(), Some("a/b.md"));
}

#[test]
fn a_watcher_event_with_the_wrong_case_still_finds_the_row() {
    let dir = tmp("case");
    write(&dir, "Notes/Alpha.md", "# Alpha\nfirst");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&dir).unwrap();
    write(&dir, "Notes/Alpha.md", "# Alpha\nsecond, longer");
    set_mtime_ahead(&dir.join("Notes/Alpha.md"), 5);
    // Windows reports whatever case the editor used.
    assert_eq!(s.note_path_event(&dir, "notes/alpha.md", true).unwrap(), PathChange::Changed);
    assert_eq!(s.freshness_counts().unwrap().changed, 1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn check_row_tells_an_edit_from_a_touch_and_a_delete_from_an_ad_hoc_note() {
    let dir = tmp("check");
    write(&dir, "a.md", "# A\nbody");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&dir).unwrap();
    let note_id = s.insert("", "", "an ad-hoc observation", "cli").unwrap();
    let a = id_of(&s, "a.md");

    let fresh = infos(&s, &dir, &[a, note_id]);
    assert_eq!(fresh[0].stale, Some(false));
    assert!(fresh[0].modified_at.is_some() && fresh[0].indexed_at.is_some(), "recency fields exposed");
    assert_eq!(fresh[1].stale, Some(false), "a note with no file cannot drift");

    // Same bytes, new mtime: a `touch` must NOT flag the hit.
    set_mtime_ahead(&dir.join("a.md"), 30);
    assert_eq!(infos(&s, &dir, &[a])[0].stale, Some(false));

    // Different bytes: flagged, with the contractual wording and both dates.
    write(&dir, "a.md", "# A\nbody, edited");
    set_mtime_ahead(&dir.join("a.md"), 60);
    let f = &infos(&s, &dir, &[a])[0];
    assert_eq!(f.stale, Some(true));
    assert!(!f.deleted);
    let note = f.stale_note.as_deref().unwrap();
    assert!(note.contains("this file changed on ") && note.contains(" after recall indexed it on "), "{note}");
    assert!(note.ends_with("read the actual file."), "{note}");
    assert!(f.file_modified_at.is_some());

    std::fs::remove_file(dir.join("a.md")).unwrap();
    let f = &infos(&s, &dir, &[a])[0];
    assert!(f.is_stale() && f.deleted, "a missing file is flagged deleted");
    assert!(f.stale_note.as_deref().unwrap().contains("deleted from disk"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_row_with_no_recorded_stat_falls_back_to_its_hash() {
    // The shape a pre-freshness database has: the stat columns do not exist at all.
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, source TEXT, valid_from TEXT, content_sha256 TEXT);",
    )
    .unwrap();
    let dir = tmp("legacy");
    write(&dir, "old.md", "unchanged text");
    let sha = sha256_hex(b"unchanged text");
    conn.execute("INSERT INTO documents VALUES (1, 'old.md', 'vault', '1700000000', ?1)", params![sha])
        .unwrap();
    let f = &annotate_with(&conn, &dir, &[1])[0];
    assert_eq!(f.stale, Some(false));
    assert_eq!(f.indexed_at.as_deref(), Some("2023-11-14 22:13 UTC"), "valid_from stands in for indexed_at");
    write(&dir, "old.md", "edited text");
    assert_eq!(annotate_with(&conn, &dir, &[1])[0].stale, Some(true));
    // An id that is not in the table says so instead of claiming freshness.
    assert_eq!(annotate_with(&conn, &dir, &[99])[0].stale, None);
    std::fs::remove_dir_all(&dir).ok();
}

/// The headline scenario: edit, rename, delete and create files in a vault, then query.
#[test]
fn edit_rename_delete_create_flag_immediately_and_one_refresh_clears_them() {
    let vault = tmp("life");
    let home = tmp("life-home");
    write(&vault, "a.md", "# Alpha\nthe original alpha text");
    write(&vault, "b.md", "plain note without a heading, about bees");
    write(&vault, "c.md", "# Gamma\nsoon to be deleted");
    write(&vault, "d.md", "# Delta\nnever touched");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();
    fake_embed(&s, None);
    assert_eq!(s.count_docs_needing_chunk_embeddings().unwrap(), 0, "starts fully embedded");
    let (a, b, c, d) = (id_of(&s, "a.md"), id_of(&s, "b.md"), id_of(&s, "c.md"), id_of(&s, "d.md"));

    // An agent edits a, renames b, deletes c, creates e.
    write(&vault, "a.md", "# Alpha\nthe REWRITTEN alpha text, now longer");
    set_mtime_ahead(&vault.join("a.md"), 60);
    std::fs::rename(vault.join("b.md"), vault.join("b-renamed.md")).unwrap();
    std::fs::remove_file(vault.join("c.md")).unwrap();
    write(&vault, "e.md", "# Epsilon\nbrand new file");

    // Query time, nothing has run yet: every changed hit is flagged, the untouched one is not.
    let q = infos(&s, &vault, &[a, b, c, d]);
    assert_eq!(q[0].stale, Some(true), "edited");
    assert!(q[1].deleted && q[2].deleted, "renamed-away and deleted files read as deleted");
    assert_eq!(q[3].stale, Some(false), "untouched file stays unflagged");

    let out = refresh::run(&s, &vault, &opts(&home)).unwrap();
    let Outcome::Ran(rep) = out else { panic!("refresh skipped: {out:?}") };
    assert_eq!((rep.updated, rep.moved, rep.inserted, rep.tombstoned), (1, 1, 1, 1), "{}", rep.summary());

    // Doc ids survive: in-place edit and rename both keep the id (and so their vectors).
    assert_eq!(id_of(&s, "a.md"), a);
    assert_eq!(id_of(&s, "b-renamed.md"), b);
    assert_eq!(id_of(&s, "d.md"), d);
    assert_eq!(live_paths(&s), ["a.md", "b-renamed.md", "d.md", "e.md"]);
    let body: String = s.conn.query_row("SELECT body FROM documents WHERE id = ?1", params![a], |r| r.get(0)).unwrap();
    assert!(body.contains("REWRITTEN"));
    let old_vectors: i64 = s
        .conn
        .query_row("SELECT count(*) FROM bge_chunk_embeddings WHERE doc_id = ?1", params![a], |r| r.get(0))
        .unwrap();
    assert_eq!(old_vectors, 1, "the old vectors are kept until re-embedded");

    // Flags are cleared.
    let e = id_of(&s, "e.md");
    let after = infos(&s, &vault, &[a, b, d, e]);
    assert!(after.iter().all(|f| f.stale == Some(false)), "{after:?}");
    assert_eq!(s.freshness_counts().unwrap(), FreshCounts::default());

    // Re-embedding: the existing stale-hash queue holds exactly the documents whose text changed
    // (a edited, b retitled by the rename, e new) and not the untouched one.
    let mut expect = vec![a, b, e];
    expect.sort_unstable();
    assert_eq!(queue(&s), expect);
    assert!(!queue(&s).contains(&d));
    let mut handed: Vec<i64> = rep.reembed_ids.clone();
    handed.sort_unstable();
    assert_eq!(handed, expect, "the refresh hands the embedder exactly the stale queue");
    fake_embed(&s, Some(&rep.reembed_ids));
    assert_eq!(s.count_docs_needing_chunk_embeddings().unwrap(), 0, "vectors re-embedded, queue empty");

    // Second refresh: nothing to do.
    let again = ran(refresh::run(&s, &vault, &opts(&home)).unwrap());
    assert_eq!((again.updated, again.moved, again.inserted, again.tombstoned), (0, 0, 0, 0));
    std::fs::remove_dir_all(&vault).ok();
    std::fs::remove_dir_all(&home).ok();
}

#[test]
fn the_watcher_down_scan_finds_what_the_watcher_would_have() {
    let vault = tmp("scan");
    write(&vault, "a.md", "# A\none");
    write(&vault, "b.md", "# B\ntwo");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();
    // No watcher ran. Changes happen anyway.
    write(&vault, "a.md", "# A\none, edited");
    set_mtime_ahead(&vault.join("a.md"), 60);
    std::fs::remove_file(vault.join("b.md")).unwrap();
    write(&vault, "new.md", "# New\nfresh");

    let r = s.scan_freshness(&vault, true, 100).unwrap();
    assert_eq!((r.changed, r.deleted, r.new_files), (1, 1, 1), "{r:?}");
    assert_eq!(s.freshness_counts().unwrap(), FreshCounts { changed: 1, deleted: 1, new_pending: 1 });
    // Idempotent: a second scan reports nothing new.
    let r2 = s.scan_freshness(&vault, true, 100).unwrap();
    assert_eq!((r2.changed, r2.deleted, r2.new_files), (0, 0, 0));
    std::fs::remove_dir_all(&vault).ok();
}

#[test]
fn the_watcher_path_and_the_scan_agree() {
    let vault = tmp("agree");
    write(&vault, "a.md", "# A\none");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();
    write(&vault, "a.md", "# A\none, edited");
    set_mtime_ahead(&vault.join("a.md"), 60);
    write(&vault, "n.md", "# N\nnew");
    assert_eq!(s.note_path_event(&vault, "a.md", false).unwrap(), PathChange::Changed);
    assert_eq!(s.note_path_event(&vault, "n.md", false).unwrap(), PathChange::NewFile);
    assert_eq!(s.note_path_event(&vault, "image.png", false).unwrap(), PathChange::Ignored);
    let before = s.freshness_counts().unwrap();
    let r = s.scan_freshness(&vault, true, 100).unwrap();
    assert_eq!((r.changed, r.new_files), (0, 0), "already flagged by the watcher: {r:?}");
    assert_eq!(s.freshness_counts().unwrap(), before);
    // A file created and removed before anyone looked leaves no queue entry behind.
    write(&vault, "tmp.md", "# T\nx");
    s.note_path_event(&vault, "tmp.md", false).unwrap();
    std::fs::remove_file(vault.join("tmp.md")).unwrap();
    s.note_path_event(&vault, "tmp.md", false).unwrap();
    assert_eq!(s.freshness_counts().unwrap().new_pending, 1);
    std::fs::remove_dir_all(&vault).ok();
}

#[test]
fn a_folder_rename_is_followed_from_its_directory_events() {
    let vault = tmp("dir");
    write(&vault, "old/x.md", "# X\nx text");
    write(&vault, "old/y.md", "# Y\ny text");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();
    let x = id_of(&s, "old/x.md");
    std::fs::rename(vault.join("old"), vault.join("moved")).unwrap();
    // One event per directory, nothing per file — as Windows reports a folder rename.
    s.note_dir_event(&vault, "old", true).unwrap();
    s.note_dir_event(&vault, "moved", true).unwrap();
    let c = s.freshness_counts().unwrap();
    assert_eq!((c.deleted, c.new_pending), (2, 2));
    // A move needs the full queue, which a refresh always builds first.
    s.scan_freshness(&vault, true, 100).unwrap();
    let rep = s.apply_flagged(&vault, true).unwrap();
    assert_eq!((rep.moved, rep.tombstoned, rep.inserted), (2, 0, 0), "both files followed the folder");
    assert_eq!(id_of(&s, "moved/x.md"), x, "doc id kept across the folder rename");
    std::fs::remove_dir_all(&vault).ok();
}

#[test]
fn a_rename_plus_a_copy_never_hands_the_original_identity_to_either() {
    let vault = tmp("amb");
    write(&vault, "a.md", "identical body text for the ambiguity case");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();
    let a = id_of(&s, "a.md");
    std::fs::rename(vault.join("a.md"), vault.join("z.md")).unwrap();
    write(&vault, "b.md", "identical body text for the ambiguity case");
    s.scan_freshness(&vault, true, 100).unwrap();
    let rep = s.apply_flagged(&vault, true).unwrap();
    assert_eq!(rep.moved, 0, "two incoming files carry the body: unknowable which moved");
    assert_eq!(rep.inserted, 2);
    assert_eq!(live_paths(&s), ["b.md", "z.md"]);
    assert!(id_of(&s, "z.md") != a && id_of(&s, "b.md") != a);
    std::fs::remove_dir_all(&vault).ok();
}

#[test]
fn a_mass_deletion_is_held_back_not_applied() {
    let vault = tmp("mass");
    for i in 0..120 {
        write(&vault, &format!("n{i:03}.md"), &format!("# N{i}\nbody number {i}"));
    }
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();
    for i in 0..60 {
        std::fs::remove_file(vault.join(format!("n{i:03}.md"))).unwrap();
    }
    s.scan_freshness(&vault, false, 100).unwrap();
    let rep = s.apply_flagged(&vault, true).unwrap();
    assert_eq!((rep.tombstoned, rep.deletes_deferred), (0, 60));
    assert_eq!(live_paths(&s).len(), 120, "nothing tombstoned; still flagged for review");
    assert_eq!(s.freshness_counts().unwrap().deleted, 60);
    // A vanished vault root (unmounted drive) refuses outright instead of flagging everything.
    let gone = vault.join("not-there");
    assert!(s.scan_freshness(&gone, true, 10).is_err());
    assert!(s.apply_flagged(&gone, true).is_err());
    std::fs::remove_dir_all(&vault).ok();
}

#[test]
fn legacy_rows_get_stamped_in_bounded_batches() {
    let vault = tmp("legacy-scan");
    for i in 0..5 {
        write(&vault, &format!("l{i}.md"), &format!("# L{i}\nlegacy {i}"));
    }
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();
    s.conn
        .execute("UPDATE documents SET modified_at = NULL, file_size = NULL, indexed_at = NULL", [])
        .unwrap();
    let r = s.scan_freshness(&vault, false, 3).unwrap();
    assert_eq!((r.legacy_stamped, r.legacy_remaining, r.changed), (3, 2, 0), "{r:?}");
    let r = s.scan_freshness(&vault, false, 3).unwrap();
    assert_eq!((r.legacy_stamped, r.legacy_remaining), (2, 0));
    std::fs::remove_dir_all(&vault).ok();
}

#[test]
fn refresh_skips_when_the_lock_is_held_or_the_machine_is_busy_and_says_so() {
    let vault = tmp("lock");
    let home = tmp("lock-home");
    write(&vault, "a.md", "# A\nx");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();

    let held = refresh::IndexLock::try_acquire_in(Some(&home)).unwrap().expect("first holder");
    match refresh::run(&s, &vault, &opts(&home)).unwrap() {
        Outcome::Skipped(why) => assert!(why.contains("lock"), "{why}"),
        Outcome::Ran(_) => panic!("ran during another index"),
    }
    drop(held);

    let mut busy = opts(&home);
    busy.busy = || Some("system busy (99% CPU > 75% limit)".into());
    assert!(matches!(refresh::run(&s, &vault, &busy).unwrap(), Outcome::Skipped(w) if w.contains("busy")));
    busy.ignore_busy = true;
    assert!(matches!(refresh::run(&s, &vault, &busy).unwrap(), Outcome::Ran(_)), "--force overrides the back-off");
    // A run records itself, so a `nightly` setting sees it.
    assert!(refresh::RefreshState::load_in(Some(&home)).last_ok_unix.is_some());
    std::fs::remove_dir_all(&vault).ok();
    std::fs::remove_dir_all(&home).ok();
}

#[test]
fn defaults_leave_ranking_alone_and_lazy_checking_on() {
    let s = Settings::default();
    assert!(s.freshness_check);
    assert_eq!((s.stale_penalty, s.recency_boost), (0.0, 0.0));
}

#[test]
fn the_daemon_patch_read_returns_only_the_refreshed_documents_vectors() {
    let vault = tmp("patch");
    write(&vault, "a.md", "# A\nalpha");
    write(&vault, "b.md", "# B\nbeta");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();
    fake_embed(&s, None);
    let (a, b) = (id_of(&s, "a.md"), id_of(&s, "b.md"));
    let only_a = s.load_chunk_embeddings_for_docs(&[a]).unwrap();
    assert!(!only_a.is_empty() && only_a.iter().all(|(id, _)| *id == a));
    assert!(s.load_chunk_embeddings_for_docs(&[]).unwrap().is_empty());
    let all = s.load_all_chunk_embeddings().unwrap();
    assert!(all.iter().any(|(id, _)| *id == b), "the unfiltered load is unchanged");
    std::fs::remove_dir_all(&vault).ok();
}

#[test]
fn a_failed_embed_is_retried_next_pass_and_does_not_advance_the_nightly_clock() {
    let vault = tmp("embfail");
    let home = tmp("embfail-home");
    write(&vault, "a.md", "# A\none");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();
    fake_embed(&s, None);
    write(&vault, "a.md", "# A\none, edited");
    set_mtime_ahead(&vault.join("a.md"), 60);

    let mut o = opts(&home);
    o.embed = true;
    o.embed_exe = Some(home.join("no-such-recall-binary"));
    let rep = ran(refresh::run(&s, &vault, &o).unwrap());
    assert!(rep.embed_failed, "{}", rep.summary());
    assert!(rep.embed_note.contains("NOT DONE"), "{}", rep.embed_note);
    let st = refresh::RefreshState::load_in(Some(&home));
    assert_eq!(st.pending_embed, vec!["a.md".to_string()], "the path is remembered");
    assert!(st.last_ok_unix.is_none(), "a failed embed must not look like a finished night");
    assert!(refresh::take_patch(Some(&home)).is_empty(), "nothing re-embedded, nothing to patch");

    // Next pass: no file flagged any more, but the remembered path is offered to the embedder again.
    let rep2 = ran(refresh::run(&s, &vault, &o).unwrap());
    assert_eq!(rep2.updated, 0);
    assert!(rep2.embed_failed, "retried, not forgotten: {}", rep2.summary());
    // With the embed switched off the work stays queued and the clock advances (explicit choice).
    o.embed = false;
    let rep3 = ran(refresh::run(&s, &vault, &o).unwrap());
    assert!(!rep3.embed_failed);
    assert_eq!(refresh::RefreshState::load_in(Some(&home)).pending_embed, vec!["a.md".to_string()]);
    std::fs::remove_dir_all(&vault).ok();
    std::fs::remove_dir_all(&home).ok();
}

#[test]
fn tombstoned_documents_are_queued_for_the_daemon_to_drop() {
    let vault = tmp("tomb");
    let home = tmp("tomb-home");
    for i in 0..8 {
        write(&vault, &format!("t{i}.md"), &format!("# T{i}\ntomb {i}"));
    }
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();
    let gone = id_of(&s, "t3.md");
    std::fs::remove_file(vault.join("t3.md")).unwrap();
    let rep = ran(refresh::run(&s, &vault, &opts(&home)).unwrap());
    assert_eq!(rep.tombstoned, 1);
    assert_eq!(refresh::take_patch(Some(&home)), vec![gone]);
    assert!(refresh::take_patch(Some(&home)).is_empty(), "taken once");
    std::fs::remove_dir_all(&vault).ok();
    std::fs::remove_dir_all(&home).ok();
}

#[test]
fn a_small_index_cannot_be_wiped_by_the_floor_of_fifty() {
    // A vault whose mount point is an empty directory passes `is_dir`; every document "vanishes".
    let vault = tmp("small");
    for i in 0..10 {
        write(&vault, &format!("s{i}.md"), &format!("# S{i}\nsmall {i}"));
    }
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();
    for i in 0..10 {
        std::fs::remove_file(vault.join(format!("s{i}.md"))).unwrap();
    }
    s.scan_freshness(&vault, true, 100).unwrap();
    let rep = s.apply_flagged(&vault, true).unwrap();
    assert_eq!((rep.tombstoned, rep.deletes_deferred), (0, 10), "all ten gone = held back, not wiped");
    assert_eq!(live_paths(&s).len(), 10);
    std::fs::remove_dir_all(&vault).ok();
}

#[test]
fn a_watcher_only_queue_never_resolves_a_move() {
    // a.md renamed to z.md, then copied to b.md; only b.md's event has arrived. One queued file
    // carries the body, but z.md is also on disk: not knowable from the queue alone.
    let vault = tmp("evq");
    write(&vault, "a.md", "identical body for the partial queue case");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&vault).unwrap();
    let a = id_of(&s, "a.md");
    std::fs::rename(vault.join("a.md"), vault.join("z.md")).unwrap();
    write(&vault, "b.md", "identical body for the partial queue case");
    s.note_path_event(&vault, "a.md", false).unwrap();
    s.note_path_event(&vault, "b.md", false).unwrap();
    let rep = s.apply_flagged(&vault, false).unwrap();
    assert_eq!(rep.moved, 0);
    assert!(id_of(&s, "b.md") != a, "the copy must not inherit the original's identity");
    std::fs::remove_dir_all(&vault).ok();
}

#[test]
fn a_touch_adopts_the_stat_taken_before_the_hash_read_and_edits_are_caught_at_ns_resolution() {
    let dir = tmp("touch");
    write(&dir, "a.md", "# A\nbody");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&dir).unwrap();
    set_mtime_ahead(&dir.join("a.md"), 40);
    let want = stat_sig(&std::fs::metadata(dir.join("a.md")).unwrap());
    assert_eq!(s.note_path_event(&dir, "a.md", false).unwrap(), PathChange::Unchanged);
    let (m, z): (i64, i64) = s
        .conn
        .query_row("SELECT modified_at, file_size FROM documents WHERE path = 'a.md'", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!((m, z), (want.mtime_ns, want.size), "stat recorded after a touch");
    // Same size, different bytes, mtime a few ns apart from the recorded one.
    write(&dir, "a.md", "# A\nbodz");
    let f = std::fs::OpenOptions::new().write(true).open(dir.join("a.md")).unwrap();
    f.set_modified(std::time::UNIX_EPOCH + Duration::from_nanos(want.mtime_ns as u64 + 1_000)).unwrap();
    drop(f);
    assert_eq!(s.note_path_event(&dir, "a.md", false).unwrap(), PathChange::Changed);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn case_insensitive_prefix_survives_characters_whose_lowercase_changes_byte_length() {
    // 'İ' (2 bytes) lower-cases to 'i' + U+0307 (3 bytes): slicing by the lowered length used to
    // return the wrong tail.
    let root = Path::new("C:/İ");
    assert_eq!(rel_from_event(root, Path::new("C:/İ/a.md"), true).as_deref(), Some("a.md"));
}

/// Codex conc r1 #5: the per-query read budget is charged by the size on disk now and is never
/// exceeded; a hit past it is judged by its stat alone.
#[test]
fn the_query_read_budget_is_enforced_where_files_are_read() {
    let dir = tmp("budget");
    let big = "x".repeat(400 * 1024); // under the 512 KB ingest cap
    write(&dir, "big.md", &big);
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&dir).unwrap();
    let id = id_of(&s, "big.md");
    // Touch: same bytes, newer mtime. Reading shows it is unchanged; the stat alone cannot.
    let f = std::fs::OpenOptions::new().write(true).open(dir.join("big.md")).unwrap();
    f.set_modified(std::time::SystemTime::now() + Duration::from_secs(5)).unwrap();
    drop(f);
    // A budget below the file's size: not read, so the changed stat is reported as a change.
    assert_eq!(annotate_with_budget(&s.conn, &dir, &[id], 300 * 1024)[0].stale, Some(true));
    // A budget that covers it: read and hashed, so the touch is recognised as no change.
    assert_eq!(annotate_with_budget(&s.conn, &dir, &[id], 600 * 1024)[0].stale, Some(false));
    // And the budget is shared: two reads of the 400 KB hit do not fit in 600 KB.
    let mut left = 600 * 1024u64;
    let row = load_rows(&s.conn, &[id]).unwrap().remove(0);
    assert_eq!(check_row_budgeted(&dir, &row, true, &mut left), Verdict::Touched(stat_sig(&std::fs::metadata(dir.join("big.md")).unwrap())));
    assert_eq!(left, 200 * 1024);
    assert!(matches!(check_row_budgeted(&dir, &row, true, &mut left), Verdict::Changed(_)));
    assert_eq!(left, 200 * 1024, "nothing read past the budget");
    // The read itself is capped (a file that grew after its stat cannot be read past the cap).
    assert!(read_capped(&dir.join("big.md"), 300 * 1024).unwrap().is_none());
    assert_eq!(read_capped(&dir.join("big.md"), 400 * 1024).unwrap().map(|b| b.len()), Some(400 * 1024));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_query_hashes_its_hits_so_a_restored_mtime_cannot_hide_an_edit() {
    let dir = tmp("restore");
    write(&dir, "a.md", "# A\nbody one");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&dir).unwrap();
    let a = id_of(&s, "a.md");
    let recorded = stat_sig(&std::fs::metadata(dir.join("a.md")).unwrap());
    // Same size, different bytes, mtime put back to exactly what was recorded (cp -p, rsync -t).
    write(&dir, "a.md", "# A\nbody two");
    let f = std::fs::OpenOptions::new().write(true).open(dir.join("a.md")).unwrap();
    f.set_modified(std::time::UNIX_EPOCH + Duration::from_nanos(recorded.mtime_ns as u64)).unwrap();
    drop(f);
    assert_eq!(
        stat_sig(&std::fs::metadata(dir.join("a.md")).unwrap()),
        recorded,
        "fixture: the stat really is identical"
    );
    // The bulk scan is stat-based and cannot see it (documented limit; `recall index` hashes)...
    assert_eq!(s.scan_freshness(&dir, false, 10).unwrap().changed, 0);
    // ...but what a reader is shown is content-verified.
    assert_eq!(infos(&s, &dir, &[a])[0].stale, Some(true));
    // Past the per-query hash budget a hit is stat-checked: the same documented limit, bounded work.
    assert_eq!(annotate_with_budget(&s.conn, &dir, &[a], 0)[0].stale, Some(false));
    assert_eq!(s.note_path_event(&dir, "a.md", false).unwrap(), PathChange::Changed);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn rows_stamped_in_milliseconds_by_an_earlier_revision_are_treated_as_unstamped() {
    let dir = tmp("msrow");
    write(&dir, "a.md", "# A\nbody");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&dir).unwrap();
    s.conn.execute("UPDATE documents SET modified_at = 1700000000000", []).unwrap(); // ms-era value
    let a = id_of(&s, "a.md");
    let f = &infos(&s, &dir, &[a])[0];
    assert_eq!(f.stale, Some(false), "hashed, found identical");
    assert!(f.modified_at.is_none(), "no 1970 date shown for an unstamped row");
    let r = s.scan_freshness(&dir, false, 10).unwrap();
    assert_eq!(r.legacy_stamped, 1, "the scan re-stamps it in nanoseconds");
    assert!(infos(&s, &dir, &[a])[0].modified_at.is_some());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn queued_vector_patches_from_several_writers_are_all_collected_once() {
    let home = tmp("patch-home");
    refresh::queue_patch(Some(&home), &[3, 1]);
    refresh::queue_patch(Some(&home), &[2]);
    refresh::queue_patch(Some(&home), &[]);
    let mut got = refresh::take_patch(Some(&home));
    got.sort_unstable();
    assert_eq!(got, vec![1, 2, 3]);
    assert!(refresh::take_patch(Some(&home)).is_empty(), "consumed");
    std::fs::remove_dir_all(&home).ok();
}

/// Git's autocrlf rewrites every line ending in a working tree without the author touching a word.
/// That is not a change: the hit must stay fresh, a scan must not flag it, and a re-index must not
/// rewrite the body or queue the document for a re-embed. A real edit made at the same time must
/// still be caught.
#[test]
fn a_line_ending_flip_is_not_a_change() {
    for (tag, indexed_as, flipped_to) in [
        ("crlf-to-lf", "# W\r\nline one\r\nline two\r\n", "# W\nline one\nline two\n"),
        ("lf-to-crlf", "# W\nline one\nline two\n", "# W\r\nline one\r\nline two\r\n"),
    ] {
        let dir = tmp(tag);
        write(&dir, "w.md", indexed_as);
        let s = Store::open_at(Path::new(":memory:")).unwrap();
        s.ingest_dir(&dir).unwrap();
        fake_embed(&s, None);
        let id = id_of(&s, "w.md");
        assert!(queue(&s).is_empty(), "{tag}: control, nothing to embed after indexing");
        assert_eq!(infos(&s, &dir, &[id])[0].stale, Some(false), "{tag}: control, fresh before the flip");

        write(&dir, "w.md", flipped_to);
        set_mtime_ahead(&dir.join("w.md"), 30);
        let f = &infos(&s, &dir, &[id])[0];
        assert_eq!(f.stale, Some(false), "{tag}: a line-ending flip was reported as a change: {f:?}");

        let _ = s.scan_freshness(&dir, false, 1_000).unwrap();
        assert_eq!(s.freshness_counts().unwrap().changed, 0, "{tag}: the scan flagged a flip");

        s.ingest_dir(&dir).unwrap();
        assert!(queue(&s).is_empty(), "{tag}: re-indexing a flip queued a re-embed");
        assert_eq!(infos(&s, &dir, &[id])[0].stale, Some(false), "{tag}: fresh after re-index");

        // An edit that arrives with the flip is still an edit.
        write(&dir, "w.md", &flipped_to.replace("two", "TWO"));
        set_mtime_ahead(&dir.join("w.md"), 60);
        assert_eq!(infos(&s, &dir, &[id])[0].stale, Some(true), "{tag}: a real edit was masked by the flip");
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Only a UNIFORM flip is forgiven. A stray CR added to a CRLF file, or a mixed-ending file that is
/// then normalised, is an edit; and every path (query-time check, scan, re-index) must agree on it.
#[test]
fn only_a_uniform_line_ending_flip_is_forgiven() {
    assert!(same_text_modulo_line_endings("a\r\nb\r\n", "a\nb\n"));
    assert!(same_text_modulo_line_endings("a\nb\n", "a\r\nb\r\n"));
    assert!(!same_text_modulo_line_endings("a\r\n", "a\r\r\n"), "an added stray CR is an edit");
    assert!(!same_text_modulo_line_endings("a\r\nb\n", "a\nb\n"), "mixed endings are not a uniform flip");
    let rec = sha256_hex("a\r\n".as_bytes());
    assert!(hash_matches_modulo_line_endings("a\n", &rec));
    assert!(!hash_matches_modulo_line_endings("a\r\r\n", &rec), "stray CR must not match the CRLF hash");
    let mixed = sha256_hex("a\r\nb\n".as_bytes());
    assert!(hash_matches_modulo_line_endings("a\r\nb\n", &mixed), "byte-identical always matches");
    assert!(!hash_matches_modulo_line_endings("a\nb\n", &mixed), "normalised mixed file is changed");

    // Through the real paths: indexed mixed, then normalised -> changed in the check AND the re-index.
    let dir = tmp("mixed");
    write(&dir, "m.md", "# M\r\nline one\nline two\r\n");
    let s = Store::open_at(Path::new(":memory:")).unwrap();
    s.ingest_dir(&dir).unwrap();
    fake_embed(&s, None);
    let id = id_of(&s, "m.md");
    write(&dir, "m.md", "# M\nline one\nline two\n");
    set_mtime_ahead(&dir.join("m.md"), 30);
    assert_eq!(infos(&s, &dir, &[id])[0].stale, Some(true), "check: mixed -> normalised is a change");
    s.ingest_dir(&dir).unwrap();
    assert_eq!(queue(&s), vec![id], "re-index agrees: it is a change, so it is queued for re-embed");

    std::fs::remove_dir_all(&dir).ok();
}

// ── Passage-level check: a changed file whose shown excerpt is still in it is not stale ─────────

fn legacy_doc(name: &str, text: &str) -> (Connection, PathBuf) {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT, source TEXT, valid_from TEXT, content_sha256 TEXT, body TEXT);",
    )
    .unwrap();
    let dir = tmp(name);
    write(&dir, "n.md", text);
    conn.execute(
        "INSERT INTO documents VALUES (1, 'n.md', 'vault', '1700000000', ?1, ?2)",
        params![sha256_hex(text.as_bytes()), text],
    )
    .unwrap();
    (conn, dir)
}

const NOTE: &str = "# Ops\n\nIntro line.\n\nThe nightly backup runs at 02:00 with rclone to B2.\n\nOutro line.\n";
// What FTS5 snippet() would show: matched words in [ ], cut ends marked with …
const SNIP: &str = "…nightly [backup] runs at 02:00 with [rclone] to B2.…";

fn one(conn: &Connection, dir: &Path, excerpt: &[&str]) -> FreshInfo {
    let terms = query_terms("nightly backup rclone");
    annotate_with_budget_excerpts(conn, dir, &[1], QUERY_HASH_BUDGET_BYTES, excerpt, &terms).remove(0)
}

#[test]
fn an_edit_elsewhere_leaves_the_passage_fresh() {
    let (conn, dir) = legacy_doc("pass-elsewhere", NOTE);
    assert_eq!(one(&conn, &dir, &[SNIP]).stale, Some(false), "untouched file");
    write(&dir, "n.md", &NOTE.replace("Intro line.", "A completely different intro."));
    let f = one(&conn, &dir, &[SNIP]);
    assert_eq!(f.stale, Some(false), "{f:?}");
    assert!(f.stale_note.as_deref().unwrap().starts_with("file changed, this passage unchanged"), "{f:?}");
    assert!(f.file_modified_at.is_some());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_edit_inside_the_passage_is_stale() {
    let (conn, dir) = legacy_doc("pass-inside", NOTE);
    write(&dir, "n.md", &NOTE.replace("02:00", "03:30"));
    let f = one(&conn, &dir, &[SNIP]);
    assert_eq!(f.stale, Some(true), "{f:?}");
    assert!(f.stale_note.as_deref().unwrap().starts_with("this passage changed"), "{f:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_deleted_file_is_stale_whatever_the_excerpt() {
    let (conn, dir) = legacy_doc("pass-deleted", NOTE);
    std::fs::remove_file(dir.join("n.md")).unwrap();
    let f = one(&conn, &dir, &[SNIP]);
    assert_eq!((f.stale, f.deleted), (Some(true), true), "{f:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_line_ending_flip_does_not_change_the_passage_but_a_space_does() {
    let (conn, dir) = legacy_doc("pass-crlf", NOTE);
    // git autocrlf flip plus an edit elsewhere: the bytes differ, the passage does not.
    write(&dir, "n.md", &NOTE.replace("Intro line.", "Other intro.").replace('\n', "\r\n"));
    let f = one(&conn, &dir, &[SNIP]);
    assert_eq!(f.stale, Some(false), "{f:?}");
    assert!(f.stale_note.as_deref().unwrap().contains("this passage unchanged"));
    // A doubled space INSIDE the passage is a change (it is a different string in code).
    write(&dir, "n.md", &NOTE.replace("runs at", "runs  at"));
    assert_eq!(one(&conn, &dir, &[SNIP]).stale, Some(true));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn no_usable_excerpt_or_no_indexed_body_keeps_the_whole_file_rule() {
    let (conn, dir) = legacy_doc("pass-none", NOTE);
    write(&dir, "n.md", &NOTE.replace("Intro line.", "Different."));
    for excerpt in [&[][..], &[""][..], &["…B2.…"][..]] {
        let f = one(&conn, &dir, excerpt);
        assert_eq!(f.stale, Some(true), "excerpt {excerpt:?}: {f:?}");
        assert!(f.stale_note.as_deref().unwrap().ends_with("read the actual file."), "{f:?}");
        assert!(!f.stale_note.as_deref().unwrap().contains("this passage changed"), "{f:?}");
    }
    // The indexed body is gone (or no longer the hashed text): nothing to verify against.
    conn.execute("UPDATE documents SET body = 'something else'", []).unwrap();
    let f = one(&conn, &dir, &[SNIP]);
    assert_eq!(f.stale, Some(true), "{f:?}");
    assert!(!f.stale_note.as_deref().unwrap().contains("this passage"), "{f:?}");
    std::fs::remove_dir_all(&dir).ok();
}

fn pc(excerpt: &str, old: &str, new: &str) -> PassageCheck {
    passage_check(excerpt, old, new, &query_terms("backup rclone docs budget"))
}

#[test]
fn passage_check_clears_only_the_same_lines_in_the_same_place() {
    use PassageCheck::*;
    assert_eq!(pc(SNIP, NOTE, &NOTE.replace("Intro line.", "Other.")), Present);
    assert_eq!(pc("see the [docs] page for details", "Please see the docs page for details.", "Top line\nPlease see the docs page for details."), Present);
    assert_eq!(pc("", NOTE, NOTE), Unusable);
    assert_eq!(pc("…B2.…", NOTE, NOTE), Unusable);
}

#[test]
fn passage_check_realistic_edits_are_never_cleared() {
    use PassageCheck::*;
    let wrap = |l: &str| format!("first\n{l}\nlast\n");
    // The same text, but the line it sits on changed: a continued number, a longer name, a prefix.
    assert_eq!(pc("the upload limit 10", &wrap("Note: the upload limit 10 applies."), &wrap("Note: the upload limit 10.5 applies.")), Absent);
    assert_eq!(pc("the upload limit 10", &wrap("Note: the upload limit 10 applies."), &wrap("Note: the upload limit 100 applies.")), Absent);
    assert_eq!(pc("return api_key", &wrap("return api_key"), &wrap("return api_key_backup")), Absent);
    assert_eq!(pc("backup_database()", &wrap("    backup_database()"), &wrap("    # backup_database()")), Absent, "commented out");
    assert_eq!(pc("delete_all()", &wrap("    delete_all()"), &wrap("        delete_all()")), Absent, "deeper block");
    assert_eq!(pc("if ok:\n    delete_all()", &wrap("if ok:\n    delete_all()"), &wrap("    if ok:\n    delete_all()")), Absent);
    assert_eq!(pc("run backup now", &wrap("run backup now"), &wrap("> run backup now")), Absent, "moved into a quote");
    // The passage text itself changed.
    assert_eq!(pc("return [value]", &wrap("return [value]"), &wrap("return value")), Absent);
    assert_eq!(pc("print('a  b') ok", &wrap("print('a  b') ok"), &wrap("print('a b') ok")), Absent);
    assert_eq!(pc("First line  \nSecond line", &wrap("First line  \nSecond line"), &wrap("First line\nSecond line")), Absent);
    assert_eq!(pc("Backup runs nightly at 02:00", &wrap("Backup runs nightly at 02:00"), &wrap("backup runs nightly at 02:00")), Absent);
    assert_eq!(pc("- [x] rotate the keys now", &wrap("- [x] rotate the keys now"), &wrap("- [ ] rotate the keys now")), Absent);
    assert_eq!(pc("step one done\nstep two done\nstep three done", &wrap("step one done\nstep two done\nstep three done"), &wrap("step one done\nstep three done")), Absent);
    // A highlight collision: the note later gained literal brackets around a word the reader saw highlighted.
    assert_eq!(pc("see the [docs] page for details", &wrap("see the docs page for details"), &wrap("see the [docs] page for details")), Absent);
    // An inner ellipsis is matched literally, as one piece.
    assert_eq!(pc("Monthly [budget]…10", &wrap("Monthly budget…10"), &wrap("Monthly budget…20")), Absent);
    assert_eq!(pc("Monthly [budget]…10", &wrap("Monthly budget…10"), &format!("new\n{}", wrap("Monthly budget…10"))), Present);
    // The two readings of the shown text (as shown / highlight stripped) both exist in the old body:
    // the unchanged template line must not clear the edited command. Which one was shown is not
    // knowable, so the answer is "cannot tell" (the whole file is reported changed), not "gone".
    let old = "Production: run release --force\nTemplate: run [release] --force\n";
    let new = "Production: run release --dry-run\nTemplate: run [release] --force\n";
    assert_eq!(passage_check("run [release] --force", old, new, &query_terms("release")), Unusable);
    // A passage that appears twice in the indexed text cannot be told apart: not cleared, even if one copy survives.
    let twice = "## Production\nThe backup runs nightly at two and keeps seven days.\n## Staging\nThe backup runs nightly at two and keeps seven days.\n";
    let prod_edited = twice.replacen("at two", "at three", 1);
    assert_eq!(pc("The [backup] runs nightly at two and keeps seven", twice, &prod_edited), Unusable);
    // Either way the hit is still flagged stale; only the wording differs.
    let (conn, dir) = legacy_doc("pass-twice", twice);
    write(&dir, "n.md", &prod_edited);
    let f = one(&conn, &dir, &["The [backup] runs nightly at two and keeps seven"]);
    assert_eq!(f.stale, Some(true), "{f:?}");
    assert!(f.stale_note.as_deref().unwrap().starts_with("this file changed"), "{f:?}");
    std::fs::remove_dir_all(&dir).ok();
}

// ── False "this passage changed" after an edit elsewhere (an appended line) ────────────────────

const SOUR: &str = "# Sourdough\n\nFeed the starter twice a day. Bake at 240 C with steam for twenty minutes, then 220 C without.\n";
const SOUR_APPENDED: &str = "# Sourdough\n\nFeed the starter twice a day. Bake at 240 C with steam for twenty minutes, then 220 C without.\nUse rye flour to revive a sluggish starter.\n";

#[test]
fn a_stemmed_highlight_does_not_make_an_untouched_passage_look_changed() {
    use PassageCheck::*;
    // FTS5 is porter-stemmed: the query word "feeding" highlights "Feed", which is not a query term as typed.
    let snip = "# [Sourdough]\n\n[Feed] the [starter] twice a day. Bake at 240 C with…";
    let terms = query_terms("sourdough starter feeding");
    assert_eq!(passage_check(snip, SOUR, SOUR_APPENDED, &terms), Present);
    // The same reading still catches a real edit on those lines.
    assert_eq!(passage_check(snip, SOUR, &SOUR_APPENDED.replace("twice", "three times"), &terms), Absent);
    let old = "All the cities we ship to are listed here.\n";
    let plural = passage_check("the [cities] we ship to are listed", old, &format!("New top line\n{old}"), &query_terms("city list"));
    assert_eq!(plural, Present);
}

#[test]
fn an_excerpt_that_ends_at_a_line_end_is_not_judged_by_the_next_line() {
    use PassageCheck::*;
    // The vector channel shows the passage from its start, here the whole note with its final newline.
    assert_eq!(passage_check(SOUR, SOUR, SOUR_APPENDED, &[]), Present);
    assert_eq!(passage_check("Intro line.\n", NOTE, &NOTE.replace("Intro line.\n", "Intro line.\nAdded right after.\n"), &[]), Present);
    assert_eq!(passage_check("\n\nIntro line.", NOTE, &NOTE.replace("# Ops", "# Operations"), &[]), Present);
    // The other direction: the same excerpt shapes still catch an edit on their own lines.
    assert_eq!(passage_check(SOUR, SOUR, &SOUR_APPENDED.replace("240 C", "250 C"), &[]), Absent);
    assert_eq!(passage_check("Intro line.\n", NOTE, &NOTE.replace("Intro line.", "Intro line, edited."), &[]), Absent);
    // The newlines at the ends are still part of what must be found: a line put between the
    // passage and its neighbour, or a shown line that changed, is a change (review, PR #50).
    assert_eq!(passage_check("\n\nRelease approved", "Header\n\nRelease approved\n", "Header\nBLOCKED\nRelease approved\n", &[]), Absent);
    assert_eq!(passage_check("Release approved\n…\n", "Release approved\n…\n", "Release approved\nBLOCKED\n", &[]), Absent);
}

#[test]
fn a_bracketed_word_of_the_note_is_not_taken_for_a_highlight_of_a_different_word() {
    use PassageCheck::*;
    // `[station]` is the note's own text; "starter" is the query term. Treating it as a highlight
    // would move the check to the untouched second line and clear an edited first line (review, PR #50).
    let old = "Feed the starter [station] daily before baking and keep the culture warm.\nFeed the starter station daily before baking and keep the culture warm.\n";
    let new = old.replacen("[station]", "[NEVER]", 1);
    let got = passage_check("[Feed] the [starter] [station] daily before baking and keep the culture warm…", old, &new, &query_terms("feeding starter"));
    assert_ne!(got, Present, "an edited line must never be cleared");
    for (word, term, yes) in [("feed", "feeding", true), ("cities", "city", true), ("starter", "starter", true), ("running", "run", true),
                              ("station", "starter", false), ("start", "starter", true), ("pager", "page", true), ("stage", "staging", true), ("api", "apiaries", false), ("organize", "organization", false)] {
        assert_eq!(same_stem(word, term), yes, "{word} / {term}");
    }
}

#[test]
fn an_excerpt_that_fits_two_overlapping_places_cannot_be_placed() {
    use PassageCheck::*;
    let old = "run deploy\nrun deploy\nrun deploy\n";
    assert_eq!(passage_check("run deploy\nrun deploy", old, "DO NOT deploy\nrun deploy\nrun deploy\n", &[]), Unusable);
}

#[test]
fn an_excerpt_that_cannot_be_placed_in_the_indexed_text_is_not_called_gone() {
    use PassageCheck::*;
    // Not in the indexed text in any reading: nothing is known about the passage, so the whole-file
    // answer stands ("this file changed"), never "this passage changed".
    assert_eq!(passage_check("words that were never in the note at all", NOTE, &NOTE.replace("Intro", "Other"), &[]), Unusable);
    let (conn, dir) = legacy_doc("pass-unplaced", NOTE);
    write(&dir, "n.md", &NOTE.replace("Intro line.", "Different."));
    let f = one(&conn, &dir, &["words that were never in the note at all"]);
    assert_eq!(f.stale, Some(true), "{f:?}");
    assert!(f.stale_note.as_deref().unwrap().starts_with("this file changed"), "{f:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_two_section_note_edited_in_one_section_flags_only_that_section() {
    let two = "# Runbook\n\n## Staging\n\nStaging deploys run every weekday at 10:00.\n\n## Production\n\nProduction deploys need two approvals.\n";
    let (conn, dir) = legacy_doc("pass-two-sections", two);
    write(&dir, "n.md", &two.replace("two approvals", "three approvals"));
    let terms = query_terms("staging production deploys");
    let get = |ex: &str| annotate_with_budget_excerpts(&conn, &dir, &[1], QUERY_HASH_BUDGET_BYTES, &[ex], &terms).remove(0);
    let a = get("## [Staging]\n\n[Staging] [deploys] run every weekday at 10:00.");
    assert_eq!(a.stale, Some(false), "{a:?}");
    assert!(a.stale_note.as_deref().unwrap().starts_with("file changed, this passage unchanged"), "{a:?}");
    let b = get("## [Production]\n\n[Production] [deploys] need two approvals.");
    assert_eq!(b.stale, Some(true), "{b:?}");
    assert!(b.stale_note.as_deref().unwrap().starts_with("this passage changed"), "{b:?}");
    std::fs::remove_dir_all(&dir).ok();
}

// ── A bracketed word of the note that could also be a highlight: never a wrong "unchanged" ──────

#[test]
fn a_literal_bracket_that_could_be_a_highlight_never_clears_an_edited_line() {
    use PassageCheck::*;
    // Review round 2, PR #50. The note's own `[start]` stems like the query word "starter", and a
    // near-duplicate line has no brackets. Read as all-highlights, the excerpt sits only on the
    // untouched second line; read correctly, it is the first line, which was edited.
    let old = "Feed the starter [start] daily before baking and keep the culture warm.\nFeed the starter start daily before baking and keep the culture warm.\n";
    let new = old.replacen("[start]", "[NEVER]", 1);
    let terms = query_terms("feeding starter");
    let shown = "[Feed] the [starter] [start] daily before baking and keep the culture warm…";
    assert_eq!(passage_check(shown, old, &new, &terms), Unusable, "two places fit the excerpt: cannot tell");
    // Without the look-alike line there is one place, and the edit on it is seen.
    let one_old = "Feed the starter [start] daily before baking and keep the culture warm.\nOther line.\n";
    assert_eq!(passage_check(shown, one_old, &one_old.replacen("[start]", "[NEVER]", 1), &terms), Absent);
    assert_eq!(passage_check(shown, one_old, &format!("{one_old}Appended.\n"), &terms), Present);
}

#[test]
fn a_wikilink_whose_word_is_the_query_word_is_judged_on_its_own_line() {
    use PassageCheck::*;
    let old = "# Baking\n\nSee [[starter]] for the feeding schedule and the flour ratios.\n\nOther notes follow here.\n";
    let terms = query_terms("starter");
    // What FTS5 shows: the matched token bracketed inside the note's own double brackets.
    let shown = "…See [[[starter]]] for the feeding schedule and the flour ratios.…";
    // Edited inside the shown line: must not say unchanged.
    assert_ne!(passage_check(shown, old, &old.replace("flour ratios", "water ratios"), &terms), Present);
    assert_ne!(passage_check(shown, old, &old.replace("[[starter]]", "[[levain]]"), &terms), Present);
    // Only a line appended elsewhere: must not say the passage changed.
    assert_ne!(passage_check(shown, old, &format!("{old}One more line.\n"), &terms), Absent);
    // A checkbox and a markdown link on the shown line behave the same way.
    let todo = "- [x] feed the starter before the weekend bake\n- [ ] buy rye flour\n";
    let t = query_terms("starter feed");
    let s = "- [x] [feed] the [starter] before the weekend bake";
    assert_ne!(passage_check(s, todo, &todo.replacen("[x]", "[ ]", 1), &t), Present);
    assert_ne!(passage_check(s, todo, &format!("{todo}- [ ] clean the jar\n"), &t), Absent);
}

#[test]
fn an_overlapping_copy_on_disk_is_still_found() {
    use PassageCheck::*;
    // Review round 2, LOW: the passage is still there, one line further down, overlapping a copy.
    assert_eq!(passage_check("run deploy\nrun deploy", "run deploy\nrun deploy\n", "xrun deploy\nrun deploy\nrun deploy\n", &[]), Present);
}
