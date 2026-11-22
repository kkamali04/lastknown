//! A note that only had a line appended was reported as "this passage changed" for some query wordings (a stemmed highlight, an excerpt ending at a line
//! end). Keyword channel only, so no model is needed. The vector channel's excerpt shape is covered
//! by the unit tests in src/store/freshness_tests.rs.
use std::path::Path;
use std::process::Command;

fn recall(home: &Path, args: &[&str]) -> String {
    let mut c = Command::new(env!("CARGO_BIN_EXE_recall"));
    c.args(args).env("USERPROFILE", home).env("HOME", home);
    for (k, _) in std::env::vars() {
        if k.starts_with("RECALL_") {
            c.env_remove(k);
        }
    }
    let out = c.output().expect("run recall");
    assert!(out.status.success(), "{args:?}: {}", String::from_utf8_lossy(&out.stderr));
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

fn ask(home: &Path, q: &str) -> String {
    recall(home, &["query", "-c", "fts5", "--top", "1", "--", q])
}

#[test]
fn an_appended_line_is_not_reported_as_a_changed_passage() {
    let tmp = std::env::temp_dir().join(format!("recall-stale-append-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let (home, vault) = (tmp.join("home"), tmp.join("vault"));
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&vault).unwrap();
    let note = vault.join("sourdough.md");
    let body = "# Sourdough\n\nFeed the starter twice a day. Bake at 240 C with steam for twenty minutes, then 220 C without.\n";
    std::fs::write(&note, body).unwrap();
    // Two sections far enough apart that one excerpt cannot span both.
    let filler = "Filler sentence that keeps the two sections well apart. ".repeat(12);
    let runbook = format!("# Runbook\n\n## Staging\n\nStaging deploys run every weekday at ten.\n\n{filler}\n\n## Production\n\nProduction deploys need two approvals.\n");
    std::fs::write(vault.join("runbook.md"), &runbook).unwrap();
    recall(&home, &["settings", "set", "vault", vault.to_str().unwrap()]);
    recall(&home, &["index"]);

    // 1. Append a line: the passage shown is untouched, whatever the query wording.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    std::fs::write(&note, format!("{body}Use rye flour to revive a sluggish starter.\n")).unwrap();
    for q in ["sourdough starter feeding", "feed the starter", "sourdough"] {
        let out = ask(&home, q);
        assert!(out.contains("this passage unchanged"), "query {q:?}: {out}");
        assert!(!out.contains("STALE"), "query {q:?}: {out}");
    }

    // 2. The other direction: an edit inside the shown passage is still flagged.
    std::fs::write(&note, body.replace("twice a day", "three times a day")).unwrap();
    let out = ask(&home, "sourdough starter feeding");
    assert!(out.contains("STALE: this passage changed"), "{out}");

    // 3. Two sections, only the second edited: the first is unchanged, the second is flagged.
    let rb = vault.join("runbook.md");
    std::fs::write(&rb, runbook.replace("two approvals", "three approvals")).unwrap();
    let a = ask(&home, "staging weekday");
    assert!(a.contains("this passage unchanged") && !a.contains("STALE"), "{a}");
    let b = ask(&home, "production approvals");
    assert!(b.contains("STALE: this passage changed"), "{b}");

    // 4. A deleted file is flagged as deleted.
    std::fs::remove_file(&rb).unwrap();
    let d = ask(&home, "production approvals");
    assert!(d.contains("STALE: this file was deleted"), "{d}");
    let _ = std::fs::remove_dir_all(&tmp);
}
