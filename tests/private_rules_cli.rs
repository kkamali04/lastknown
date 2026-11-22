//! The private rules file through the real binary (unit tests bypass the machine's file):
//! a valid file decides `documents.tier` at index time; an invalid one stops recall at startup.
use std::path::{Path, PathBuf};
use std::process::Command;

fn recall(home: &Path, vault: &Path) -> std::process::Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_recall"));
    c.arg("index").env("USERPROFILE", home).env("HOME", home).env("RECALL_VAULT", vault);
    for k in ["RECALL_ABSTAIN", "RECALL_CHANNELS", "RECALL_EXCLUDE_EXTRA", "RECALL_TIERS"] {
        c.env_remove(k);
    }
    c.output().expect("run recall")
}

fn setup(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
    let tmp = std::env::temp_dir().join(format!("recall-private-rules-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let (home, vault) = (tmp.join("home"), tmp.join("vault"));
    std::fs::create_dir_all(home.join(".recall")).unwrap();
    // its own repo root, so rg does not apply a parent .gitignore (temp dirs can sit inside one)
    std::fs::create_dir_all(vault.join(".git")).unwrap();
    std::fs::create_dir_all(vault.join("third/party")).unwrap();
    std::fs::write(vault.join("third/party/copy.md"), "# copy\n\nthird-party text").unwrap();
    std::fs::write(vault.join("plan.md"), "# plan\n\nmy own note").unwrap();
    (tmp, home, vault)
}

/// `path, tier` of every live document, read straight from the store the binary wrote.
fn tiers(home: &Path) -> Vec<(String, Option<String>)> {
    let conn = rusqlite::Connection::open(home.join(".recall").join("recall.db")).unwrap();
    let mut st = conn.prepare("SELECT path, tier FROM documents WHERE valid_to IS NULL ORDER BY path").unwrap();
    let rows = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    rows.map(Result::unwrap).collect()
}

/// `recall_dev_dir` keeps that folder's eval/results/ out of query results; without it, it is an
/// ordinary folder.
#[test]
fn recall_dev_dir_keeps_its_eval_results_out_of_queries() {
    for (tag, rules, expect_hit) in [("dev", r#"{"recall_dev_dir": "dev/recall"}"#, false), ("nodev", "{}", true)] {
        let (tmp, home, vault) = setup(tag);
        std::fs::create_dir_all(vault.join("dev/recall/eval/results")).unwrap();
        std::fs::write(vault.join("dev/recall/eval/results/run.md"), "# run\n\nzebrafoo artifact").unwrap();
        std::fs::write(home.join(".recall").join("private-rules.json"), rules).unwrap();
        assert!(recall(&home, &vault).status.success());
        let mut q = Command::new(env!("CARGO_BIN_EXE_recall"));
        q.args(["query", "zebrafoo", "-j", "-c", "fts5", "--no-daemon"])
            .env("USERPROFILE", &home).env("HOME", &home).env("RECALL_VAULT", &vault).env("RECALL_ABSTAIN", "off");
        let out = q.output().unwrap();
        assert!(out.status.success(), "{tag}: {}", String::from_utf8_lossy(&out.stderr));
        let hits: Vec<serde_json::Value> = serde_json::from_slice(&out.stdout).expect("query --json");
        let found = hits.iter().any(|h| h["vault_path"].as_str().is_some_and(|p| p.ends_with("run.md")));
        assert_eq!(found, expect_hit, "{tag}: {hits:?}");
        let _ = std::fs::remove_dir_all(tmp);
    }
}

#[test]
fn a_valid_private_file_decides_tiers_and_an_invalid_one_refuses() {
    let (tmp, home, vault) = setup("ok");
    std::fs::write(
        home.join(".recall").join("private-rules.json"),
        r#"{"tiers": [{"tier": "VENDOR", "match": "contains", "value": "/party/"},
                      {"tier": "BRAIN", "match": "root_note_file"}]}"#,
    )
    .unwrap();
    let out = recall(&home, &vault);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(
        tiers(&home),
        vec![("plan.md".into(), Some("BRAIN".into())), ("third/party/copy.md".into(), Some("VENDOR".into()))]
    );

    let (tmp2, home2, vault2) = setup("bad");
    std::fs::write(
        home2.join(".recall").join("private-rules.json"),
        r#"{"tiers": [{"tier": "MINE", "match": "prefix", "value": "a/"}]}"#,
    )
    .unwrap();
    let out = recall(&home2, &vault2);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success() && err.contains("private rules file is invalid"), "{err}");
    assert!(!home2.join(".recall").join("recall.db").exists(), "refused before anything was written");

    // no file: the public defaults, which assume no vault layout (everything is the user's own)
    let (tmp3, home3, vault3) = setup("none");
    assert!(recall(&home3, &vault3).status.success());
    assert!(tiers(&home3).iter().all(|(_, t)| t.as_deref() == Some("BRAIN")), "{:?}", tiers(&home3));
    for t in [tmp, tmp2, tmp3] {
        let _ = std::fs::remove_dir_all(t);
    }
}
