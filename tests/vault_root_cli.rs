//! CLI-level regression for 2026-11-06: `recall index --path-list` with vault-relative paths and
//! no RECALL_VAULT resolved them under the absent default `~/recall-notes`, skipped every one and
//! exited 0 ("Indexed 0 docs"). It must refuse before touching the store, and work once the vault
//! is named.
use std::path::Path;
use std::process::Command;

fn recall(home: &Path, vault: Option<&Path>, list: &Path) -> std::process::Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_recall"));
    c.args(["index", "--path-list"]).arg(list);
    c.env("USERPROFILE", home).env("HOME", home).env_remove("RECALL_VAULT");
    for k in ["RECALL_ABSTAIN", "RECALL_CHANNELS", "RECALL_EXCLUDE_EXTRA", "RECALL_DB"] {
        c.env_remove(k);
    }
    if let Some(v) = vault {
        c.env("RECALL_VAULT", v);
    }
    c.output().expect("run recall")
}

#[test]
fn path_list_index_refuses_a_missing_vault_and_works_with_one() {
    let tmp = std::env::temp_dir().join(format!("recall-vault-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let (home, vault) = (tmp.join("home"), tmp.join("vault"));
    let rel = "notes/.captures/2026-11-06/session.md";
    std::fs::create_dir_all(vault.join("notes/.captures/2026-11-06")).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(vault.join(rel), "# Session\n\ncaptured text").unwrap();
    let list = tmp.join("list.txt");
    std::fs::write(&list, rel).unwrap();

    let out = recall(&home, None, &list);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "must refuse: {stderr}");
    assert!(stderr.contains("vault root not found") && stderr.contains("RECALL_VAULT"), "{stderr}");
    assert!(!home.join(".recall").join("recall.db").exists(), "refused before the store was opened");

    let out = recall(&home, Some(&vault), &list);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{stdout} {}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("Indexed 1 docs"), "{stdout}");
    let _ = std::fs::remove_dir_all(&tmp);
}
