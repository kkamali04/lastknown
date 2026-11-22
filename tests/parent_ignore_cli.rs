//! A vault under a parent repo whose .gitignore is `*` (a home-folder repo) was indexed partly,
//! with exit 0: rg applied the PARENT's ignore rules. Only the vault's OWN rules may hide notes,
//! and when they do, index says how many.
use std::path::Path;
use std::process::Command;

fn index(home: &Path, vault: &Path) -> std::process::Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_recall"));
    c.arg("index");
    c.env("USERPROFILE", home).env("HOME", home).env("RECALL_VAULT", vault);
    for k in ["RECALL_ABSTAIN", "RECALL_CHANNELS", "RECALL_EXCLUDE_EXTRA", "RECALL_DB"] {
        c.env_remove(k);
    }
    c.output().expect("run recall")
}

#[test]
fn parent_gitignore_star_does_not_hide_vault_notes() {
    let tmp = std::env::temp_dir().join(format!("recall-parent-ignore-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let (home, vault) = (tmp.join("home"), tmp.join("repo").join("vault"));
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(vault.join("sub")).unwrap();
    std::fs::create_dir_all(tmp.join("repo").join(".git")).unwrap();
    std::fs::write(tmp.join("repo").join(".gitignore"), "*\n").unwrap();
    for n in ["a", "b", "c", "sub/d", "sub/e"] {
        std::fs::write(vault.join(format!("{n}.md")), format!("# {n}\n\nnote {n}")).unwrap();
    }
    // The vault's OWN rule still applies, and is reported.
    std::fs::create_dir_all(vault.join("skipme")).unwrap();
    std::fs::write(vault.join(".gitignore"), "skipme/
").unwrap();
    std::fs::write(vault.join("skipme").join("x.md"), "# x

hidden").unwrap();

    let out = index(&home, &vault);
    let (so, se) = (String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{so} {se}");
    assert!(so.contains("Indexed 5 docs"), "all 5 visible notes, not just the top-level ones: {so} {se}");
    assert!(se.contains("1 note(s) are hidden by ignore rules"), "{se}");
    let _ = std::fs::remove_dir_all(&tmp);
}
