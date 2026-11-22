//! Terminal output is plain text: no emoji or pictographs, and no escape bytes unless colour is on.
//! With colour on, every line that opens a colour closes it. Keyword channel only (no model).
use std::path::{Path, PathBuf};
use std::process::Command;

fn cmd(home: &Path, env: &[(&str, &str)], args: &[&str]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_recall"));
    c.args(args).env("USERPROFILE", home).env("HOME", home);
    for (k, _) in std::env::vars() {
        if k.starts_with("RECALL_") || ["NO_COLOR", "CLICOLOR_FORCE"].contains(&k.as_str()) {
            c.env_remove(k);
        }
    }
    c.envs(env.iter().copied());
    c
}

/// stdout then stderr, whatever the exit code.
fn recall(home: &Path, env: &[(&str, &str)], args: &[&str]) -> String {
    let out = cmd(home, env, args).output().expect("run recall");
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// Emoji, pictographs, dingbats, arrows and the joiners that build them. The em dash and the
/// middle dot of the scoring line are ordinary text and are not in here.
fn pictograph(c: char) -> bool {
    matches!(c as u32,
        0x2190..=0x21FF | 0x2300..=0x23FF | 0x2460..=0x24FF | 0x25A0..=0x27BF | 0x2900..=0x297F
        | 0x2B00..=0x2BFF | 0x200D | 0x20E3 | 0xFE0F | 0x1F000..=0x1FAFF)
}

fn assert_plain(what: &str, out: &str) {
    assert!(!out.contains('\x1b'), "{what}: escape byte in {out:?}");
    let bad: String = out.chars().filter(|c| pictograph(*c)).collect();
    assert!(bad.is_empty(), "{what}: pictograph(s) {bad:?} in {out}");
}

/// Every `ESC [ ... m` in the line, in order.
fn sgr(line: &str) -> Vec<&str> {
    line.split('\x1b').skip(1).map(|s| &s[..=s.find('m').expect("an escape that is not a colour")]).collect()
}

fn strip(out: &str) -> String {
    let mut plain = String::new();
    for (i, part) in out.split('\x1b').enumerate() {
        plain.push_str(if i == 0 { part } else { &part[part.find('m').unwrap() + 1..] });
    }
    plain
}

/// A vault with a fresh note, a note that goes stale, a note that is deleted, and a note whose
/// own text carries an escape sequence.
fn setup(name: &str) -> (PathBuf, PathBuf) {
    let tmp = std::env::temp_dir().join(format!("recall-plain-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let (home, vault) = (tmp.join("home"), tmp.join("vault"));
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&vault).unwrap();
    std::fs::write(vault.join("api.md"), "# API notes\n\nThe API port is 8080.\n").unwrap();
    std::fs::write(vault.join("garden.md"), "# Garden\n\nWater the tomato plants every morning.\n").unwrap();
    std::fs::write(vault.join("gone.md"), "# Lease\n\nThe lease renewal is due in March.\n").unwrap();
    std::fs::write(vault.join("esc.md"), "# \x1b[31mRed\n\nA zebra note with \x1b[31ma raw escape in it.\n").unwrap();
    recall(&home, &[], &["settings", "set", "vault", vault.to_str().unwrap()]);
    (home, vault)
}

const QUERIES: [&[&str]; 9] = [
    &["query", "-c", "fts5", "--", "tomato plants"],
    &["query", "-c", "fts5", "--", "API port"],          // stale after the edit below
    &["query", "-c", "fts5", "--", "lease renewal"],     // deleted file
    &["query", "-c", "fts5", "--", "zebra escape"],      // the note that carries an escape
    &["query", "-c", "fts5", "--", "qqqzzzxxx"],         // nothing found
    &["query", "-c", "nonsense", "--", "x"],             // an error
    &["status"],
    &["recent"],
    &["daemon", "status"],
];

#[test]
fn output_is_plain_without_colour_and_every_colour_is_closed_with_it() {
    let (home, vault) = setup("out");
    let index = recall(&home, &[], &["index"]);
    assert_plain("index", &index);
    assert!(index.contains("Indexed 4 docs") && index.contains("docs in index"), "{index}");
    for gone in ["WHAT I DID", "phantom", "tombstone", "0 updated", "0 moved"] {
        assert!(!index.contains(gone), "index still prints {gone:?}: {index}");
    }
    assert_plain("save", &recall(&home, &[], &["save", "remember the milk"]));

    std::thread::sleep(std::time::Duration::from_millis(1100));
    std::fs::write(vault.join("api.md"), "# API notes\n\nThe API port is 9090.\n").unwrap();
    std::fs::remove_file(vault.join("gone.md")).unwrap();

    for args in QUERIES {
        let plain = recall(&home, &[], args);
        assert_plain(&format!("{args:?}"), &plain);

        let forced: Vec<&str> = ["--color", "always"].iter().chain(args).copied().collect();
        let coloured = recall(&home, &[], &forced);
        for line in coloured.lines().filter(|l| l.contains('\x1b')) {
            let codes = sgr(line);
            assert_eq!(codes.last(), Some(&"[0m"), "{args:?}: line does not end its colour: {line:?}");
            // open, close, open, close: no span is left open inside the line either.
            assert!(codes.chunks(2).all(|p| p.len() == 2 && p[0] != "[0m" && p[1] == "[0m"), "{args:?}: {line:?}");
        }
        assert_eq!(strip(&coloured), plain, "{args:?}: colour changed the text");
    }
    let error = recall(&home, &[], &["--color", "always", "query", "-c", "nonsense", "--", "x"]);
    assert!(error.contains("\x1b[1;31merror:\x1b[0m"), "{error:?}");

    let stale = recall(&home, &[], &["--color", "always", "query", "-c", "fts5", "--", "API port"]);
    assert!(stale.contains("\x1b[1;33m!! STALE: this passage changed"), "{stale:?}");
    assert!(stale.contains("\x1b[1;33mWarning: 1 of 1 results point at files that changed"), "{stale:?}");
    assert!(stale.contains("\x1b[36m[API]\x1b[0m"), "{stale:?}");
    let deleted = recall(&home, &[], &["--color", "always", "query", "-c", "fts5", "--", "lease renewal"]);
    assert!(deleted.contains("\x1b[1;31m!! STALE: this file was deleted"), "{deleted:?}");

    // The environment rules, with stdout piped (not a terminal).
    let q = QUERIES[1];
    assert!(recall(&home, &[("CLICOLOR_FORCE", "1")], q).contains('\x1b'));
    assert_plain("NO_COLOR beats CLICOLOR_FORCE", &recall(&home, &[("CLICOLOR_FORCE", "1"), ("NO_COLOR", "1")], q));
    let never: Vec<&str> = ["--color", "never"].iter().chain(q).copied().collect();
    assert_plain("--color never beats CLICOLOR_FORCE", &recall(&home, &[("CLICOLOR_FORCE", "1")], &never));
    let always: Vec<&str> = ["--color", "always"].iter().chain(q).copied().collect();
    assert!(recall(&home, &[("NO_COLOR", "1")], &always).contains('\x1b'), "--color always beats NO_COLOR");

    // JSON is never coloured, whatever is forced.
    let json = recall(&home, &[], &["--color", "always", "query", "-c", "fts5", "--json", "--", "API port"]);
    assert_plain("--json", &json);
    assert!(json.trim_start().starts_with('['), "{json}");
}

#[test]
fn diagnostics_are_opt_in_and_query_headers_are_safe() {
    let (home, _vault) = setup("diagnostics");
    assert!(cmd(&home, &[], &["index"]).status().unwrap().success());
    let args = ["query", "-c", "fts5", "--no-daemon", "--", "API tomato"];
    let plain = recall(&home, &[], &args);
    assert!(!plain.contains("rank-sum") && !plain.contains("does not mention") && !plain.contains("(weak)"), "{plain}");
    let verbose = recall(&home, &[("RECALL_VERBOSE", "1")], &args);
    assert!(verbose.contains("rank-sum") && verbose.contains("does not mention"), "{verbose}");
    let out = recall(&home, &[], &["query", "-c", "fts5", "--no-daemon", "--verbose", "--", "API tomato"]);
    assert!(out.contains("rank-sum"), "{out}");
    let json_args = ["query", "-c", "fts5", "--no-daemon", "--json", "--", "API tomato"];
    let json = cmd(&home, &[], &json_args).output().unwrap();
    let verbose_json = cmd(&home, &[("RECALL_VERBOSE", "1")], &json_args).output().unwrap();
    assert!(json.status.success() && verbose_json.status.success());
    assert_eq!(json.stdout, verbose_json.stdout, "verbosity changed machine output");
    let hits: serde_json::Value = serde_json::from_slice(&json.stdout).unwrap();
    assert!(hits[0].get("rrf").is_some() && hits[0].get("lacks").is_some());

    let controls = recall(&home, &[], &["query", "-c", "fts5", "--", "API\r\u{8}\u{9b}\u{9d}port"]);
    assert!(!controls.chars().any(|c| c.is_control() && !matches!(c, '\n' | '\t')), "{controls:?}");
    // Synthetic secret-shaped value: the header must use the same output scrubber as hits.
    let secret = format!("ghp_{}", "Q".repeat(36));
    let out = recall(&home, &[], &["query", "-c", "fts5", "--", &secret]);
    assert!(!out.contains(&secret), "query header echoed a synthetic token");
    let split_secret = format!("ghp_{}\r{}", "Q".repeat(18), "Q".repeat(18));
    let out = recall(&home, &[], &["query", "-c", "fts5", "--", &split_secret]);
    assert!(!out.contains(&secret), "terminal normalization reconstructed a synthetic token");
    // Simulate a legacy/raw row to exercise the human title boundary independently of ingest scrubbing.
    let db = rusqlite::Connection::open(home.join(".recall").join("recall.db")).unwrap();
    assert_eq!(db.execute("UPDATE documents SET title = ?1 WHERE path LIKE '%api.md'", [&split_secret]).unwrap(), 1);
    drop(db);
    let out = recall(&home, &[], &["query", "-c", "fts5", "--", "API port"]);
    assert!(!out.contains(&secret) && !out.contains('\r'), "human title reconstructed a synthetic token");
}

/// Ctrl+C on a command that is still running on a terminal: the reset reaches the terminal and the
/// process still dies of the signal. Into a pipe nothing is written: a pipe holds data.
#[cfg(unix)]
#[test]
fn an_interrupt_resets_a_terminal_only_and_still_ends_the_process_by_the_signal() {
    use std::io::Read;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::process::{CommandExt, ExitStatusExt};
    use std::process::Stdio;
    let (home, _vault) = setup("int");
    for on_terminal in [true, false] {
        let (mut ours, mut theirs) = (0, 0);
        let made = unsafe {
            if on_terminal {
                libc::openpty(&mut ours, &mut theirs, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut())
            } else {
                let mut ends = [0; 2];
                let rc = libc::pipe(ends.as_mut_ptr());
                (ours, theirs) = (ends[0], ends[1]);
                rc
            }
        };
        assert_eq!(made, 0);
        let mut ui = cmd(&home, &[("TERM", "xterm")], &["--color", "always", "ui", "--no-open"]);
        // A test run started as a background job inherits SIGINT ignored, and recall keeps it ignored.
        unsafe {
            ui.pre_exec(|| {
                libc::signal(libc::SIGINT, libc::SIG_DFL);
                Ok(())
            })
        };
        // Keep a parent-owned far end until the output is drained: Darwin flushes queued PTY
        // output on the last slave close, which otherwise discards the reset before we read it.
        let far = unsafe { std::fs::File::from_raw_fd(theirs) };
        let mut child = ui.stdout(Stdio::from(far.try_clone().unwrap())).stderr(Stdio::null()).spawn().expect("spawn recall ui");
        drop(ui); // the command's copy; `far` deliberately remains open
        let mut near = unsafe { std::fs::File::from_raw_fd(ours) };
        let (mut seen, mut buf) = (Vec::new(), [0u8; 4096]);
        // The second line: the server is up and blocking.
        while !String::from_utf8_lossy(&seen).contains("only key)") {
            let n = near.read(&mut buf).expect("read recall ui");
            assert!(n > 0, "recall ui ended early: {:?}", String::from_utf8_lossy(&seen));
            seen.extend_from_slice(&buf[..n]);
        }
        unsafe { libc::kill(child.id() as i32, libc::SIGINT) };
        let status = child.wait().unwrap();
        // The child has exited, so all its writes are complete. Drain without waiting for EOF
        // from the parent-owned far end, preserving the descriptor's other status flags.
        unsafe {
            let flags = libc::fcntl(near.as_raw_fd(), libc::F_GETFL);
            assert_ne!(flags, -1, "F_GETFL: {}", std::io::Error::last_os_error());
            assert_ne!(libc::fcntl(near.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK), -1,
                       "F_SETFL: {}", std::io::Error::last_os_error());
        }
        loop {
            match near.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => seen.extend_from_slice(&buf[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => panic!("draining recall ui output: {e}"),
            }
        }
        drop(far);
        let seen = String::from_utf8_lossy(&seen);
        assert_eq!(seen.ends_with("\x1b[0m"), on_terminal, "terminal={on_terminal}: {seen:?}");
        assert_eq!(seen.contains('\x1b'), on_terminal, "terminal={on_terminal}: {seen:?}");
        assert_eq!(status.signal(), Some(libc::SIGINT), "{status:?}");
    }
}
