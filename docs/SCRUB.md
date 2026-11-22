# Scrubbing: what it is for, and what it is not

recall removes secrets and personal data from what it stores and returns (`src/scrub.rs`). This page
states the threat model, so nobody reads the scrubber as more than it is.

## Threat model

The scrubber protects against **accidental** leaks: a key copied from a web page with a zero-width
space in it, a byte-order mark at the start of a file, ANSI colour codes around a token in a log,
CRLF line endings, a stray control character from a terminal.

It is **not** a defence against someone who crafts text to smuggle a credential past it. That person
already holds the credential, and a pattern-based scrubber cannot win that game. Do not rely on it as
a security boundary.

## What is stored, and when

There are two places the scrubber runs, and they are separate settings.

**What recall returns** is always scrubbed while the `scrub.secrets` / `scrub.pii` layers are on
(the default): every snippet, title and path that `recall query`, the daemon, the MCP server or
`recall ui` hands back.

**What recall stores** in `~/.recall/recall.db` depends on `scrub.store_redacted`:

- **on**: the note text is scrubbed before it is stored, so the body, the search index, the content
  hash and the embeddings are all made from the redacted text. The value is not in the database, and
  searching for it finds nothing. A hit carries `redactions: [{kind, line}]` pointing at the line in
  the original file.
- **off**: the database holds the note text as it is on disk. Results are still scrubbed on the way
  out, but anyone who can read `recall.db` can read the values.

Which one you get. The first row that applies decides:

| | Rule | `store_redacted` |
|---|---|---|
| 1 | `RECALL_SCRUB_STORE_REDACTED=0\|1` in the environment | the env value |
| 2 | `settings.json` states it (`recall settings set scrub.store_redacted on\|off`, or the key in the file) | what it says, including an explicit off |
| 3 | The database records that its index was started redacted | **on** |
| 4 | None of the above: an existing index made before this record existed, or one started unredacted on purpose | **off**, unchanged. recall does not change how an existing index is stored. |

How a database gets the record in row 3: the first time recall opens a database for writing and
finds nothing indexed in it (a new install in any order: `recall settings set vault`, the
installer, `recall ui` setup, `recall status`, the daemon, then the first `recall index`), it
decides how the index starts. Unless rows 1 or 2 say off, that is redacted, and recall

- writes the record into the database itself (table `profile_meta`), and
- copies the choice to `~/.recall/settings.json` as `"scrub": {"store_redacted": true}`, keeping
  your other settings.

The database record is the one that counts. If `settings.json` cannot be written (read-only),
cannot be read as JSON, or is deleted later, recall prints a warning when the copy fails and keeps
storing redacted text, on that run and on every later one.

Row 4 includes anyone whose first index was made by an earlier build with
`recall settings set vault` (or the installer, which runs it) before `recall index`: that order
left `store_redacted` off, so that index holds unredacted text. Opening such a database with this
version does not mark it.

Things that follow from the rules:

- **Turning it off on a redacted index** (`recall settings set scrub.store_redacted off`) is
  allowed. The next `recall index` re-reads every note, not only new ones, and stores all of them
  unredacted. Removing the key again brings redaction back (row 3).
- **`recall index --reset true`** keeps the state it found: a redacted index stays redacted, an
  explicit off stays off, an existing unredacted index stays unredacted.
- **An index that ends up empty** (for example a reset while the notes folder is empty) counts as
  nothing indexed: the next open starts it redacted unless rows 1 or 2 say off.

Check what is in force and which row decided it: `recall status --json` shows
`scrub.store_redacted` and `scrub.store_redacted_source` (`env ...`, `settings.json`,
`database (this index was started redacted)`, or `default (existing index, no stated choice)`).
`recall settings get scrub.store_redacted` shows only what the file says.

### Switching an existing index to redacted storage

```bash
recall daemon down                              # if you run one; it reads the setting when it starts
recall settings set scrub.store_redacted on
rm ~/.recall/recall.db ~/.recall/recall.db-wal ~/.recall/recall.db-shm   # Windows: del in %USERPROFILE%\.recall
recall index
recall embed
```

Delete the database rather than re-index over it. A plain `recall index` after the switch does
re-read every note and store it in redacted form (in a three-note test no planted value was left in
the file afterwards), but recall does not ask SQLite to wipe the pages it frees and never compacts
the file, so on a real index the old text can remain in unused pages. Deleting `recall.db` also
drops saved observations (`recall save`) and the vectors; `recall export > saved.json` first keeps
a copy of the observations, and `recall embed` rebuilds the vectors.

## What it does with hidden characters

Zero-width characters, bidi marks and overrides, soft hyphens, BOM, NUL and other control characters,
and ANSI/VT escape sequences (including the 8-bit C1 forms) can sit inside a credential without
changing how it looks. A pattern run over the raw text then misses it.

When a text contains such characters, recall reads it three ways: as it is (a NUL can be the word
boundary a key needs), with them removed (`sk-<ZWSP>proj-...` becomes one key), and with them replaced
by a space. The reading that redacts the most wins. If none redacts anything, the original text comes
back byte for byte.

- Joiners between two non-ASCII characters (emoji ZWJ sequences, Persian and Indic ZWNJ) are never
  touched, so a split key elsewhere in the text does not break them.
- If a redaction is made in a text with other hidden characters in the ASCII parts, those are removed
  from the returned text.
- Cost: about three scrubs of the text per pass (at most three passes, and a second pass only runs if
  the first found something), against one for text without hidden characters, which pays a single scan.

## Known limits

1. A text holding credentials that need **different** readings (one needs its NUL boundary kept,
   another needs a hidden character removed) can keep part of one key.
2. One credential with **both** a boundary-supplying and a key-splitting hidden character, for example
   `x<NUL>sk-<NUL>proj-...`, can keep part of the key. This is a crafted construction, not an accident.
3. Line endings: recall treats a uniform CRLF/LF flip of a file as "not changed". A file with mixed
   line endings that is then normalised, or one that gains a stray CR, is reported as changed.
4. On an existing unredacted index (row 4 above) with a `settings.json` that is not readable JSON,
   recall cannot tell whether you meant redaction on. It stores text unredacted and prints a warning
   on every run that stores text.
5. Two commands on a brand-new profile at the same instant, one of them pinned to
   `RECALL_SCRUB_STORE_REDACTED=0`: the pinned one may store a first unredacted document while the
   other marks the database as started redacted. Later runs redact; that one document stays as it
   was stored until its note is indexed again.

Credit: the hidden-character list and the escape-sequence pattern are ported from ai-memory
(MIT, Copyright (c) 2026 Fabio Akita), see `THIRD_PARTY_NOTICES.md`.
