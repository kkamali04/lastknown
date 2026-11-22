"""
CI Coverage Smoke Test for recall status --json.

Runs `recall status --json`, parses coverage_pct, and exits 1 if below threshold.

Environment variables:
    RECALL_EXE           Path to the recall binary (default: ./target/release/recall.exe)
    RECALL_MIN_COVERAGE  Minimum acceptable coverage_pct (default: 99.0)

Default floor is 99.0 (matches scripts/smoke-status.ps1): a handful of empty-path
`cli` scratch docs are structurally un-embedded, so live vault coverage tops out a
hair under 100% (e.g. 100998/100999). A 100.0 floor would be permanently RED and
mask real catastrophic coverage loss, which is what this smoke exists to catch.
"""

import json
import os
import subprocess
import sys


def main() -> int:
    exe = os.environ.get(
        "RECALL_EXE",
        os.path.join(os.path.dirname(__file__), "..", "target", "release", "recall.exe"),
    )
    min_coverage = float(os.environ.get("RECALL_MIN_COVERAGE", "99.0"))

    exe = os.path.normpath(exe)

    print(f"[smoke] recall exe : {exe}")
    print(f"[smoke] min coverage: {min_coverage}%")

    if not os.path.isfile(exe):
        print(f"FAIL  recall binary not found: {exe}", file=sys.stderr)
        return 1

    try:
        result = subprocess.run(
            [exe, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print("FAIL  recall status --json timed out after 30s", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"FAIL  could not launch recall: {exc}", file=sys.stderr)
        return 1

    if result.returncode != 0:
        print(f"FAIL  recall exited {result.returncode}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"FAIL  invalid JSON from recall status --json: {exc}", file=sys.stderr)
        print(f"      stdout was: {result.stdout!r}", file=sys.stderr)
        return 1

    if "coverage_pct" not in data:
        print("FAIL  'coverage_pct' key missing from JSON output", file=sys.stderr)
        print(f"      keys present: {list(data.keys())}", file=sys.stderr)
        return 1

    coverage = float(data["coverage_pct"])

    print(f"[smoke] coverage_pct = {coverage:.2f}%")
    print(f"[smoke] docs_valid   = {data.get('docs_valid', 'N/A')}")
    print(f"[smoke] docs_total   = {data.get('docs_total', 'N/A')}")
    print(f"[smoke] embeddings   = {data.get('embeddings', 'N/A')}")

    if coverage < min_coverage:
        print(
            f"FAIL  coverage {coverage:.2f}% is below threshold {min_coverage:.2f}%",
            file=sys.stderr,
        )
        return 1

    print(f"PASS  coverage {coverage:.2f}% >= {min_coverage:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
