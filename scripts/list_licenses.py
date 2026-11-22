"""Print the crate/version/SPDX table used in THIRD_PARTY_NOTICES.md. stdin: `cargo metadata` JSON."""
import json
import sys

meta = json.load(sys.stdin)
rows = sorted({(p["name"], p["version"], p.get("license") or p.get("license_file") or "UNSPECIFIED")
               for p in meta["packages"] if p["name"] != "recall"})
print("| crate | version | licence (SPDX, from the crate manifest) |\n|---|---|---|")
for n, v, l in rows:
    print(f"| {n} | {v} | {l} |")
