"""Render README illustrations from the public aggregate snapshot. Requires matplotlib.

Run from the repository root: python docs/images/render.py --preview-dir /path/to/previews
SVGs are checked-in documentation assets; this script is not part of installation/runtime.
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--preview-dir", type=Path)
args = parser.parse_args()
data = json.loads((ROOT / "bench/evidence/public-results.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
    "font.monospace": ["DejaVu Sans Mono", "Consolas", "Liberation Mono", "monospace"], "svg.fonttype": "none",
    "svg.hashsalt": "lastknown-readme-20261122", "savefig.dpi": 160})


def save(fig, name, title, description):
    svg = HERE / (name + ".svg")
    fig.savefig(svg, metadata={"Title": title, "Description": description,
        "Creator": "Lastknown documentation renderer", "Date": None}, facecolor=fig.get_facecolor())
    # Matplotlib emits platform line endings and path-line padding; keep repository bytes stable.
    svg.write_bytes(("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n").encode("utf-8"))
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview_dir / (name + ".png"), facecolor=fig.get_facecolor())
    plt.close(fig)


# A behavioral illustration, explicitly not a screenshot of an unimplemented UI.
fig = plt.figure(figsize=(12, 5.2), facecolor="#101d26")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 1200), ylim=(520, 0))
ax.axis("off")
ink, muted, green, amber = "#edf6f5", "#b1c4cb", "#66dfb6", "#ffc475"
ax.text(48, 49, "GOOD MEMORY STARTS WITH A SOURCE", color=green, fontsize=11, weight="bold", va="center")
ax.text(48, 98, "The note changed. Your agent should know.", color=ink, fontsize=25, weight="bold", va="center")
ax.text(48, 132, "Lastknown checks the indexed excerpt against the file on disk.", color=muted, fontsize=14, va="center")

cards = [
    (48, "01", "Index a note", "File: port 8080", "Result: port 8080", "FRESH", green,
     ["The source matches", "the indexed text."]),
    (438, "02", "Edit the source", "File: port 9090", "Result: port 8080", "STALE", amber,
     ["The old excerpt is flagged.", "Read the current file."]),
    (828, "03", "Reindex", "File: port 9090", "Result: port 9090", "FRESH AGAIN", green,
     ["The next result reflects", "the updated source."]),
]
for x, num, title, file, result, status, accent, explanation in cards:
    ax.add_patch(FancyBboxPatch((x, 174), 324, 284, boxstyle="round,pad=0,rounding_size=16",
        facecolor="#192d37", edgecolor="#34505a", linewidth=1))
    ax.text(x+22, 202, num, color=accent, fontsize=11, weight="bold", va="center")
    ax.text(x+22, 239, title, color=ink, fontsize=17, weight="bold", va="center")
    ax.text(x+22, 278, file, color=muted, fontsize=13, fontfamily="monospace", va="center")
    ax.text(x+22, 310, result, color=ink, fontsize=13, fontfamily="monospace", va="center")
    ax.plot([x+22, x+302], [331, 331], color="#34505a", lw=1)
    ax.text(x+22, 358, status, color=accent, fontsize=12, weight="bold", va="center")
    for i, line in enumerate(explanation):
        ax.text(x+22, 397+i*25, line, color=muted, fontsize=12.5, va="center")
for left in (383, 773):
    ax.add_patch(FancyArrowPatch((left, 316), (left+43, 316), arrowstyle="-|>",
        mutation_scale=18, linewidth=1.5, color=muted))
ax.text(48, 492, "Illustration of the portable demo. Source-text drift is not a factual-truth or answerability check.",
    color=muted, fontsize=11, va="center")
save(fig, "freshness", "Lastknown freshness workflow",
    "Illustration: index a note with port8080, edit its source to9090 and flag the old excerpt stale, then reindex to return9090.")


# Standard plot with a common zero-to-one axis; no generated statistical imagery.
fig = plt.figure(figsize=(12, 7.2), facecolor="#f5f8f7")
ax = fig.add_axes([0.165, 0.19, 0.70, 0.62], facecolor="#f5f8f7")
rows = data["beir"]["rows"]
colors = {"keyword": "#8c9ca8", "bge": "#5680b4", "fused": "#087c65"}
text_colors = {"keyword": "#4b6070", "bge": "#315a89", "fused": "#087c65"}
labels = {"keyword": "Keyword", "bge": "BGE", "fused": "Fused"}
offsets = {"keyword": 0.23, "bge": 0, "fused": -0.23}
for channel in ("keyword", "bge", "fused"):
    ys = [i + offsets[channel] for i in range(len(rows))]
    values = [row[channel] for row in rows]
    ax.barh(ys, values, height=0.17, color=colors[channel], label=labels[channel], zorder=3)
    for y, value in zip(ys, values):
        label = str(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
        ax.text(value+0.012, y, label, color=text_colors[channel], fontsize=11,
            va="center", fontfamily="monospace", weight="bold" if channel == "fused" else "normal")
ax.set_yticks(range(len(rows)), [row["dataset"] for row in rows], fontsize=13, color="#203e48")
ax.set_xlim(0, 1)
ax.set_ylim(4.55, -0.65)
ax.set_xticks([0, .2, .4, .6, .8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
              fontsize=11, color="#60747b")
ax.set_xlabel("nDCG@10  ·  higher is better", fontsize=11, labelpad=13, color="#60747b")
ax.xaxis.grid(True, color="#dbe5e2", linewidth=.8, zorder=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="both", length=0, pad=12)
fig.text(.05, .94, "What the retrieval audit found", fontsize=24, weight="bold", color="#163b3b")
fig.text(.05, .898, "Five BEIR datasets · TEST halves · Nov 12, 2026", fontsize=13, color="#60747b")
fig.legend(handles=ax.containers, labels=[labels[c] for c in colors], loc="upper right",
    bbox_to_anchor=(.95, .88), ncol=3, frameon=False, fontsize=11, handlelength=1.2, columnspacing=1.5)
fig.text(.05, .075, "Offline fusion audit, not a memory-product leaderboard. Four TEST halves were not fully unseen.",
    fontsize=10.5, color="#60747b")
fig.text(.05, .042, "Only SciFact was rerun for the Nov 22 alpha. Values, binary-capture differences and protocols: docs/EVALUATION.md.",
    fontsize=10.2, color="#60747b")
save(fig, "retrieval", "Five-set BEIR retrieval audit: keyword, BGE and fusion",
    "Historical offline fusion audit. Common0-to1axis. Figures read from bench/evidence/public-results.json; see evaluation protocol and limits.")
print("Rendered freshness.svg and retrieval.svg from the public evidence snapshot")
