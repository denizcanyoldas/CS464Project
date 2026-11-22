"""Create the TA overview from the recorded predictions and original validation images."""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from dataset import read_image
from features import prepare

TEAL, ORANGE, INK, MUTED = "#006d70", "#b34b29", "#152c3c", "#4d606e"


def records(folder, filename):
    with (folder / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def style_axis(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c2ced2")
    ax.tick_params(length=0, colors=MUTED, pad=7)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dce5e7")
    ax.set_ylabel("Validation macro-F1", color=MUTED)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def make_figure(folder, archive):
    curve = records(folder, "learning_curve.csv")
    rotations = records(folder, "rotations.csv")
    predictions = records(folder, "validation_predictions.csv")
    meta = json.loads((folder / "run.json").read_text())
    plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
                         "axes.titlecolor": INK, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    fig.text(.065, .936, "Can a plankton classifier recognize rotated images?", size=24, weight="bold")
    fig.text(.065, .897, f"ZooLake  /  35 classes  /  {len(predictions):,} held-out validation images  /  "
             f"logistic regression  /  seed {meta['seed']}", color=MUTED, size=12)
    left = fig.add_axes([.083, .535, .365, .262])
    right = fig.add_axes([.565, .535, .365, .262])
    for ax in (left, right):
        style_axis(ax)
    left.set_title("Learning from more labeled images", loc="left", pad=15, weight="bold")
    unaug = [r for r in curve if r["model"].startswith("plain_")]
    x = [int(r["budget_percent"]) for r in unaug]
    y = [float(r["macro_f1"]) for r in unaug]
    left.plot(x, y, "o-", color=TEAL, lw=2.5)
    for xx, yy in zip(x, y):
        left.annotate(f"{yy:.1%}", (xx, yy), xytext=(0, 8), textcoords="offset points",
                      ha="center", color=TEAL, weight="bold", fontsize=10)
    left.set_xticks(x, [f"{v}%" for v in x])
    left.set_xlabel("Share of training images · original views only", labelpad=8)
    right.set_title("Rotation sensitivity with 10% training data", loc="left", pad=15, weight="bold")
    for key, label, color, marker in [("plain_10", "Original views", INK, "o"),
                                      ("d4_10", "Rotations + flips (D4)", ORANGE, "s")]:
        data = [r for r in rotations if r["model"] == key]
        right.plot([int(r["angle"]) for r in data], [float(r["macro_f1"]) for r in data],
                   marker=marker, label=label, color=color, lw=2, markersize=5)
    right.set_xticks(range(0, 360, 45), [f"{a}°" for a in range(0, 360, 45)])
    right.set_xlabel("Counterclockwise rotation at validation", labelpad=8)
    right.legend(loc="lower center", ncols=2, frameon=False, fontsize=9)
    for ax in (left, right):
        ax.set_ylim(0, 1)
    fig.text(.065, .414, "What the full-data model predicted", size=17, weight="bold")
    fig.text(.065, .382, "Four validation examples selected to illustrate both correct predictions and errors.",
             size=11, color=MUTED)
    examples = []
    for name, correct in [("asterionella", True), ("daphnia", True), ("cyclops", False), ("rotifers", False)]:
        pool = [r for r in predictions if r["label"] == name and (r["label"] == r["plain_100"]) == correct]
        if pool:
            examples.append(pool[0])
    for i, r in enumerate(examples):
        xx = .065 + i * .224
        ax = fig.add_axes([xx, .162, .193, .195], facecolor="black")
        ax.imshow(prepare(read_image(archive, r["path"])))
        ax.set_aspect("equal")
        ax.axis("off")
        fig.text(xx + .0965, .132, f"Label  ·  {r['label']}", ha="center", size=11)
        fig.text(xx + .0965, .107, f"Model  ·  {r['plain_100']}", ha="center", weight="bold",
                 color=TEAL if r["label"] == r["plain_100"] else ORANGE, size=11)
    fig.text(.065, .045, "Images: Kyathanahally et al. (2021), ZooLake / Eawag, CC BY 4.0.  "
             "Images resized and padded.  •  The final test set is reserved.", color=MUTED, size=9)
    fig.savefig(folder / "overview.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/zoolake.zip"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        make_figure(args.results, archive)
