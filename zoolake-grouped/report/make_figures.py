"""Render the report figures from the official audit and frozen LR validation run."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
INK = "#172c3c"
BLUE = "#26687a"
MUTED = "#748893"
LIGHT = "#e5edf0"


def read(path):
    return json.loads((WORKSPACE / path).read_text(encoding="utf-8"))


def style(axis):
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(LIGHT)
    axis.tick_params(colors=INK, length=0, labelsize=8)
    axis.tick_params(which="minor", length=0)
    axis.set_axisbelow(True)


def class_counts(audit):
    from collections import Counter

    counts = Counter(row["label"] for row in audit["records"])
    labels = sorted(counts, key=counts.get, reverse=True)
    figure, axis = plt.subplots(figsize=(7.0, 2.2), layout="constrained")
    axis.bar(range(35), [counts[label] for label in labels], color=[BLUE] + [MUTED] * 33 + [BLUE], width=0.76)
    axis.set(yscale="log", ylim=(8, 6800), xticks=[0, 9, 19, 29, 34],
             xticklabels=["1", "10", "20", "30", "35"], yticks=[10, 100, 1000],
             yticklabels=["10", "100", "1,000"], xlabel="Class rank by frequency", ylabel="Images (log scale)")
    axis.annotate("dinobryon · 3,321", (0, counts[labels[0]]), xytext=(4, 3),
                  textcoords="offset points", color=INK, fontsize=8, va="bottom")
    axis.annotate("chaoborus · 10", (34, counts[labels[-1]]), xytext=(-7, 44),
                  textcoords="offset points", color=INK, fontsize=8, ha="right",
                  arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.7})
    axis.grid(axis="y", color=LIGHT)
    style(axis)
    figure.savefig(HERE / "class_distribution.pdf", facecolor="white")
    plt.close(figure)


def minority_f1(metrics):
    validation = metrics["views"]["single"]["per_class"]
    counts = metrics["selected_training_counts"]
    order = sorted(metrics["minority_labels_from_full_training"], key=lambda label: (counts[label], label))
    figure, axis = plt.subplots(figsize=(7.0, 2.2), layout="constrained")
    positions = range(len(order))
    axis.barh(positions, [validation[label]["f1"] for label in order], color=BLUE, height=0.63)
    axis.set(xlim=(0, 0.55), xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], xlabel="Validation F1",
             yticks=positions, yticklabels=[f"{label}  ({validation[label]['support']})" for label in order])
    axis.invert_yaxis()
    axis.grid(axis="x", color=LIGHT)
    style(axis)
    for position, label in zip(positions, order):
        score = validation[label]["f1"]
        axis.text(score + 0.012, position, f"{score:.2f}", va="center", color=INK, fontsize=7.5)
    figure.savefig(HERE / "minority_f1.pdf", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="artifacts/audit/audit.json")
    parser.add_argument("--metrics", default="artifacts/runs/baseline-lr-s464/metrics.json")
    args = parser.parse_args()
    class_counts(read(args.audit))
    minority_f1(read(args.metrics))
