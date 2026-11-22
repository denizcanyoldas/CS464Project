"""Export publication artifacts from the one locked final-test evaluation."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def split_identity(split_path, frozen):
    raw = split_path.read_bytes()
    if sha256(raw).hexdigest() != frozen["split_sha256"]:
        raise ValueError("The exported split differs from the frozen test manifest.")
    manifest = json.loads(raw)
    membership = {
        "labels": manifest["labels"],
        "rows": sorted((row["path"], row["label"], row["split"], row["sha256"])
                       for row in manifest["records"]),
        "subsets": {
            seed: {fraction: sorted(paths) for fraction, paths in fractions.items()}
            for seed, fractions in manifest["subsets"].items()
        },
    }
    canonical = json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()
    return {"manifest_sha256": frozen["split_sha256"],
            "portable_membership_sha256": sha256(canonical).hexdigest(),
            "labels": len(membership["labels"]), "images": len(membership["rows"])}


def export(source, split_path, out):
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    frozen = json.loads((source / "frozen_run.json").read_text(encoding="utf-8"))
    identity = split_identity(split_path, frozen)
    labels = metrics["labels"]
    view = frozen["evaluation_view"]
    if metrics["partition"] != "test" or set(metrics["views"]) != {view}:
        raise ValueError("Expected one frozen final-test view.")
    score = metrics["views"][view]
    counts = np.asarray(score["confusion_counts"])
    supports = np.array([score["per_class"][label]["support"] for label in labels])
    if (counts.shape != (len(labels), len(labels))
            or not np.array_equal(counts.sum(axis=1), supports)
            or (supports <= 0).any()
            or supports.sum() != metrics["n_eval"]):
        raise ValueError("Confusion counts, support, and label map disagree.")
    out.mkdir(parents=True, exist_ok=False)
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (out / "split_identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    columns = ["label", "support", "precision", "recall", "f1"]
    ordered = sorted(labels, key=lambda label: (score["per_class"][label]["support"], label))
    with (out / "test_per_class.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows({"label": label, **score["per_class"][label]} for label in ordered)

    normalized = counts / supports[:, None]
    figure, axes = plt.subplots(figsize=(10.2, 9.5), layout="constrained")
    image = axes.imshow(normalized, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    ticks = range(len(labels))
    axes.set(xticks=ticks, yticks=ticks, ylabel="True label (support)",
             xlabel="Predicted label", title="Final test confusion by true class")
    axes.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    axes.set_yticklabels([f"{label} ({n})" for label, n in zip(labels, supports)], fontsize=7)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.78, format=PercentFormatter(1))
    colorbar.set_label("Share of each true class")
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"test_confusion.{suffix}", dpi=170)
    plt.close(figure)

    errors = counts.copy()
    np.fill_diagonal(errors, 0)
    ranking = sorted(
        ((int(errors[i, j]), labels[i], labels[j]) for i, j in zip(*np.nonzero(errors))),
        key=lambda row: (-row[0], row[1], row[2]),
    )[:10]
    with (out / "test_top_confusions.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["count", "true", "predicted"])
        writer.writerows(ranking)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    export(args.test, args.split, args.out)
