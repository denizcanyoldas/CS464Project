"""Evaluation with a fixed class taxonomy and support-aware plots."""

import csv
import json
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def summarize(labels, true_labels, predictions, minority):
    truth = np.asarray(true_labels)
    scores = {}
    for name, probabilities in predictions.items():
        if probabilities.shape != (len(truth), len(labels)):
            raise ValueError(f"Invalid probability shape for {name}: {probabilities.shape}")
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError(f"Invalid probabilities for {name}")
        predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=labels, zero_division=0,
        )
        scores[name] = {
            "macro_f1": float(f1.mean()),
            "accuracy": float(accuracy_score(truth, predicted)),
            "minority_macro_f1": float(f1[[label in minority for label in labels]].mean()),
            "per_class": {
                label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
                for label, p, r, f, n in zip(labels, precision, recall, f1, support)
            },
            "confusion_counts": confusion_matrix(truth, predicted, labels=labels).tolist(),
        }
    return scores


def save_evaluation(
    destination: Path, labels, rows, predictions, minority, train_counts, partition,
):
    from matplotlib import pyplot as plt

    scores = summarize(labels, [row["label"] for row in rows], predictions, minority)
    payload = {
        "partition": partition, "labels": labels, "n_eval": len(rows),
        "minority_labels_from_full_training": sorted(minority),
        "selected_training_counts": train_counts, "views": scores,
    }
    (destination / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (destination / "predictions.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["path", "true_label", "mode", "predicted_label", *[f"p:{label}" for label in labels]])
        for mode, probabilities in predictions.items():
            for row, probability in zip(rows, probabilities):
                writer.writerow([row["path"], row["label"], mode, labels[probability.argmax()], *probability])

    order = sorted(labels, key=lambda label: (train_counts[label], label))
    y = np.arange(len(order))
    figure, axes = plt.subplots(figsize=(7.3, max(3, 0.24 * len(order) + 1.8)), layout="constrained")
    width = 0.72 / len(scores)
    for index, (mode, score) in enumerate(scores.items()):
        offset = (index - (len(scores) - 1) / 2) * width
        axes.barh(y + offset, [score["per_class"][label]["f1"] for label in order],
                  height=width, label={"single": "One view", "d4": "Eight D4 views"}.get(mode, mode))
    support = next(iter(scores.values()))["per_class"]
    axes.set(yticks=y, yticklabels=[f"{label} ({support[label]['support']})" for label in order],
             xlim=(0, 1), xlabel="F1", title=f"{partition.title()} F1 by class (evaluation support)")
    axes.tick_params(axis="y", labelsize=8)
    axes.invert_yaxis()
    axes.grid(axis="x", alpha=0.25)
    if len(scores) > 1:
        axes.legend(loc="lower right")
    figure.savefig(destination / "per_class_f1.png", dpi=170)
    plt.close(figure)
    return scores


def save_learning_curve(runs, output):
    """Summarize fixed-split validation runs by model, view mode and label budget."""
    from matplotlib import pyplot as plt

    groups = {}
    model_settings = {}
    split_hashes = set()
    for run in runs:
        provenance = json.loads((run / "run.json").read_text(encoding="utf-8"))
        summary = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if summary["partition"] != "validation":
            raise ValueError("Learning curves require validation runs.")
        config = provenance["config"]
        model = config["model"]
        comparables = {k: v for k, v in config.items() if k not in {"seed", "fraction", "augment", "tta"}}
        if model_settings.setdefault(model, comparables) != comparables:
            raise ValueError(f"Run settings differ within {model} beyond seed, fraction, augmentation, and TTA.")
        split_hashes.add(provenance["split_sha256"])
        for view, scores in summary["views"].items():
            key = (model, config["augment"], view, config["fraction"])
            seeds = groups.setdefault(key, {})
            if config["seed"] in seeds:
                raise ValueError(f"Duplicate seed {config['seed']} for {key}.")
            seeds[config["seed"]] = scores
    if len(split_hashes) != 1:
        raise ValueError("Learning curves require runs using the same fixed split.")

    by_series = {}
    rows = []
    metric_names = ("macro_f1", "minority_macro_f1")
    for (model, augment, view, fraction), scores in sorted(groups.items()):
        stats = {}
        for metric in metric_names:
            values = [score[metric] for score in scores.values()]
            mean = statistics.mean(values)
            spread = statistics.stdev(values) if len(values) > 1 else None
            stats[metric] = (mean, spread)
            seeds = " ".join(map(str, sorted(scores)))
            rows.append((model, augment, view, fraction, len(values), seeds, metric, mean, spread))
        by_series.setdefault((model, augment, view), {})[fraction] = stats
    output.mkdir(parents=True)
    with (output / "learning_curve.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["model", "augmentation", "view", "train_percent", "seed_count", "seeds", "metric", "mean", "sd"])
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True, layout="constrained")
    for axis, metric, title in zip(axes, metric_names, ("All labels", "Minority labels")):
        for (model, augment, view), by_fraction in by_series.items():
            fractions = sorted(by_fraction)
            values = [by_fraction[f][metric] for f in fractions]
            axis.errorbar(fractions, [mean for mean, _ in values],
                          yerr=[spread or 0 for _, spread in values],
                          marker="o", capsize=3, label=f"{model} / train {augment} / eval {view}")
        axis.set(title=title, xlabel="Training fold used (%)", xticks=[10, 25, 50, 100], ylim=(0, 1))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Validation macro-F1")
    axes[1].legend(fontsize=8)
    figure.savefig(output / "learning_curve.png", dpi=170)
    plt.close(figure)
