"""Summarize fixed-split ZooLake validation runs and matched D4 contrasts."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory


METRICS = ("macro_f1", "minority_macro_f1", "accuracy")
AUGMENTATIONS = ("none", "d4")
VIEWS = ("single", "d4")
BUDGETS = (10, 25, 50, 100)
CONDITION = ("model", "training_percent", "train_augmentation", "inference_view")
AGGREGATES = tuple(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "sd"))
DELTAS = tuple(f"delta_{metric}" for metric in METRICS)
DELTA_AGGREGATES = tuple(f"{delta}_{stat}" for delta in DELTAS for stat in ("mean", "sd"))


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _observations(paths):
    split_hash = None
    evaluation = None
    schedules = {}
    seen_conditions = set()
    subset_counts = {}
    individual = []
    for path in paths:
        try:
            run = _load_json(path / "run.json")
            metrics = _load_json(path / "metrics.json")
            if metrics["partition"] != "validation":
                raise ValueError("only validation runs are accepted")
            if split_hash is None:
                split_hash = run["split_sha256"]
            if not split_hash or run["split_sha256"] != split_hash:
                raise ValueError("split hashes differ across runs")

            config = run["config"]
            model, budget, aug, seed = (config[key] for key in ("model", "fraction", "augment", "seed"))
            if budget not in BUDGETS or aug not in AUGMENTATIONS or not isinstance(seed, int):
                raise ValueError("unknown training budget, augmentation or seed")
            controls = {
                key: value for key, value in config.items()
                if key not in {"seed", "fraction", "augment", "tta"}
            }
            original = schedules.setdefault(model, controls)
            differences = sorted(key for key in original.keys() | controls.keys()
                                 if original.get(key) != controls.get(key))
            if differences:
                raise ValueError(f"{model} settings differ in {', '.join(differences)}")
            identity = model, budget, aug, seed
            if identity in seen_conditions:
                raise ValueError(f"duplicate training run for {identity}")
            seen_conditions.add(identity)

            labels = tuple(metrics["labels"])
            minority = tuple(sorted(metrics["minority_labels_from_full_training"]))
            n_eval = metrics["n_eval"]
            if n_eval <= 0 or run["validation_images"] != n_eval or not set(minority) <= set(labels):
                raise ValueError("inconsistent validation inventory or minority labels")
            n_train = run["training_images"]
            train_counts = metrics["selected_training_counts"]
            if set(labels) != train_counts.keys() or sum(train_counts.values()) != n_train:
                raise ValueError("training class counts differ from the run inventory")
            if subset_counts.setdefault((budget, seed), train_counts) != train_counts:
                raise ValueError("training counts differ for the same budget and seed")

            all_views = metrics["views"]
            if set(all_views) != set(VIEWS if config["tta"] else ("single",)):
                raise ValueError("inference views do not match the recorded TTA setting")
            training_log = path / "training_log.json"
            log = _load_json(training_log) if training_log.is_file() else {}
            timing = log.get("inference_timing", {})
            for view, scores in sorted(all_views.items()):
                support = tuple(scores["per_class"][label]["support"] for label in labels)
                cohort = labels, minority, n_eval, support
                if sum(support) != n_eval:
                    raise ValueError("per-class support differs from the validation image count")
                if evaluation is None:
                    evaluation = cohort
                if evaluation != cohort:
                    raise ValueError("validation taxonomy, minority cohort or support differs across runs")
                values = {key: float(scores[key]) for key in METRICS}
                if any(not 0 <= value <= 1 for value in values.values()):
                    raise ValueError("all validation scores must be finite and in [0, 1]")
                individual.append({
                    "model": model, "training_percent": budget, "train_augmentation": aug,
                    "inference_view": view, "seed": seed, "training_images": n_train,
                    "validation_images": n_eval, "run": path.name, "split_sha256": split_hash,
                    "model_sha256": run["model_sha256"], **values,
                    "device": log.get("device", config.get("device")),
                    "fit_and_predict_seconds": run.get("fit_and_predict_seconds"),
                    "view_forward_seconds": timing.get(f"{view}_forward_seconds"),
                    "single_d4_prediction_agreement": metrics.get("single_d4_prediction_agreement"),
                })
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error
    if not individual:
        raise ValueError("at least one validation run is required")
    individual.sort(key=lambda row: tuple(row[key] for key in (*CONDITION, "seed")))
    return individual, split_hash, evaluation


def _stats(rows, names):
    output = {}
    for name in names:
        values = [row[name] for row in rows]
        output[f"{name}_mean"] = mean(values) if values else None
        output[f"{name}_sd"] = stdev(values) if len(values) > 1 else None
    return output


def _seed_list(seeds):
    return ",".join(str(seed) for seed in sorted(seeds))


def _scores_by_condition(individual):
    groups = defaultdict(dict)
    for row in individual:
        key = tuple(row[field] for field in CONDITION)
        groups[key][row["seed"]] = row
    return groups


def _grouped_rows(groups):
    return [
        {
            **dict(zip(CONDITION, key)),
            "seed_count": len(by_seed), "seeds": _seed_list(by_seed),
            **_stats(by_seed.values(), METRICS),
        }
        for key, by_seed in sorted(groups.items())
    ]


def _paired_rows(groups):
    pairs, contrasts = [], []
    configs = {(model, budget) for model, budget, _, _ in groups}
    for model, budget in sorted(configs):
        comparisons = [
            ("d4_training", "inference_view", view, (model, budget, "none", view), (model, budget, "d4", view))
            for view in VIEWS
        ] + [
            ("d4_tta", "train_augmentation", aug, (model, budget, aug, "single"), (model, budget, aug, "d4"))
            for aug in AUGMENTATIONS
        ]
        for contrast, control, held_value, baseline_key, treatment_key in comparisons:
            baseline = groups.get(baseline_key, {})
            treatment = groups.get(treatment_key, {})
            if not baseline and not treatment:
                continue
            same_seeds = baseline.keys() & treatment.keys()
            descriptor = {
                "contrast": contrast, "model": model, "training_percent": budget,
                "held_constant": control, "value": held_value,
            }
            matched = []
            for seed in sorted(same_seeds):
                before, after = baseline[seed], treatment[seed]
                if contrast == "d4_tta" and before["model_sha256"] != after["model_sha256"]:
                    raise ValueError("TTA comparison requires both views of the same fitted model")
                matched.append({
                    **descriptor, "seed": seed,
                    "baseline_run": before["run"], "treatment_run": after["run"],
                    **{f"delta_{metric}": after[metric] - before[metric] for metric in METRICS},
                })
            pairs.extend(matched)
            contrasts.append({
                **descriptor, "pair_count": len(matched), "paired_seeds": _seed_list(same_seeds),
                "baseline_only_seeds": _seed_list(baseline.keys() - treatment.keys()),
                "treatment_only_seeds": _seed_list(treatment.keys() - baseline.keys()),
                **_stats(matched, DELTAS),
            })
    return pairs, contrasts


def _csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _table(headings, rows):
    header = "| " + " | ".join(headings) + " |"
    divider = "| " + " | ".join("---" for _ in headings) + " |"
    return "\n".join([header, divider, *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def _estimate(row, name, signed=False):
    value, spread = row[f"{name}_mean"], row[f"{name}_sd"]
    if value is None:
        return "n/a"
    display = f"{value:+.3f}" if signed else f"{value:.3f}"
    return f"{display} ± {spread:.3f}" if spread is not None else display


def _markdown(individual, grouped, contrasts, split_hash, evaluation):
    labels, minority, n_eval, _ = evaluation
    return "\n\n".join([
        "# ZooLake grouped validation study",
        f"Fixed split SHA-256 `{split_hash}`. This study uses {n_eval:,} validation images "
        f"across {len(labels)} labels.",
        f"The {len(minority)} labels in the minority cohort are "
        + ", ".join(f"`{label}`" for label in minority) + ".",
        "## Mean scores by condition",
        "Each value is the mean and sample standard deviation across the listed seeds. "
        "Single runs have an undefined standard deviation. Seed variation describes this fixed split.",
        _table(
            ("Model", "Train %", "Train augmentation", "Inference", "Seeds (n)", "Macro-F1", "Minority macro-F1", "Accuracy"),
            (
                [*(row[k] for k in CONDITION), f"{row['seeds']} ({row['seed_count']})",
                 *(_estimate(row, metric) for metric in METRICS)]
                for row in grouped
            ),
        ),
        "## Matched D4 contrasts",
        "Training D4 subtracts the unaugmented model at the same inference view. "
        "TTA subtracts single-view inference from averaged D4 inference on the same fitted model. "
        "Only the listed pairs contribute to the deltas. Positive values favor D4.",
        _table(
            ("Contrast", "Model", "Train %", "Held fixed", "Pairs (n)", "Baseline only", "D4 only",
             "Δ macro-F1", "Δ minority macro-F1", "Δ accuracy"),
            (
                [row["contrast"], row["model"], row["training_percent"], f"{row['held_constant']} = {row['value']}",
                 f"{row['paired_seeds'] or '–'} ({row['pair_count']})",
                 row["baseline_only_seeds"] or "–", row["treatment_only_seeds"] or "–",
                 *(_estimate(row, name, signed=True) for name in DELTAS)]
                for row in contrasts
            ),
        ),
        "## Individual seeds",
        _table(
            ("Model", "Train %", "Train augmentation", "Inference", "Seed", "Training images",
             "Macro-F1", "Minority macro-F1", "Accuracy"),
            (
                [*(row[k] for k in CONDITION), row["seed"], row["training_images"],
                 *(f"{row[metric]:.3f}" for metric in METRICS)]
                for row in individual
            ),
        ),
        "[Learning curves](learning_curves.pdf) · [Grouped CSV](grouped_scores.csv) · "
        "[Per-seed contrasts](paired_deltas.csv) · [Pair coverage](paired_summary.csv)",
    ]) + "\n"


def _plot(rows, destination):
    from matplotlib.figure import Figure

    budgets_by_model = defaultdict(set)
    for row in rows:
        budgets_by_model[row["model"]].add(row["training_percent"])
    models = [model for model, budgets in budgets_by_model.items() if len(budgets) > 1]
    if not models:
        models = list(budgets_by_model)
    figure = Figure(figsize=(8.1, 2.45 * len(models) + 0.55), layout="constrained")
    axes = figure.subplots(len(models), 2, squeeze=False, sharex=True, sharey=True)
    line_styles = {
        ("none", "single"): ("#236885", "o", "-", "No augmentation · Single"),
        ("none", "d4"): ("#236885", "s", "--", "No augmentation · D4 TTA"),
        ("d4", "single"): ("#AE4E23", "o", "-", "D4 training · Single"),
        ("d4", "d4"): ("#AE4E23", "s", "--", "D4 training · D4 TTA"),
    }
    legend = {}
    for index, model in enumerate(models):
        for col, (metric, panel) in enumerate(zip(METRICS[:2], ("All labels", "Minority labels"))):
            axis = axes[index, col]
            for (aug, view), (color, marker, style, label) in line_styles.items():
                series = [row for row in rows if (row["model"], row["train_augmentation"],
                                                  row["inference_view"]) == (model, aug, view)]
                if series:
                    line = axis.errorbar(
                        [row["training_percent"] for row in series],
                        [row[f"{metric}_mean"] for row in series],
                        yerr=[row[f"{metric}_sd"] or 0 for row in series],
                        fmt=marker, linestyle=style, color=color, linewidth=1.3, markersize=4,
                        capsize=2, barsabove=True, label=label,
                    )
                    legend.setdefault(label, line)
            axis.set(title=f"{model} · {panel}", xticks=BUDGETS, ylim=(0, 1.04))
            axis.grid(axis="y", linewidth=0.5, alpha=0.32)
            axis.spines[["top", "right"]].set_visible(False)
            if col == 0:
                axis.set_ylabel("Validation macro-F1")
    figure.supxlabel("Training fold used (%)")
    figure.legend(legend.values(), legend.keys(), loc="outside upper center",
                  ncol=2, frameon=False, fontsize=8.5)
    destinations = destination / "learning_curves"
    figure.savefig(destinations.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(destinations.with_suffix(".png"), bbox_inches="tight", dpi=220)


def write_study(runs, out):
    if out.exists():
        raise ValueError(f"output directory already exists: {out}")
    individual, split_hash, evaluation = _observations(runs)
    groups = _scores_by_condition(individual)
    grouped = _grouped_rows(groups)
    pairs, contrasts = _paired_rows(groups)
    contrast_columns = ("contrast", "model", "training_percent", "held_constant", "value")

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".study-", dir=out.parent) as temporary:
        destination = Path(temporary) / "study"
        destination.mkdir()
        _csv(destination / "individual_scores.csv", tuple(individual[0]), individual)
        _csv(destination / "grouped_scores.csv", (*CONDITION, "seed_count", "seeds", *AGGREGATES), grouped)
        _csv(destination / "paired_deltas.csv",
             (*contrast_columns, "seed", "baseline_run", "treatment_run", *DELTAS), pairs)
        _csv(destination / "paired_summary.csv",
             (*contrast_columns, "pair_count", "paired_seeds", "baseline_only_seeds",
              "treatment_only_seeds", *DELTA_AGGREGATES), contrasts)
        (destination / "summary.md").write_text(
            _markdown(individual, grouped, contrasts, split_hash, evaluation), encoding="utf-8",
        )
        _plot(grouped, destination)
        destination.rename(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path, help="Validation run directories")
    parser.add_argument("--out", required=True, type=Path, help="New study output directory")
    args = parser.parse_args()
    try:
        write_study(args.runs, args.out)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Validation study: {args.out}")


if __name__ == "__main__":
    main()
