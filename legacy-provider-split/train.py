"""Train ZooLake baselines and evaluate on the publisher's cleaned validation set."""
import argparse
import hashlib
import json
import platform
import zipfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dataset import ARCHIVE_SHA256, SOURCE, audit, read_image, save_csv
from features import d4, describe, prepare, rotate

BUDGETS = (10, 25, 50, 100)
ANGLES = tuple(range(0, 360, 45))


def nested_ranks(labels, seed):
    rng = np.random.default_rng(seed)
    ranks = np.zeros(len(labels), dtype=int)
    for name in np.unique(labels):
        order = rng.permutation(np.flatnonzero(labels == name))
        ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def fit(features, labels, views=1):
    # Weight each original equally across experiments, including its eight D4 views.
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=800,
                                                                  tol=1e-5, solver="lbfgs"))
    pipeline.fit(features, labels, logisticregression__sample_weight=np.full(len(labels), 1 / views))
    if pipeline[-1].n_iter_.max() >= pipeline[-1].max_iter:
        raise RuntimeError("Logistic regression did not converge.")
    return pipeline


def scores(truth, predicted, classes):
    return dict(macro_f1=float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0)),
                accuracy=float(accuracy_score(truth, predicted)))


def read_features(archive, rows):
    return np.stack([describe(prepare(read_image(archive, r["path"]))) for r in rows])


def run(archive, out, seed):
    print("Auditing images and published splits...", flush=True)
    rows, excluded, original_counts = audit(archive)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    train_y = np.array([r["label"] for r in train])
    val_y = np.array([r["label"] for r in val])
    classes = np.unique(train_y)
    counts = Counter(train_y)
    if set(classes) != set(val_y) or set(classes) != {r["label"] for r in rows if r["split"] == "test"}:
        raise ValueError("Not all classes are represented in the splits.")
    ranks = nested_ranks(train_y, seed)
    for row in rows:
        row["within_class_train_rank"] = ""
    for row, rank in zip(train, ranks):
        row["within_class_train_rank"] = int(rank)
    save_csv(out / "manifest.csv", rows)
    save_csv(out / "excluded.csv", excluded)
    class_counts = Counter((r["split"], r["label"]) for r in rows)
    save_csv(out / "class_counts.csv", [dict(label=c, **{s: class_counts[s, c] for s in ("train", "val", "test")})
                                         for c in classes])

    print("Extracting clean training and validation features...", flush=True)
    train_x = read_features(archive, train)
    val_tiles = [prepare(read_image(archive, r["path"])) for r in val]
    val_x = np.stack([describe(tile) for tile in val_tiles])
    truth_rows = [dict(path=r["path"], label=r["label"]) for r in val]
    experiments, models = [], {}

    def record(key, model, budget, num_images, views):
        prediction = model.predict(val_x)
        for row, p in zip(truth_rows, prediction):
            row[key] = p
        experiments.append(dict(model=key, budget_percent=budget, training_images=num_images,
                                views_per_image=views, fit_rows=num_images * views,
                                **scores(val_y, prediction, classes)))
        models[key] = model
        print(key, experiments[-1], flush=True)

    for budget in BUDGETS:
        limits = np.array([max(1, (counts[c] * budget + 50) // 100) for c in train_y])
        indices = np.flatnonzero(ranks <= limits)
        record(f"plain_{budget}", fit(train_x[indices], train_y[indices]), budget, len(indices), 1)
        if budget == 10:
            views = np.stack([describe(v) for i in indices
                              for v in d4(prepare(read_image(archive, train[i]["path"])))])
            record("d4_10", fit(views, np.repeat(train_y[indices], 8), 8), 10, len(indices), 8)
            del views

    save_csv(out / "learning_curve.csv", experiments)
    rotations = []
    for angle in ANGLES:
        features = val_x if angle == 0 else np.stack([describe(rotate(tile, angle)) for tile in val_tiles])
        for key in ("plain_10", "d4_10", "plain_100"):
            prediction = models[key].predict(features)
            original = np.array([r[key] for r in truth_rows])
            rotations.append(dict(model=key, angle=angle, agreement=float(np.mean(original == prediction)),
                                  **scores(val_y, prediction, classes)))
            if angle:
                for row, p in zip(truth_rows, prediction):
                    row[f"{key}_rot{angle}"] = p
    save_csv(out / "rotations.csv", rotations)
    save_csv(out / "validation_predictions.csv", truth_rows)

    per_class = []
    for key in models:
        prediction = [r[key] for r in truth_rows]
        precision, recall, f1, supports = precision_recall_fscore_support(
            val_y, prediction, labels=classes, zero_division=0)
        for i, name in enumerate(classes):
            per_class.append(dict(model=key, label=name, support=int(supports[i]),
                                  precision=precision[i], recall=recall[i], f1=f1[i]))
    save_csv(out / "per_class.csv", per_class)
    cm = confusion_matrix(val_y, [r["plain_100"] for r in truth_rows], labels=classes)
    save_csv(out / "confusion.csv", [dict(actual=c, **dict(zip(classes, row))) for c, row in zip(classes, cm)])

    scaler, lr = models["plain_100"]
    np.savez_compressed(out / "baseline.npz", classes=classes, mean=scaler.mean_, scale=scaler.scale_,
                        coef=lr.coef_, intercept=lr.intercept_)
    # A validation example allows a live, offline demo with just the portable model.
    example = next(r for r in truth_rows if r["label"] == "asterionella" and r["plain_100"] == r["label"])
    (out / "demo.jpeg").write_bytes(archive.read(example["path"]))
    metadata = dict(source=SOURCE, archive_sha256=ARCHIVE_SHA256, seed=seed, published_counts=original_counts,
                    clean_counts=dict(Counter(r["split"] for r in rows)), classes=len(classes),
                    image_files=len(rows) + len(excluded), excluded_images=len(excluded),
                    duplicate_groups=len({r["pixel_sha256"] for r in excluded}), features=train_x.shape[1],
                    input_side=128, width_range=[min(r["width"] for r in rows), max(r["width"] for r in rows)],
                    height_range=[min(r["height"] for r in rows), max(r["height"] for r in rows)],
                    metric_split="val", demo_path=example["path"], demo_label=example["label"],
                    python=platform.python_version(), packages={p: version(p) for p in
                        ("numpy", "scipy", "scikit-learn", "opencv-python-headless", "Pillow", "threadpoolctl")})
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/zoolake.zip"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=464)
    args = parser.parse_args()
    with args.archive.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Archive checksum mismatch. Use the original Eawag ZooLake release.")
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive, threadpool_limits(limits=2):
        run(archive, args.out, args.seed)
