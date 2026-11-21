"""Score a fitted grouped LR model on physical quarter-turns of validation canvases."""
import argparse
import csv
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoolake.__main__ import partitions, read_split, sha256, validate_files
from zoolake.classical import describe_view
from zoolake.images import d4_view, load_canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    manifest, split_hash = read_split(args.split)
    if split_hash != run["split_sha256"] or sha256(args.run / "model.joblib") != run["model_sha256"]:
        raise ValueError("Run, split and saved model must match.")
    labels, _, validation, _ = partitions(manifest, run["config"])
    validate_files(args.data_root, validation)
    bundle = joblib.load(args.run / "model.joblib")
    features = []
    for row in validation:
        canvas = load_canvas(args.data_root / row["path"], bundle["image_side"])
        features.append([describe_view(d4_view(canvas, k), bundle.get("morphology", False))
                         for k in range(4)])
    vectors = np.asarray(features)
    with threadpool_limits(limits=run["config"]["threads"]):
        indices = bundle["model"].predict(vectors.reshape(-1, vectors.shape[-1]))
    predictions = np.asarray(labels)[indices].reshape(len(validation), 4)
    with args.out.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(("path", "label", "0", "90", "180", "270"))
        writer.writerows((r["path"], r["label"], *p) for r, p in zip(validation, predictions))
