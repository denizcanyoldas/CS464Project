"""Refit the repository-root main baseline on this experiment's grouped folds."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train import fit, scores
from features import describe, prepare
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoolake.__main__ import partitions, read_split, validate_files


def features(root, rows):
    vectors = []
    for row in rows:
        with Image.open(root / row["path"]) as image:
            vectors.append(describe(prepare(image)))
    return np.stack(vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest, _ = read_split(args.split)
    labels, training, validation, _ = partitions(manifest, {"seed": 464, "fraction": 100})
    validate_files(args.data_root, training + validation)
    with threadpool_limits(limits=2):
        model = fit(features(args.data_root, training), [r["label"] for r in training])
        predicted = model.predict(features(args.data_root, validation))
    print(scores([r["label"] for r in validation], predicted, labels))
    with args.out.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(("path", "label", "main_prediction"))
        writer.writerows((r["path"], r["label"], y) for r, y in zip(validation, predicted))
