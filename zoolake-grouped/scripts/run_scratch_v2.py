"""Run isolated Scratch CNN V2 validation experiments."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from zoolake.__main__ import data_root, partitions, validate_files  # noqa: E402
from zoolake.cnn_v2 import train_scratch_v2  # noqa: E402
from zoolake.metrics import save_evaluation  # noqa: E402


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def safe_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}. Choose a new output directory; "
            "existing experiments are never overwritten."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_one(config, manifest, split_hash, root, output_root, seed, fraction, augment):
    settings = dict(config["training"])
    settings.update(
        {
            "seed": seed,
            "fraction": fraction,
            "augment": augment,
            "model": "scratch",
        }
    )

    labels, training, validation, minority = partitions(manifest, settings)
    validate_files(root, training + validation)

    run_name = f"seed-{seed}_fraction-{fraction}_augment-{augment}"
    target = safe_output(output_root / run_name)
    train_counts = dict(sorted(Counter(row["label"] for row in training).items()))

    print(
        f"\nStarting {run_name}: {len(training):,} training images, "
        f"{len(validation):,} validation images, {len(labels)} labels.",
        flush=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".scratch-v2-", dir=target.parent
    ) as temporary:
        output = Path(temporary) / "result"
        output.mkdir()
        started = time.perf_counter()

        predictions = train_scratch_v2(
            training, validation, root, labels, settings, output
        )
        elapsed = time.perf_counter() - started

        scores = save_evaluation(
            output,
            labels,
            validation,
            predictions,
            minority,
            train_counts,
            "validation",
        )

        provenance = {
            "experiment": "scratch-cnn-v2",
            "config": settings,
            "split_sha256": split_hash,
            "model_file": "checkpoint.pt",
            "model_sha256": file_sha256(output / "checkpoint.pt"),
            "training_images": len(training),
            "validation_images": len(validation),
            "fit_and_predict_seconds": elapsed,
            "dataset_root": str(root),
        }
        (output / "run.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        output.rename(target)

    for view, result in scores.items():
        print(
            f"Validation {view}: macro-F1={result['macro_f1']:.4f}, "
            f"minority macro-F1={result['minority_macro_f1']:.4f}, "
            f"accuracy={result['accuracy']:.4f}",
            flush=True,
        )
    print(f"Saved: {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/scratch_v2.toml")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--pilot", action="store_true", help="Run only seed 464, 100%%, no augmentation.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fraction", type=int, choices=(10, 25, 50, 100), default=None)
    parser.add_argument("--augment", choices=("none", "d4"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    args = parser.parse_args()

    with args.config.open("rb") as stream:
        config = tomllib.load(stream)

    manifest, split_hash = load_split(args.split)
    root = (
        args.data_root
        or os.getenv("ZOOLAKE_DATA_ROOT")
        or config["data"]["root"]
    )
    root = Path(root).expanduser().resolve()
    output_root = Path(args.out_root or config["experiment"]["output_root"])
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    if args.device:
        config["training"]["device"] = args.device

    if args.pilot:
        seeds, fractions, augmentations = [464], [100], ["none"]
    elif any(value is not None for value in (args.seed, args.fraction, args.augment)):
        seeds = [args.seed if args.seed is not None else config["experiment"]["seeds"][0]]
        fractions = [args.fraction if args.fraction is not None else 100]
        augmentations = [
            args.augment if args.augment is not None else "none"
        ]
    else:
        seeds = config["experiment"]["seeds"]
        fractions = config["experiment"]["fractions"]
        augmentations = config["experiment"]["augmentations"]

    print(f"Dataset root: {root}")
    print(f"Split manifest: {args.split.resolve()}")
    print(f"Split SHA-256: {split_hash}")
    print(f"Output root: {output_root}")
    print(f"Device requested: {config['training']['device']}")

    completed = []
    for augment in augmentations:
        for fraction in fractions:
            for seed in seeds:
                completed.append(
                    run_one(
                        config,
                        manifest,
                        split_hash,
                        root,
                        output_root,
                        seed,
                        fraction,
                        augment,
                    )
                )

    print(f"\nCompleted {len(completed)} run(s).")
    for path in completed:
        print(path)


if __name__ == "__main__":
    main()
