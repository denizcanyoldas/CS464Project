"""Command-line entry points for data preparation, training and held-out evaluation."""

import argparse
from collections import Counter
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import tomllib


CLASSICAL = ("lr", "svm", "xgb")
CNN = ("scratch", "efficientnet-b0")
MODEL_FILES = {**dict.fromkeys(CLASSICAL, "model.joblib"), **dict.fromkeys(CNN, "checkpoint.pt")}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Decode and inspect an extracted ZooLake release")
    audit.add_argument("--data-root", type=Path)
    audit.add_argument("--metadata", type=Path, help="Optional explicit path,label,split,group CSV")
    audit.add_argument("--group-by", choices=("frame", "three_minutes", "five_minutes", "ten_minutes", "hour", "day", "none"))
    audit.add_argument("--out", type=Path)

    split = commands.add_parser("split", help="Freeze feasible folds and nested training subsets")
    split.add_argument("--audit", type=Path)
    split.add_argument("--out", type=Path)
    split.add_argument("--strategy", choices=("provider", "stratified"))
    split.add_argument("--group-by", choices=("group", "acquisition_day"))
    split.add_argument("--split-seed", type=int)
    split.add_argument("--subset-seeds", type=int, nargs="+")
    split.add_argument("--fold-map", type=Path, help="Restore fixed group,split assignments from CSV")

    train = commands.add_parser("train", help="Fit on train and score on validation only")
    train.add_argument("--data-root", type=Path)
    train.add_argument("--split", type=Path)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--model", choices=CLASSICAL + CNN)
    train.add_argument("--seed", type=int)
    train.add_argument("--fraction", type=int, choices=(10, 25, 50, 100))
    train.add_argument("--augment", choices=("none", "d4"))
    train.add_argument("--tta", action=argparse.BooleanOptionalAction)
    train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    train.add_argument("--class-weight", choices=("none", "balanced"))
    for name in ("image-side", "threads", "epochs", "batch-size", "workers"):
        train.add_argument(f"--{name}", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--regularization-c", type=float)

    test = commands.add_parser("evaluate-test", help="Score a selected frozen run on the final holdout")
    test.add_argument("--data-root", type=Path)
    test.add_argument("--split", type=Path)
    test.add_argument("--run", type=Path, required=True)
    test.add_argument("--out", type=Path, required=True)
    test.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    test.add_argument("--view", choices=("single", "d4", "both"), help="Evaluation view chosen on validation")

    curve = commands.add_parser("plot-curve", help="Plot fixed-split validation macro-F1 across training budgets")
    curve.add_argument("--runs", type=Path, nargs="+", required=True)
    curve.add_argument("--out", type=Path, required=True)
    commands.add_parser("hardware", help="Check local CPU, CUDA and PyTorch Metal availability")
    return parser


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_split(path):
    snapshot = path.read_bytes()
    return json.loads(snapshot), hashlib.sha256(snapshot).hexdigest()


def data_root(args, config):
    return Path(args.data_root or os.getenv("ZOOLAKE_DATA_ROOT") or config["data"]["root"]).expanduser().resolve()


def validate_files(root, rows):
    if not root.is_dir():
        raise ValueError(f"ZooLake directory not found: {root}. Set --data-root to the extracted archive directory.")
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid relative image path: {relative}")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Missing ZooLake image: {path}. Use the same extraction root as the audit.")
        expected = row.get("file_sha256")
        if expected and sha256(path) != expected:
            raise ValueError(f"Image bytes changed since the audit: {path}. Re-audit the dataset.")


def validated_settings(config, args):
    settings = {k: v if (v := getattr(args, k, None)) is not None else default
                for k, default in config["training"].items()}
    for name in ("threads", "epochs", "batch_size", "learning_rate", "regularization_c"):
        if settings[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if settings["image_side"] < 16 or settings["workers"] < 0:
        raise ValueError("image_side must be at least 16 and workers must be nonnegative")
    return settings


def partitions(manifest, settings):
    labels = manifest["labels"]
    try:
        paths = set(manifest["subsets"][str(settings["seed"])][str(settings["fraction"])])
    except KeyError as error:
        raise ValueError("Training subset is missing for the requested seed and fraction. See the split manifest.") from error
    full_training = [row for row in manifest["records"] if row["split"] == "train"]
    training = [row for row in full_training if row["path"] in paths]
    validation = [row for row in manifest["records"] if row["split"] == "validation"]
    if {row["path"] for row in training} != paths:
        raise ValueError("Subset includes paths outside its training partition.")
    if {row["label"] for row in training} != set(labels) or {row["label"] for row in validation} != set(labels):
        raise ValueError("Train and validation must contain the entire frozen label vocabulary.")
    full_counts = Counter(row["label"] for row in full_training)
    rank = (len(labels) + 3) // 4
    cutoff = sorted(full_counts.values())[rank - 1]
    minority = sorted(label for label in labels if full_counts[label] <= cutoff)
    return labels, training, validation, minority


def versions():
    names = ("numpy", "Pillow", "scipy", "scikit-learn", "matplotlib", "torch", "torchvision", "xgboost")
    installed = {"python": platform.python_version(), "machine": platform.machine()}
    for name in names:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return installed


def report_scores(scores, partition):
    for mode, score in scores.items():
        print(f"{partition} {mode}: macro-F1 {score['macro_f1']:.4f}, "
              f"minority macro-F1 {score['minority_macro_f1']:.4f}, accuracy {score['accuracy']:.4f}")


def _destination(path):
    path = path.resolve()
    if path.exists():
        raise ValueError(f"Output already exists: {path}. Choose a new --out to preserve the previous run.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def fit(args, config, split_path):
    from .metrics import save_evaluation

    settings = validated_settings(config, args)
    manifest, split_hash = read_split(split_path)
    labels, training, validation, minority = partitions(manifest, settings)
    target = _destination(args.out)
    root = data_root(args, config)
    validate_files(root, training + validation)
    train_counts = dict(sorted(Counter(row["label"] for row in training).items()))
    model = settings["model"]
    print(f"Training {model} on {len(training):,} images and evaluating {len(validation):,} validation images.", flush=True)
    with tempfile.TemporaryDirectory(prefix=".run-", dir=target.parent) as temporary:
        output = Path(temporary) / "result"
        output.mkdir()
        start = time.perf_counter()
        if model in CLASSICAL:
            from .classical import train_classical
            probabilities = train_classical(model, training, validation, root, labels, settings, output)
        elif model in CNN:
            from .cnn import train_cnn
            probabilities = train_cnn(model, training, validation, root, labels, settings, output)
        else:
            raise ValueError(f"Unknown model: {model}")
        elapsed = time.perf_counter() - start
        scores = save_evaluation(output, labels, validation, probabilities, minority, train_counts, "validation")
        model_file = MODEL_FILES[model]
        provenance = {
            "config": settings, "split_sha256": split_hash, "model_file": model_file,
            "model_sha256": sha256(output / model_file), "versions": versions(),
            "training_images": len(training), "validation_images": len(validation),
            "fit_and_predict_seconds": elapsed,
        }
        (output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        output.rename(target)
    report_scores(scores, "validation")
    print(f"Artifacts: {target}")


def evaluate(args, config, split_path):
    from .metrics import save_evaluation

    run = read_json(args.run / "run.json")
    settings = run["config"]
    manifest, split_hash = read_split(split_path)
    if run["split_sha256"] != split_hash:
        raise ValueError("Split hash differs from the frozen run manifest.")
    if run["model_file"] != MODEL_FILES.get(settings["model"]):
        raise ValueError("Frozen run has an unexpected model filename.")
    if run["model_sha256"] != sha256(args.run / run["model_file"]):
        raise ValueError("Saved model hash differs from the frozen run manifest.")
    view = args.view or ("both" if settings["tta"] else "single")
    tta = view in {"d4", "both"}
    if tta and not settings["tta"]:
        raise ValueError("This frozen run did not evaluate D4 inference on validation.")
    labels, training, _, minority = partitions(manifest, settings)
    test = [row for row in manifest["records"] if row["split"] == "test"]
    if {row["label"] for row in test} != set(labels):
        raise ValueError("Final test must contain the entire frozen label vocabulary.")
    root = data_root(args, config)
    target = _destination(args.out)
    validate_files(root, test)
    if settings["model"] in CLASSICAL:
        from .classical import predict_classical
        probabilities = predict_classical(args.run, test, root, tta, settings["threads"])
    else:
        from .cnn import predict_cnn
        probabilities = predict_cnn(args.run, test, root, tta, args.device or settings["device"])
    if view != "both":
        probabilities = {view: probabilities[view]}
    train_counts = dict(sorted(Counter(row["label"] for row in training).items()))
    with tempfile.TemporaryDirectory(prefix=".test-", dir=target.parent) as temporary:
        output = Path(temporary) / "result"
        output.mkdir()
        scores = save_evaluation(output, labels, test, probabilities, minority, train_counts, "test")
        test_protocol = {**run, "evaluation_view": view}
        (output / "frozen_run.json").write_text(json.dumps(test_protocol, indent=2) + "\n", encoding="utf-8")
        output.rename(target)
    report_scores(scores, "test")
    print(f"Artifacts: {target}")


def hardware():
    info = {"platform": platform.platform(), "logical_cpus": os.cpu_count(), "versions": versions()}
    try:
        import torch
    except ImportError:
        info["pytorch"] = "unavailable; install requirements-cnn.txt"
    else:
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = torch.backends.mps.is_available()
    print(json.dumps(info, indent=2))


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        if args.command == "hardware":
            hardware()
            return 0
        if args.command == "plot-curve":
            from .metrics import save_learning_curve
            save_learning_curve(args.runs, _destination(args.out))
            print(f"Learning curve: {args.out}")
            return 0
        with args.config.open("rb") as source:
            config = tomllib.load(source)
        defaults = config["data"]
        if args.command == "audit":
            from .audit import audit_dataset
            output = args.out or Path(defaults["audit_dir"])
            report = audit_dataset(
                data_root(args, config), output, args.metadata,
                group_by=args.group_by or defaults["audit_group_by"],
            )
            if report.get("error"):
                print(report["error"])
            print(f"Audit {report['status']}: {output}")
            return 0 if report["status"] == "ok" else 2
        if args.command == "split":
            from .splits import create_splits
            options = config["splits"]
            output = args.out or Path(defaults["split_dir"])
            create_splits(
                args.audit or Path(defaults["audit_dir"]) / "audit.json",
                output, args.strategy or options["strategy"],
                args.split_seed if args.split_seed is not None else options["seed"],
                args.subset_seeds or options["subset_seeds"], group_by=args.group_by or options["group_by"],
                fold_map=args.fold_map,
            )
            print(f"Split artifacts: {output}")
            return 0
        split_path = args.split or Path(defaults["split_dir"]) / "splits.json"
        if args.command == "train":
            fit(args, config, split_path)
        else:
            evaluate(args, config, split_path)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
