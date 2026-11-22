"""Freeze a validation-only model selection and evaluate it on the holdout."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from . import __main__ as pipeline


SEED = 464
FRACTION = 100
RULE = "Highest saved validation macro-F1, then single view, then run name and path."


def _json_snapshot(path):
    snapshot = path.read_bytes()
    return json.loads(snapshot), sha256(snapshot).hexdigest()


def _candidate(path, base, split_hash, labels, train_counts, validation_counts):
    run, run_hash = _json_snapshot(path / "run.json")
    settings = run["config"]
    if (settings["seed"], settings["fraction"]) != (SEED, FRACTION):
        raise ValueError(f"{path.name}: final candidates must use seed {SEED} and fraction {FRACTION}%.")
    if run["split_sha256"] != split_hash:
        raise ValueError(f"{path.name}: split hash differs from the supplied manifest.")
    if run["training_images"] != sum(train_counts.values()) or run["validation_images"] != sum(
        validation_counts.values()
    ):
        raise ValueError(f"{path.name}: image totals differ from the split manifest.")

    model_file = pipeline.MODEL_FILES.get(settings["model"])
    if model_file is None or run["model_file"] != model_file:
        raise ValueError(f"{path.name}: unexpected model artifact filename.")
    model_hash = pipeline.sha256(path / model_file)
    if model_hash != run["model_sha256"]:
        raise ValueError(f"{path.name}: model hash differs from run.json.")

    metrics, metrics_hash = _json_snapshot(path / "metrics.json")
    if "metrics_sha256" in run and metrics_hash != run["metrics_sha256"]:
        raise ValueError(f"{path.name}: metrics hash differs from run.json.")
    if metrics["partition"] != "validation" or metrics["labels"] != labels:
        raise ValueError(f"{path.name}: expected validation metrics with the frozen label order.")
    if metrics["n_eval"] != sum(validation_counts.values()) or metrics["selected_training_counts"] != train_counts:
        raise ValueError(f"{path.name}: reported image support differs from the split manifest.")
    expected_views = {"single", "d4"} if settings["tta"] else {"single"}
    if set(metrics["views"]) != expected_views:
        raise ValueError(f"{path.name}: saved inference modes differ from the run settings.")

    scores = {}
    for view, result in sorted(metrics["views"].items()):
        counts = {label: record["support"] for label, record in result["per_class"].items()}
        if counts != validation_counts:
            raise ValueError(f"{path.name}: per-class validation support differs for {view}.")
        score = result["macro_f1"]
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{path.name}: invalid saved macro-F1 for {view}.")
        scores[view] = score

    log_hash = None
    device = "cpu"
    if settings["model"] in pipeline.CNN:
        log_file = path / "training_log.json"
        if log_file.is_file():
            training_log, log_hash = _json_snapshot(log_file)
            device = training_log["device"]
        else:
            device = settings["device"]
        if not isinstance(device, str) or device.split(":", 1)[0] not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"{path.name}: cannot determine the actual CNN validation device.")

    return {
        "run": os.path.relpath(path, base), "model": settings["model"], "model_file": model_file,
        "model_sha256": model_hash, "run_sha256": run_hash, "metrics_sha256": metrics_hash,
        "training_log_sha256": log_hash, "device": device, "validation_macro_f1": scores,
    }


def _selection(runs, split_path, base):
    candidates = sorted(path.resolve() for path in runs)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Give at least one distinct candidate run directory.")
    manifest, split_hash = pipeline.read_split(split_path)
    labels, train, validation, _ = pipeline.partitions(manifest, {"seed": SEED, "fraction": FRACTION})
    if len(train) != sum(row["split"] == "train" for row in manifest["records"]):
        raise ValueError("The 100% subset must contain the entire training fold.")
    train_counts = Counter(row["label"] for row in train)
    validation_counts = Counter(row["label"] for row in validation)
    verified = [
        _candidate(path, base, split_hash, labels, train_counts, validation_counts)
        for path in candidates
    ]
    choices = (
        (candidate, view, score)
        for candidate in verified
        for view, score in candidate["validation_macro_f1"].items()
    )
    winner, view, score = min(
        choices,
        key=lambda item: (-item[2], item[1] != "single", Path(item[0]["run"]).name, item[0]["run"]),
    )
    return {
        "schema_version": 1,
        "selection_rule": RULE,
        "seed": SEED,
        "training_fraction": FRACTION,
        "split_sha256": split_hash,
        "candidates": verified,
        "chosen": {**winner, "evaluation_view": view, "chosen_validation_macro_f1": score},
    }


def freeze(runs, split_path, output):
    output = output.resolve()
    result = _selection(runs, split_path, output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".freeze-", dir=output.parent) as temporary:
        staged = Path(temporary) / "selection.json"
        staged.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        # Windows rename is atomic and refuses to replace an existing selection.
        if os.name == "nt":
            staged.rename(output)
        else:
            os.link(staged, output)
    chosen = result["chosen"]
    print(f"Frozen {chosen['run']} ({chosen['evaluation_view']}), "
          f"validation macro-F1 {chosen['chosen_validation_macro_f1']:.4f}.")
    print(f"Selection: {output}")


def evaluate(selection_path, split_path, data_root, output):
    selection_path = selection_path.resolve()
    snapshot = selection_path.read_bytes()
    frozen = json.loads(snapshot)
    base = selection_path.parent
    runs = [base / candidate["run"] for candidate in frozen["candidates"]]
    verified = _selection(runs, split_path, base)
    if verified != frozen:
        raise ValueError("Frozen selection differs from the verified candidate hashes or selection rule.")

    chosen = frozen["chosen"]
    output = pipeline._destination(output)
    with TemporaryDirectory(prefix=".frozen-test-", dir=output.parent) as temporary:
        staged = Path(temporary) / "result"
        args = SimpleNamespace(
            run=base / chosen["run"], out=staged, data_root=data_root,
            view=chosen["evaluation_view"], device=chosen["device"],
        )
        pipeline.evaluate(args, {}, split_path)
        (staged / "selection.json").write_bytes(snapshot)
        staged.rename(output)
    print(f"Frozen test artifacts: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze", help="Choose one run and view using saved validation metrics")
    freezing.add_argument("--runs", type=Path, required=True, nargs="+")
    freezing.add_argument("--split", type=Path, required=True)
    freezing.add_argument("--out", type=Path, required=True)
    evaluation = commands.add_parser("evaluate", help="Evaluate the selection on test without fitting")
    evaluation.add_argument("--selection", type=Path, required=True)
    evaluation.add_argument("--split", type=Path, required=True)
    evaluation.add_argument("--data-root", type=Path, required=True)
    evaluation.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            freeze(args.runs, args.split, args.out)
        else:
            evaluate(args.selection, args.split, args.data_root, args.out)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
