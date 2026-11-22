"""Verify and summarize Deniz's paired validation runs on the team's frozen folds."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoolake.__main__ import partitions, read_json, read_split, sha256

# Published in final-study/test/split_identity.json. Includes all training subset memberships.
MEMBERSHIP = "686731c225aaa96ee13bcbeaf29035a3eee08541a2d15e59ee450d840e0e83fc"
BUDGETS = [(100, 464), (10, 464), (10, 465), (10, 466)]


def check_membership(manifest):
    rows = manifest["records"]
    content = dict(labels=manifest["labels"],
                   rows=sorted((r["path"],r["label"],r["split"],r["sha256"]) for r in rows),
                   subsets={s: {f: sorted(paths) for f,paths in fs.items()}
                            for s,fs in manifest["subsets"].items()})
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",",":")).encode()).hexdigest()
    if digest != MEMBERSHIP:
        raise ValueError("Split or nested training subsets differ from Mert's published membership.")
    return digest


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(split_path, runs, out):
    manifest, split_hash = read_split(split_path)
    digest = check_membership(manifest)
    labels, train, val, minority = partitions(manifest, dict(seed=464, fraction=100))
    expected = [(r["path"],r["label"]) for r in val]
    truth = [r["label"] for r in val]
    evidence = [dict(path=p, label=c) for p,c in expected]
    minority_idx = [labels.index(c) for c in minority]
    train_counts, val_counts = Counter(r["label"] for r in train), Counter(truth)
    per_class = [dict(label=c, training=train_counts[c], validation=val_counts[c], minority=c in minority)
                 for c in labels]

    def measure(predictions):
        cm = confusion_matrix(truth, predictions, labels=labels)
        class_f1 = 2 * cm.diagonal() / (cm.sum(axis=0) + cm.sum(axis=1))
        return (dict(macro_f1=class_f1.mean(), minority_macro_f1=class_f1[minority_idx].mean(),
                     accuracy=cm.trace()/cm.sum()), class_f1)

    scores, rotations, models = [], [], {}
    for fraction, seed in BUDGETS:
        families = ["lr","svm","xgb"] if fraction == 100 else ["lr"]
        for family in families:
            controls = []
            for model in (family, family+"-morph"):
                name = f"{model}-{fraction}-{seed}"
                run = runs/name
                meta, metrics = read_json(run/"run.json"), read_json(run/"metrics.json")
                config = meta["config"].copy()
                if (config.pop("model"),config["fraction"],config["seed"]) != (model,fraction,seed):
                    raise ValueError(f"Wrong model or training budget in {name}")
                controls.append(config)
                _, selected, _, _ = partitions(manifest, config)
                if (meta["split_sha256"] != split_hash or sha256(run/meta["model_file"]) != meta["model_sha256"]
                        or metrics["partition"] != "validation" or set(metrics["views"]) != {"single"}
                        or Counter(r["label"] for r in selected) != metrics["selected_training_counts"]
                        or metrics["minority_labels_from_full_training"] != minority):
                    raise ValueError(f"Run provenance differs in {name}")
                preds = csv_rows(run/"predictions.csv")
                if [(r["path"],r["true_label"]) for r in preds] != expected:
                    raise ValueError(f"Validation identities differ in {name}")
                ys = [r["predicted_label"] for r in preds]
                values, f1 = measure(ys)
                for k,v in values.items():
                    if not np.isclose(v, metrics["views"]["single"][k]):
                        raise ValueError(f"Cannot reproduce {k} for {name}")
                scores.append(dict(model=model, fraction=fraction, seed=seed, training=len(selected),
                                   validation=len(val), **values))
                for e,p in zip(evidence,ys):
                    e[name] = p
                if fraction == 100:
                    for e,v in zip(per_class,f1):
                        e[model] = v
                models[name] = meta["model_sha256"]
            if controls[0] != controls[1]:
                raise ValueError(f"Unmatched settings for {family}")
    main = csv_rows(runs/"main.csv")
    if [(r["path"],r["label"]) for r in main] != expected:
        raise ValueError("Legacy baseline rows differ.")
    main_preds = [r["main_prediction"] for r in main]
    values, f1 = measure(main_preds)
    scores.append(dict(model="main_lr", fraction=100, seed=464, training=len(train), validation=len(val), **values))
    for e,p in zip(evidence, main_preds):
        e["main_lr"] = p
    for e,v in zip(per_class,f1):
        e["main_lr"] = v
    for model in ("lr","lr-morph"):
        turns = csv_rows(runs/f"turns-{model}.csv")
        if ([(r["path"],r["label"]) for r in turns] != expected or
                [r["0"] for r in turns] != [e[f"{model}-100-464"] for e in evidence]):
            raise ValueError("Quarter-turn run does not match the 0-degree control.")
        for angle in ("0","90","180","270"):
            ys = [r[angle] for r in turns]
            values,_ = measure(ys)
            rotations.append(dict(model=model, angle=angle, **values,
                                  agreement=np.mean([r[angle]==r["0"] for r in turns])))
            if angle != "0":
                for e,p in zip(evidence,ys):
                    e[f"{model}-deg{angle}"] = p
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out/"scores.csv",scores)
    save_csv(out/"per_class_full.csv",per_class)
    save_csv(out/"validation_predictions.csv",evidence)
    save_csv(out/"quarter_turns.csv",rotations)
    (out/"provenance.json").write_text(json.dumps(dict(portable_membership_sha256=digest,
        local_manifest_sha256=split_hash, full_training=len(train), validation=len(val),
        models=models, packages=meta["versions"]), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("split","runs","out"):
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args()
    summarize(args.split,args.runs,args.out)
