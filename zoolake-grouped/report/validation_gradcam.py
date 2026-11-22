"""Explain the most frequent single-view validation confusions of a fitted EfficientNet-B0."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoolake.__main__ import partitions, read_split, sha256, validate_files
from zoolake.cnn import RoiImages, build_model, resolve_device
from zoolake.images import load_canvas


def class_maps(model, tensor, sample, targets):
    # The final convolutional block feeds global pooling and the classification head.
    activations = model.features(tensor)
    logits = model.classifier(torch.flatten(model.avgpool(activations), 1))
    maps = []
    for index, target in enumerate(targets):
        gradient, = torch.autograd.grad(logits[sample, target], activations,
                                        retain_graph=index + 1 < len(targets))
        weights = gradient[sample:sample+1].mean(dim=(2, 3), keepdim=True)
        weighted = (weights * activations[sample:sample+1]).sum(1, keepdim=True).relu()
        smooth = F.interpolate(weighted.detach(), size=tensor.shape[-2:], mode="bilinear",
                               align_corners=False)[0, 0].cpu().numpy()
        maps.append(smooth / smooth.max() if smooth.max() > 0 else smooth)
    return logits.detach().softmax(1)[sample].cpu().numpy(), maps, list(activations.shape[-2:])


def generate(run_dir, split, root, out):
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest, split_hash = read_split(split)
    checkpoint = run_dir / "checkpoint.pt"
    if (meta["config"]["model"] != "efficientnet-b0" or split_hash != meta["split_sha256"]
            or sha256(checkpoint) != meta["model_sha256"]):
        raise ValueError("Expected the matching EfficientNet run, checkpoint and split.")
    labels, _, validation, _ = partitions(manifest, meta["config"])
    with (run_dir / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["mode"] == "single"]
    if [(r["path"], r["true_label"]) for r in rows] != [(r["path"], r["label"]) for r in validation]:
        raise ValueError("Saved predictions must match every validation row in manifest order.")
    counts = Counter((r["true_label"], r["predicted_label"]) for r in rows
                     if r["true_label"] != r["predicted_label"])
    ranking = sorted(counts, key=lambda key: (-counts[key], key))
    selected = ranking[:3]
    requested = ("eudiaptomus", "cyclops")
    if requested in counts and requested not in selected:
        selected.append(requested)
    if not selected:
        raise ValueError("No validation confusions to explain.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["labels"] != labels or state["model"] != "efficientnet-b0":
        raise ValueError("Checkpoint label order or model differs from manifest.")
    torch.set_num_threads(state["threads"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = resolve_device("auto")
    model = build_model("efficientnet-b0", len(labels)).to(device)
    model.load_state_dict(state["weights"])
    model.eval()
    dataset = RoiImages(validation, root, labels, state["image_side"])
    positions = {r["path"]: i for i,r in enumerate(validation)}
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(selected), 3, squeeze=False, figsize=(9, 2.6 * len(selected) + 1.3))
    examples = []
    truth_support = Counter(r["true_label"] for r in rows)
    for i, (true_label, predicted_label) in enumerate(selected):
        row = min((r for r in rows if (r["true_label"], r["predicted_label"]) == (true_label, predicted_label)),
                  key=lambda r: r["path"])
        idx = positions[row["path"]]
        start = idx // state["batch_size"] * state["batch_size"]
        stop = min(start + state["batch_size"], len(validation))
        validate_files(root, validation[start:stop])
        image = np.asarray(load_canvas(root / row["path"], state["image_side"]))
        # Replay the original evaluation batch to match the convolution kernels and probabilities.
        tensor = torch.stack([dataset[k][0] for k in range(start,stop)]).to(device)
        targets = [labels.index(predicted_label), labels.index(true_label)]
        probabilities, maps, resolution = class_maps(model, tensor, idx-start, targets)
        if probabilities.argmax() != targets[0] or abs(probabilities[targets[0]] - float(row["p:"+predicted_label])) > 0.001:
            raise ValueError(f"Grad-CAM forward pass differs from validation prediction for {row['path']}")
        titles = [f"True: {true_label}", f"Predicted: {predicted_label}", f"True: {true_label}"]
        for j, ax in enumerate(axes[i]):
            ax.imshow(image, interpolation="nearest")
            if j:
                cam = maps[j-1]
                ax.imshow(cam, cmap="inferno", vmin=0, vmax=1, alpha=0.72 * cam)
            ax.set_title(titles[j], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[i, 0].set_ylabel(f"#{ranking.index((true_label,predicted_label))+1}   ·   "
                              f"{counts[(true_label,predicted_label)]}/{truth_support[true_label]}", fontsize=9)
        examples.append(dict(path=row["path"], true_label=true_label, predicted_label=predicted_label,
                             predicted_probability=float(probabilities[targets[0]]),
                             true_probability=float(probabilities[targets[1]]), activation_grid=resolution,
                             rank=ranking.index((true_label,predicted_label))+1,
                             error_count=counts[(true_label,predicted_label)], true_label_support=truth_support[true_label]))
    fig.suptitle("EfficientNet-B0 · validation errors", fontsize=18)
    fig.text(.5,.94,"Input image                 Grad-CAM for prediction                 Grad-CAM for true class",
             ha="center", fontsize=10)
    fig.text(.5,.023,"ZooLake / Kyathanahally et al. (2021), CC BY 4.0. Resized images and normalized class maps.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0,.045,1,.915))
    fig.savefig(out / "gradcam.png", dpi=135)
    plt.close(fig)
    with (out / "cnn_confusions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("actual","predicted","count","actual_class_support"))
        writer.writerows((a,p,counts[(a,p)],truth_support[a]) for a,p in ranking)
    (out / "gradcam_cases.json").write_text(json.dumps(dict(partition="validation", mode="single",
        checkpoint_sha256=meta["model_sha256"], examples=examples), indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("run","split","data-root","out"):
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args()
    generate(args.run, args.split, args.data_root, args.out)
