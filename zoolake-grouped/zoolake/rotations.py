"""Evaluate a trusted saved model at eight orientations on fixed validation images."""

import argparse
from bisect import bisect_left
from collections import Counter
import csv
from contextlib import nullcontext
import json
import math
from pathlib import Path
import tempfile
import tomllib

import numpy as np
from PIL import Image, ImageOps

from .__main__ import (
    CLASSICAL, MODEL_FILES, _destination, data_root, partitions, read_json,
    read_split, sha256, validate_files, versions,
)
from .images import d4_view
from .metrics import summarize


ANGLES = range(0, 360, 45)
OUTER_BORDER = 1
BICUBIC_SUPPORT = 2
METRICS = {
    "macro_f1": "Macro-F1",
    "minority_macro_f1": "Minority macro-F1",
    "accuracy": "Accuracy",
    "agreement_vs_0": "Prediction agreement with 0°",
}


def _rotation_safe_size(width, height, side):
    longest = max(width, height)

    def resize_to(length):
        return max(1, round(length * width / longest)), max(1, round(length * height / longest))

    def exceeds_border(length):
        w, h = resize_to(length)
        # Integer centering can shift the ROI by half a pixel on either axis.
        extent_x = w + (side - w) % 2 + 2 * BICUBIC_SUPPORT
        extent_y = h + (side - h) % 2 + 2 * BICUBIC_SUPPORT
        return math.hypot(extent_x, extent_y) > side - 2 * OUTER_BORDER

    # The first unsafe size's index is the largest safe long-edge length.
    length = bisect_left(range(1, longest + 1), True, key=exceeds_border)
    if length == 0:
        raise ValueError("The model canvas is too small for the rotation border.")
    return resize_to(length)


def load_guarded_canvas(path, side):
    """Preserve the model's canonical ROI size when its full rectangle can rotate."""
    with Image.open(path) as raw:
        source = ImageOps.exif_transpose(raw).convert("RGB")
    canonical = source.copy() if max(source.size) > side else source
    canonical.thumbnail((side, side), resample=Image.Resampling.BILINEAR)
    safe_size = _rotation_safe_size(canonical.width, canonical.height, side)
    roi = canonical if safe_size == canonical.size else source.resize(safe_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (side, side))
    canvas.paste(roi, ((side - roi.width) // 2, (side - roi.height) // 2))
    return canvas, max(safe_size) / max(canonical.size)


def diagonal(canvas):
    """Interpolate just the 45-degree member of the diagonal rotation orbit."""
    return canvas.rotate(45, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))


def _predictor(model_path, settings, labels, device):
    if settings["model"] in CLASSICAL:
        import joblib
        from threadpoolctl import threadpool_limits

        from .classical import image_features

        if device not in (None, "auto", "cpu"):
            raise ValueError("Classical models require --device cpu or auto.")
        bundle = joblib.load(model_path)
        if bundle["labels"] != labels or bundle["image_side"] != settings["image_side"]:
            raise ValueError("The saved model's labels or image side differ from the run manifest.")
        model = bundle["model"]
        if list(model.classes_) != list(range(len(labels))):
            raise ValueError("The saved model's probability columns differ from the run manifest.")

        def predict(canvases):
            return model.predict_proba(np.stack([image_features(image) for image in canvases]))

        return predict, "cpu", threadpool_limits(limits=settings["threads"])

    import torch

    from .cnn import CHANNEL_MEAN, CHANNEL_STD, _predict, build_model, resolve_device

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint["labels"] != labels or any(
        checkpoint[name] != settings[name] for name in ("model", "image_side", "threads", "batch_size")
    ):
        raise ValueError("The saved checkpoint's model settings differ from the run manifest.")
    torch.set_num_threads(settings["threads"])
    hardware = resolve_device(device or settings["device"])
    model = build_model(settings["model"], len(labels))
    model.load_state_dict(checkpoint["weights"])
    model.to(hardware)

    def predict(canvases):
        rgb = torch.from_numpy(np.stack([np.asarray(image) for image in canvases]))
        batch = (rgb.permute(0, 3, 1, 2).float() / 255 - CHANNEL_MEAN) / CHANNEL_STD
        return _predict(model, [(batch, None)], hardware, tta=False)["single"]

    return predict, str(hardware), nullcontext()


def _orbit_scores(canvases, predict, tta):
    scores = [
        predict([d4_view(image, index) for image in canvases])
        for index in range(8 if tta else 4)
    ]
    mean = np.stack(scores).mean(axis=0) if tta else None
    # Exact quarter turns permute the eight D4 views, so one mean serves the orbit.
    return [
        {"single": scores[index], "d4": mean} if tta else {"single": scores[index]}
        for index in range(4)
    ]


def _sweep(root, rows, side, batch_size, predict, tta):
    chunks = {angle: [] for angle in ANGLES}
    scales = []
    for start in range(0, len(rows), batch_size):
        rois, batch_scales = zip(*(
            load_guarded_canvas(root / row["path"], side) for row in rows[start:start + batch_size]
        ))
        scales.extend(batch_scales)
        for offset, canvases in ((0, rois), (45, [diagonal(image) for image in rois])):
            for turn, scores in enumerate(_orbit_scores(canvases, predict, tta)):
                chunks[offset + 90 * turn].append(scores)
    probabilities = {
        angle: {
            view: np.concatenate([batch[view] for batch in batches])
            for view in batches[0]
        }
        for angle, batches in chunks.items()
    }
    return probabilities, scales


def _score_sweep(labels, rows, probability_by_angle, minority):
    truth = [row["label"] for row in rows]
    zero = {
        view: probabilities.argmax(axis=1)
        for view, probabilities in probability_by_angle[0].items()
    }
    scores_by_angle = []
    for angle, predictions in probability_by_angle.items():
        metrics = summarize(labels, truth, predictions, minority)
        for view, probabilities in predictions.items():
            scores_by_angle.append({
                "angle_degrees": angle, "view": view,
                **{name: metrics[view][name] for name in METRICS if name != "agreement_vs_0"},
                "agreement_vs_0": float(np.mean(probabilities.argmax(axis=1) == zero[view])),
            })
    summary = {}
    for view in zero:
        values = [row for row in scores_by_angle if row["view"] == view]
        summary[view] = {}
        for metric in METRICS:
            worst = min(row[metric] for row in values)
            summary[view][metric] = {
                "mean": float(np.mean([row[metric] for row in values])),
                "worst": worst,
                "worst_angles_degrees": [row["angle_degrees"] for row in values if row[metric] == worst],
            }
    return scores_by_angle, summary


def _save_plot(path, scores, summary, model):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharex=True)
    for axis, (metric, title) in zip(axes.flat, METRICS.items()):
        for view in summary:
            rows = [row for row in scores if row["view"] == view]
            axis.plot(
                [row["angle_degrees"] for row in rows], [row[metric] for row in rows],
                marker="o", markersize=4, label={"single": "One view", "d4": "Eight D4 views"}[view],
            )
        axis.set(title=title, xticks=list(ANGLES), ylim=(0, 1.02))
        axis.grid(alpha=0.22)
    axes[0, 0].legend(loc="best")
    figure.supxlabel("Counterclockwise rotation (degrees)")
    figure.suptitle(f"{model} · Validation rotation sweep with per-ROI margins")
    figure.savefig(path, dpi=170)


def evaluate_rotations(run_dir, split_path, root, output, *, tta=False, device=None):
    root = root.resolve()
    run = read_json(run_dir / "run.json")
    settings = run["config"]
    manifest, split_hash = read_split(split_path)
    if run["split_sha256"] != split_hash:
        raise ValueError("Split hash differs from the frozen run manifest.")
    if run["model_file"] != MODEL_FILES.get(settings["model"]):
        raise ValueError("Frozen run has an unexpected model filename.")
    model_path = run_dir / run["model_file"]
    if run["model_sha256"] != sha256(model_path):
        raise ValueError("Saved model hash differs from the frozen run manifest.")

    labels, _, validation, minority = partitions(manifest, settings)
    if not validation or len(validation) != run["validation_images"]:
        raise ValueError("Validation size differs from the frozen run manifest.")
    if len({row["path"] for row in validation}) != len(validation):
        raise ValueError("The split contains repeated validation image paths.")
    for row in validation:
        if not row.get("file_sha256"):
            raise ValueError("A validation image is missing its audited file hash. Regenerate the split.")
        if not (root / row["path"]).resolve().is_relative_to(root):
            raise ValueError(f"Validation image path escapes the dataset directory: {row['path']}")
    validate_files(root, validation)

    target = _destination(output)
    predict, hardware, limits = _predictor(model_path, settings, labels, device)
    side = settings["image_side"]
    with limits:
        probabilities, scales = _sweep(root, validation, side, settings["batch_size"], predict, tta)
    scores, summary = _score_sweep(labels, validation, probabilities, minority)
    unchanged = scales.count(1.0)
    payload = {
        "partition": "validation",
        "n_eval": len(validation),
        "label_support": dict(sorted(Counter(row["label"] for row in validation).items())),
        "minority_labels_from_full_training": minority,
        "angles_degrees": list(ANGLES),
        "direction": "counterclockwise",
        "device": hardware,
        "split_sha256": split_hash,
        "model_sha256": run["model_sha256"],
        "model_config": settings,
        "versions": versions(),
        "rotation_preprocessing": {
            "canonical_image_side": side,
            "outer_black_border_pixels": OUTER_BORDER,
            "reserved_bicubic_support_pixels_per_roi_edge": BICUBIC_SUPPORT,
            "unchanged_roi_count": unchanged,
            "shrunk_roi_count": len(scales) - unchanged,
            "retained_long_edge_ratio": {
                "minimum": min(scales), "mean": float(np.mean(scales)), "median": float(np.median(scales)),
            },
            "resizing": "bilinear",
            "diagonal_resampling": "bicubic",
            "fill_rgb": [0, 0, 0],
            "protocol": (
                "Decode the EXIF-oriented RGB ROI and use the model's canonical size when the centered "
                "rectangle and its bicubic support fit within the black border at any angle. For other ROIs, "
                "choose the largest safe long edge with the canonical aspect ratio rounded to pixels and "
                "resize from the source. The retained ratio is relative to the canonical long edge. All "
                "views of each ROI share one scale. Odd multiples of 45 degrees are exact quarter turns of "
                "the interpolated 45-degree canvas."
            ),
        },
        "summary_protocol": "Mean and minimum across all eight angles, including 0, for each view mode.",
        "per_angle": scores,
        "by_view": summary,
    }
    with tempfile.TemporaryDirectory(prefix=".rotations-", dir=target.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        (stage / "rotations.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with (stage / "rotations.csv").open("w", newline="", encoding="utf-8") as output_csv:
            writer = csv.DictWriter(output_csv, fieldnames=["angle_degrees", "view", *METRICS])
            writer.writeheader()
            writer.writerows(scores)
        _save_plot(stage / "rotations.png", scores, summary, settings["model"])
        stage.rename(target)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    parser.add_argument("--run", type=Path, required=True, help="Trusted local model run directory")
    parser.add_argument("--split", type=Path, help="The split manifest used in the saved run")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tta", action="store_true", help="Also score eight-view D4 probability averaging")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    try:
        with args.config.open("rb") as source:
            config = tomllib.load(source)
        root = data_root(args, config)
        split_path = args.split or Path(config["data"]["split_dir"]) / "splits.json"
        summary = evaluate_rotations(args.run, split_path, root, args.out, tta=args.tta, device=args.device)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")
    for view, metrics in summary.items():
        print(f"Validation {view}: mean macro-F1 {metrics['macro_f1']['mean']:.4f}, "
              f"worst {metrics['macro_f1']['worst']:.4f}")
    print(f"Rotation sweep: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
