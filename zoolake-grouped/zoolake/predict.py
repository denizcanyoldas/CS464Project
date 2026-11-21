"""Print top-k predictions for one local image from a trusted ZooLake training run."""

import argparse
import json
from pathlib import Path

import numpy as np

from .__main__ import MODEL_FILES, read_json, sha256
from .images import d4_views, load_canvas
from .rotations import _predictor


def predict_image(run_dir, image_path, *, top_k=5, view="single", device="cpu"):
    if top_k < 1:
        raise ValueError("--top-k must be positive.")
    if view not in ("single", "d4"):
        raise ValueError("--view must be single or d4.")
    run = read_json(run_dir / "run.json")
    settings = run["config"]
    model_name = settings["model"]
    if model_name not in MODEL_FILES or run["model_file"] != MODEL_FILES[model_name]:
        raise ValueError("Frozen run has an unexpected model name or filename.")
    model_path = run_dir / run["model_file"]
    if sha256(model_path) != run["model_sha256"]:
        raise ValueError("Saved model hash differs from the frozen run manifest.")

    metrics = read_json(run_dir / "metrics.json")
    labels = metrics["labels"]
    if (
        metrics["partition"] != "validation" or not isinstance(labels, list) or not labels
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Expected the frozen run's validation metrics and unique ordered labels.")

    predict, hardware, limits = _predictor(model_path, settings, labels, device)
    canvas = load_canvas(image_path, settings["image_side"])
    canvases = list(d4_views(canvas)) if view == "d4" else [canvas]
    with limits:
        probabilities = np.asarray(predict(canvases))
    if (
        probabilities.shape != (len(canvases), len(labels))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("Saved model returned invalid class probabilities.")
    pooled = probabilities.mean(axis=0)
    order = np.argsort(-pooled, kind="stable")[:top_k]
    return {
        "image": str(image_path),
        "model": model_name,
        "model_sha256": run["model_sha256"],
        "view": view,
        "device": hardware,
        "predictions": [
            {"label": labels[index], "probability": float(pooled[index])} for index in order
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Trusted local run containing the saved model")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--view", choices=("single", "d4"), default="single")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        result = predict_image(args.run, args.image, top_k=args.top_k, view=args.view, device=args.device)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
