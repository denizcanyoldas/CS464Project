"""Predict the class of a local RGB image with the full-data baseline."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from features import describe, prepare


def predict(model, image):
    vector = (describe(prepare(image)).astype(np.float64) - model["mean"]) / model["scale"]
    scores = model["coef"] @ vector + model["intercept"]
    probabilities = np.exp(scores - scores.max())
    return model["classes"], probabilities / probabilities.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "results/baseline.npz")
    args = parser.parse_args()
    with np.load(args.model, allow_pickle=False) as model, Image.open(args.image) as image:
        classes, probabilities = predict(model, image)
        for idx in np.argsort(-probabilities)[:3]:
            print(f"{classes[idx]:24s} {probabilities[idx]:.1%}")
