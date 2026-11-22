"""Check local inference using a synthetic image and a tiny frozen model."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import joblib
import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier

from zoolake.__main__ import sha256
from zoolake.classical import image_features
from zoolake.images import load_canvas
from zoolake.predict import predict_image


class PredictTests(unittest.TestCase):
    def test_d4_emits_json_and_validates_frozen_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            specimen = root / "generated.png"
            image = Image.new("RGB", (12, 16))
            for x in range(2, 7):
                image.putpixel((x, 4), (100, 200, 200))
            image.save(specimen)
            labels = ["blue", "red"]
            features = image_features(load_canvas(specimen, 16))
            sandbox_model = DummyClassifier(strategy="prior").fit(np.tile(features, (3, 1)), [0, 1, 1])
            model_path = root / "model.joblib"
            joblib.dump({"model": sandbox_model, "labels": labels, "image_side": 16}, model_path)
            (root / "run.json").write_text(json.dumps({
                "config": {"model": "lr", "image_side": 16, "threads": 1},
                "model_file": model_path.name,
                "model_sha256": sha256(model_path),
            }), encoding="utf-8")
            metrics_path = root / "metrics.json"
            metrics = {"partition": "validation", "labels": labels}
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

            process = subprocess.run(
                [sys.executable, "-m", "zoolake.predict", "--run", str(root), "--image", str(specimen),
                 "--top-k", "2", "--view", "d4"],
                capture_output=True, text=True, check=True, timeout=30,
            )
            result = json.loads(process.stdout)
            self.assertEqual(result["view"], "d4")
            self.assertEqual(result["device"], "cpu")
            self.assertEqual([item["label"] for item in result["predictions"]], ["red", "blue"])
            self.assertAlmostEqual(result["predictions"][0]["probability"], 2 / 3)

            metrics_path.write_text(json.dumps({**metrics, "labels": labels[::-1]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "labels or image side differ"):
                predict_image(root, specimen)
            model_path.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "model hash differs"):
                predict_image(root, specimen)
