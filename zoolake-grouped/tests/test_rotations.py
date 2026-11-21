"""Smoke checks for the controlled rotation protocol and validation-only inputs."""

from argparse import Namespace
import csv
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest

import numpy as np
from PIL import Image

from zoolake.__main__ import fit, sha256
from zoolake.images import d4_view, load_canvas
from zoolake.rotations import ANGLES, _orbit_scores, _predictor, diagonal, evaluate_rotations, load_guarded_canvas


class RotationTests(unittest.TestCase):
    def test_diagonal_keeps_the_outer_border_black(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "roi.png"
            for side in (16, 17, 128, 129):
                Image.new("RGB", (side, side), "white").save(path)
                guarded, retained = load_guarded_canvas(path, side)
                self.assertLess(retained, 1)
                rotated = np.asarray(diagonal(guarded))
                self.assertEqual(rotated.shape, (side, side, 3))
                self.assertEqual(rotated[0].max(), 0)
                self.assertEqual(rotated[-1].max(), 0)
                self.assertEqual(rotated[:, 0].max(), 0)
                self.assertEqual(rotated[:, -1].max(), 0)
                self.assertEqual(rotated[side // 2, side // 2].min(), 255)

    def test_safe_roi_keeps_canonical_pixels_and_exif_orientation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "portrait.jpg"
            image = Image.new("RGB", (30, 41), "red")
            image.putpixel((2, 3), (255, 255, 255))
            exif = image.getexif()
            exif[274] = 6
            image.save(path, exif=exif)
            guarded, retained = load_guarded_canvas(path, 64)
            self.assertEqual(retained, 1)
            self.assertEqual(guarded.tobytes(), load_canvas(path, 64).tobytes())

    def test_thin_roi_retains_more_scale_than_square_roi(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "roi.png"
            Image.new("RGB", (128, 34), "white").save(path)
            guarded, retained = load_guarded_canvas(path, 128)
            self.assertGreater(retained, 0.9)
            self.assertLess(retained, 1)
            self.assertEqual(np.asarray(diagonal(guarded))[[0, -1]].max(), 0)

    def test_orbit_averages_all_eight_probabilities(self):
        image = Image.new("RGB", (3, 3))
        for index in range(9):
            image.putpixel((index % 3, index // 3), (index * 20, 0, 0))

        def predict(images):
            first = np.array([view.getpixel((0, 1))[0] / 255 for view in images])
            return np.stack((first, 1 - first), axis=1)

        scores = _orbit_scores([image], predict, tta=True)
        for turn, mode in enumerate(scores):
            quarter_turned = d4_view(image, turn)
            explicit = np.stack([predict([d4_view(quarter_turned, k)]) for k in range(8)])
            np.testing.assert_allclose(mode["d4"], explicit.mean(axis=0))
            np.testing.assert_allclose(mode["single"], explicit[0])

    @unittest.skipUnless(find_spec("torch") is not None, "PyTorch is optional")
    def test_frozen_cnn_uses_the_same_tensor_preprocessing(self):
        import torch

        from zoolake.cnn import build_model, predict_cnn

        labels = ["blue", "red"]
        settings = {"model": "scratch", "image_side": 16, "threads": 2, "batch_size": 4, "device": "cpu"}
        model = build_model("scratch", len(labels))
        with TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint_path = run / "checkpoint.pt"
            torch.save({**settings, "labels": labels, "weights": model.state_dict()}, checkpoint_path)
            roi = run / "roi.png"
            Image.new("RGB", (16, 16), "red").save(roi)
            canvas = diagonal(load_guarded_canvas(roi, 16)[0])
            canvas.save(run / "rotated.png")
            regular = predict_cnn(run, [{"path": "rotated.png", "label": "red"}], run, True, "cpu")
            predict, device, limits = _predictor(checkpoint_path, settings, labels, None)
            self.assertEqual(device, "cpu")
            with limits:
                rotated = _orbit_scores([canvas], predict, tta=True)[0]
            for mode in regular:
                np.testing.assert_allclose(rotated[mode], regular[mode], atol=1e-7)

    def test_frozen_validation_sweep_and_audited_hashes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            labels = ["blue", "red"]
            records = []
            for label, color in zip(labels, ((0, 30, 240), (240, 30, 0))):
                for index in range(10):
                    path = root / f"{label}-{index}.png"
                    image = Image.new("RGB", (16, 16), color)
                    image.putpixel((index, 0), (255, 255, 255))
                    image.save(path)
                    records.append({
                        "path": path.name, "label": label, "file_sha256": sha256(path),
                        "split": "train" if index < 8 else "validation",
                    })
                records.append({"path": f"missing-held-out-{label}.png", "label": label, "split": "test"})
            train_paths = [row["path"] for row in records if row["split"] == "train"]
            manifest = {"labels": labels, "records": records, "subsets": {"464": {"100": train_paths}}}
            split_path = root / "splits.json"
            split_path.write_text(json.dumps(manifest), encoding="utf-8")
            run = root / "run"
            with Path("configs/default.toml").open("rb") as source:
                config = tomllib.load(source)
            fit(Namespace(data_root=root, image_side=16, out=run), config, split_path)

            output = root / "rotation-eval"
            summary = evaluate_rotations(run, split_path, root, output, tta=True)
            payload = json.loads((output / "rotations.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["partition"], "validation")
            self.assertEqual(payload["n_eval"], 4)
            self.assertEqual(payload["by_view"], summary)
            self.assertEqual(set(summary), {"single", "d4"})
            geometry = payload["rotation_preprocessing"]
            self.assertEqual(geometry["unchanged_roi_count"], 0)
            self.assertEqual(geometry["shrunk_roi_count"], 4)
            with (output / "rotations.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2 * len(ANGLES))
            for view in summary:
                mode_rows = [row for row in rows if row["view"] == view]
                self.assertEqual([int(row["angle_degrees"]) for row in mode_rows], list(ANGLES))
                self.assertEqual(float(mode_rows[0]["agreement_vs_0"]), 1.0)
            self.assertTrue((output / "rotations.png").is_file())

            (root / "red-8.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Image bytes changed"):
                evaluate_rotations(run, split_path, root, root / "invalid-image")
            split_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Split hash differs"):
                evaluate_rotations(run, split_path, root, root / "invalid-split")
