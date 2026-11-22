from pathlib import Path
from importlib.util import find_spec
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from zoolake.classical import predict_classical, train_classical
from zoolake.images import d4_views, load_canvas
from zoolake.morphology import morphology_features
from zoolake.rotations import _predictor
from zoolake.splits import _restore_folds


class MorphologyTests(unittest.TestCase):
    def test_features_are_finite_and_invariant_across_d4(self):
        image = Image.new("RGB", (17, 17))
        pixels = np.array(image)
        pixels[:6, :9] = np.arange(54).reshape(6, 9, 1) * 4
        pixels[-1, 0] = 255
        for tile in (image, Image.fromarray(pixels)):
            expected = morphology_features(tile)
            self.assertEqual(expected.shape, (31,))
            self.assertTrue(np.isfinite(expected).all())
            for view in d4_views(tile):
                np.testing.assert_allclose(morphology_features(view), expected, atol=1e-6)

    def test_model_reloads_same_features_and_probabilities(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for label, color in (("a", (250, 20, 10)), ("b", (10, 20, 250))):
                for i in range(3):
                    path = f"{label}{i}.png"
                    image = Image.new("RGB", (5 + i, 8), color)
                    image.save(root / path)
                    rows.append({"path": path, "label": label})
            train = [row for row in rows if "2" not in row["path"]]
            val = [row for row in rows if "2" in row["path"]]
            config = dict(seed=464, image_side=16, augment="none", class_weight="balanced",
                          regularization_c=1.0, tta=True, threads=1)
            models = ["lr-morph", "svm-morph"]
            if find_spec("xgboost"):
                models.append("xgb-morph")
            for name in models:
                with self.subTest(model=name):
                    before = train_classical(name, train, val, root, ["a", "b"], config, root)
                    after = predict_classical(root, val, root, True, 1)
                    for mode in ("single", "d4"):
                        np.testing.assert_array_equal(before[mode], after[mode])
                    predict, _, limits = _predictor(root / "model.joblib", dict(config, model=name),
                                                    ["a", "b"], "cpu")
                    with limits:
                        single = predict([load_canvas(root / r["path"], 16) for r in val])
                    np.testing.assert_allclose(before["single"], single)

    def test_fold_map_rejects_crossing_component(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            path.write_text("group,split\na,train\nb,validation\nc,test\nd,test\n", encoding="utf-8")
            rows = [dict(group=g, path=g, label="p") for g in "abcd"]
            self.assertEqual(_restore_folds(path, [[r] for r in rows], ["p"])["a"], "train")
            with self.assertRaisesRegex(ValueError, "linked component"):
                _restore_folds(path, [rows[:2], rows[2:]], ["p"])
