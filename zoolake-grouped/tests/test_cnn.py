"""CPU smoke tests for the optional PyTorch backends."""

from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image


@unittest.skipUnless(find_spec("torch") is not None, "PyTorch is optional")
class CnnTests(unittest.TestCase):
    def test_scratch_checkpoint_reloads_with_both_view_modes(self):
        from zoolake.cnn import predict_cnn, train_cnn

        with TemporaryDirectory() as directory:
            root = Path(directory)
            train, validation = [], []
            for label, color in (("blue", (0, 30, 200)), ("red", (200, 30, 0))):
                for index in range(8):
                    path = f"{label}-{index}.png"
                    image = Image.new("RGB", (16, 16), color)
                    image.putpixel((index, 2), (255, 255, 255))
                    image.save(root / path)
                    members = train if index < 6 else validation
                    members.append({"path": path, "label": label})
            settings = {
                "seed": 464, "image_side": 16, "batch_size": 4, "workers": 0,
                "threads": 2, "epochs": 1, "class_weight": "balanced", "device": "cpu",
                "augment": "d4", "learning_rate": 0.001, "tta": True,
            }
            output = train_cnn("scratch", train, validation, root, ["blue", "red"], settings, root)
            reloaded = predict_cnn(root, validation, root, True, "cpu")
            self.assertEqual(set(output), {"single", "d4"})
            for view in output:
                self.assertEqual(output[view].shape, (4, 2))
                np.testing.assert_allclose(output[view].sum(axis=1), 1, atol=1e-6)
                np.testing.assert_array_equal(output[view], reloaded[view])
            log = json.loads((root / "training_log.json").read_text(encoding="utf-8"))
            self.assertEqual(log["selection_metric"], "single_view_validation_macro_f1")
            self.assertEqual(log["best_epoch"], 1)

    @unittest.skipUnless(find_spec("torchvision") is not None, "torchvision is optional")
    def test_efficientnet_has_a_zoolake_classification_head(self):
        import torch
        from zoolake.cnn import build_model

        classifier = build_model("efficientnet-b0", 35, pretrained=False)
        classifier.eval()
        with torch.inference_mode():
            self.assertEqual(classifier(torch.zeros(2, 3, 64, 64)).shape, (2, 35))
