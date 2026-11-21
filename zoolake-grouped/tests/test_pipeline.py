"""Run the CLI end to end on a tiny generated image dataset."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from zoolake.images import d4_view, d4_views


class PipelineTests(unittest.TestCase):
    def test_validation_and_frozen_test_are_separate(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            data = workspace / "images"
            for label, color in (("blue", (0, 30, 240)), ("red", (240, 30, 0))):
                folder = data / label
                folder.mkdir(parents=True)
                for index in range(20):
                    image = Image.new("RGB", (16, 16), color)
                    image.putpixel((index % 16, index // 16), (255, 255, 255))
                    image.save(folder / f"{index}.png")

            audit_dir = workspace / "audit"
            split_dir = workspace / "split"
            run_dir = workspace / "run"
            test_dir = workspace / "test"
            env = {**os.environ, "MPLBACKEND": "Agg"}

            def cli(*args, success=True):
                result = subprocess.run(
                    [sys.executable, "-m", "zoolake", *map(str, args)],
                    capture_output=True, text=True, env=env, timeout=60,
                )
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result

            cli("audit", "--data-root", data, "--out", audit_dir)
            cli("split", "--audit", audit_dir / "audit.json", "--out", split_dir)
            split = split_dir / "splits.json"
            cli("train", "--data-root", data, "--split", split, "--out", run_dir,
                "--augment", "d4", "--tta", "--image-side", 16)
            validation = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(validation["partition"], "validation")
            self.assertEqual(set(validation["views"]), {"single", "d4"})
            self.assertEqual(validation["views"]["d4"]["macro_f1"], 1.0)
            cli("evaluate-test", "--data-root", data, "--split", split, "--run", run_dir, "--out", test_dir,
                "--view", "d4")
            test = json.loads((test_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(test["partition"], "test")
            self.assertEqual(set(test["views"]), {"d4"})
            self.assertEqual(test["views"]["d4"]["macro_f1"], 1.0)

            with split.open("a", encoding="utf-8") as destination:
                destination.write("\n")
            invalid = cli("evaluate-test", "--data-root", data, "--split", split,
                          "--run", run_dir, "--out", workspace / "invalid", success=False)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("Split hash differs", invalid.stderr)

    def test_d4_views_form_same_orbit(self):
        image = Image.new("L", (3, 3))
        image.putdata(range(9))
        views = {view.tobytes() for view in d4_views(image)}
        self.assertEqual(len(views), 8)
        for index in range(8):
            self.assertEqual({view.tobytes() for view in d4_views(d4_view(image, index))}, views)
