"""Checks for matched validation study summaries."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from zoolake.study import write_study


class StudyTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def make_run(self, name, seed, aug, score, *, tta=None, split="a", epochs=3, partition="validation"):
        output = self.root / name
        output.mkdir()
        run = {
            "split_sha256": split * 64, "model_sha256": f"{seed:064x}",
            "training_images": 10, "validation_images": 8,
            "config": {
                "model": "lr", "fraction": 10, "augment": aug, "seed": seed,
                "tta": tta is not None, "epochs": epochs,
            },
        }
        scores = {
            name: {
                "macro_f1": value, "minority_macro_f1": value / 2, "accuracy": value + 0.1,
                "per_class": {"x": {"support": 2}, "y": {"support": 6}},
            }
            for name, value in (("single", score), ("d4", tta)) if value is not None
        }
        metrics = {
            "partition": partition, "labels": ["x", "y"], "n_eval": 8,
            "minority_labels_from_full_training": ["x"],
            "selected_training_counts": {"x": 2, "y": 8}, "views": scores,
        }
        (output / "run.json").write_text(json.dumps(run), encoding="utf-8")
        (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return output

    def test_report_preserves_seed_pairing_and_sample_sd(self):
        runs = [
            self.make_run("none-1", 1, "none", 0.40, tta=0.45),
            self.make_run("none-2", 2, "none", 0.60),
            self.make_run("d4-1", 1, "d4", 0.50, tta=0.48),
            self.make_run("d4-3", 3, "d4", 0.55),
        ]
        out = self.root / "study"
        write_study(runs, out)
        with (out / "grouped_scores.csv").open(encoding="utf-8", newline="") as stream:
            row = next(row for row in csv.DictReader(stream)
                       if row["train_augmentation"] == "none" and row["inference_view"] == "single")
        self.assertEqual(row["seeds"], "1,2")
        self.assertAlmostEqual(float(row["macro_f1_mean"]), 0.5)
        self.assertAlmostEqual(float(row["macro_f1_sd"]), 0.141421356)
        with (out / "paired_summary.csv").open(encoding="utf-8", newline="") as stream:
            row = next(row for row in csv.DictReader(stream)
                       if row["contrast"] == "d4_training" and row["value"] == "single")
        self.assertEqual((row["paired_seeds"], row["baseline_only_seeds"], row["treatment_only_seeds"]),
                         ("1", "2", "3"))
        self.assertEqual(row["pair_count"], "1")
        self.assertEqual(row["delta_macro_f1_sd"], "")
        self.assertAlmostEqual(float(row["delta_macro_f1_mean"]), 0.1)
        self.assertTrue((out / "learning_curves.pdf").read_bytes().startswith(b"%PDF"))
        self.assertIn("Matched D4 contrasts", (out / "summary.md").read_text(encoding="utf-8"))

    def test_rejects_mixed_splits_settings_or_test_results(self):
        reference = self.make_run("base", 1, "none", 0.4)
        options = (
            ("split hashes", {"split": "b"}),
            ("settings differ", {"epochs": 4}),
            ("only validation", {"partition": "test"}),
        )
        for index, (error, kwargs) in enumerate(options):
            with self.subTest(error=error):
                run = self.make_run(f"other-{index}", 2, "d4", 0.5, **kwargs)
                out = self.root / f"study-{index}"
                with self.assertRaisesRegex(ValueError, error):
                    write_study([reference, run], out)
                self.assertFalse(out.exists())
