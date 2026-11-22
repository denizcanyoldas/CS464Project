"""Validate final selection and delegation without fitting or opening test images."""

from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from zoolake import finalize


class FinalizeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.split = self.root / "split.json"
        self.selection = self.root / "selection.json"
        self.labels = ["blue", "red"]
        self.counts = dict.fromkeys(self.labels, 1)
        rows = [
            {"label": label, "split": partition, "path": f"{partition}/{label}.png"}
            for label in self.labels for partition in ("train", "validation", "test")
        ]
        self.split.write_text(json.dumps({
            "labels": self.labels, "records": rows,
            "subsets": {"464": {"100": ["train/blue.png", "train/red.png"]}},
        }), encoding="utf-8")

    def make_run(self, name, single, d4=None, *, model="lr", device="cpu", seed=464):
        path = self.root / "runs" / name
        path.mkdir(parents=True)
        model_file = "model.joblib" if model == "lr" else "checkpoint.pt"
        (path / model_file).write_bytes(b"synthetic model fixture")
        run = {
            "config": {"model": model, "seed": seed, "fraction": 100, "tta": d4 is not None, "device": "auto"},
            "split_sha256": sha256(self.split.read_bytes()).hexdigest(),
            "model_file": model_file,
            "model_sha256": sha256((path / model_file).read_bytes()).hexdigest(),
            "training_images": 2,
            "validation_images": 2,
        }
        if model != "lr":
            (path / "training_log.json").write_text(json.dumps({"device": device}), encoding="utf-8")
        scores = {"single": single} if d4 is None else {"single": single, "d4": d4}
        metrics = {
            "partition": "validation", "labels": self.labels,
            "n_eval": 2, "selected_training_counts": self.counts,
            "views": {
                view: {"macro_f1": score, "per_class": {
                    label: {"support": count} for label, count in self.counts.items()
                }}
                for view, score in scores.items()
            },
        }
        for filename, data in (("run.json", run), ("metrics.json", metrics)):
            (path / filename).write_text(json.dumps(data), encoding="utf-8")
        return path

    def freeze(self, *runs):
        with redirect_stdout(io.StringIO()):
            finalize.freeze(runs, self.split, self.selection)
        return json.loads(self.selection.read_bytes())

    def test_ties_prefer_single_then_alphabetical_run(self):
        first = self.make_run("alpha", 0.4, 0.6)
        second = self.make_run("beta", 0.6, 0.6)
        third = self.make_run("gamma", 0.6)
        frozen = self.freeze(third, second, first)
        self.assertEqual(frozen["chosen"]["run"], "runs/beta")
        self.assertEqual(frozen["chosen"]["evaluation_view"], "single")

    def test_evaluation_verifies_and_delegates_only_chosen_view_and_device(self):
        cnn = self.make_run("scratch", 0.4, 0.7, model="scratch", device="mps")
        lr = self.make_run("lr", 0.6)
        self.freeze(cnn, lr)
        output = self.root / "final"

        def fake_evaluate(args, config, split_path):
            self.assertEqual(args.run, cnn.resolve())
            self.assertEqual((args.view, args.device), ("d4", "mps"))
            self.assertEqual(split_path, self.split)
            args.out.mkdir()

        with patch.object(finalize.pipeline, "evaluate", side_effect=fake_evaluate) as delegate:
            with redirect_stdout(io.StringIO()):
                finalize.evaluate(self.selection, self.split, self.root, output)
            delegate.assert_called_once()
        self.assertEqual((output / "selection.json").read_bytes(), self.selection.read_bytes())

    def test_changed_saved_metrics_refuse_evaluation(self):
        run = self.make_run("lr", 0.5)
        self.freeze(run)
        with (run / "metrics.json").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with patch.object(finalize.pipeline, "evaluate") as delegate:
            with self.assertRaisesRegex(ValueError, "candidate hashes or selection rule"):
                finalize.evaluate(self.selection, self.split, self.root, self.root / "final")
            delegate.assert_not_called()

    def test_nonselection_seed_is_rejected(self):
        run = self.make_run("other-seed", 0.5, seed=465)
        with self.assertRaisesRegex(ValueError, "must use seed 464"):
            self.freeze(run)
