import csv
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from zoolake.audit import audit_dataset


class AuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "dataset"
        self.output = Path(temp.name) / "audit"

    def image(self, relative_path, color=(32, 0, 0)):
        image_path = self.root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 8), color).save(image_path)
        return image_path

    def test_missing_path_writes_diagnostics(self):
        report = audit_dataset(self.root, self.output)
        self.assertEqual(report["status"], "missing")
        self.assertIn("--data-root", report["error"])
        self.assertEqual(json.loads((self.output / "audit.json").read_text()), report)
        self.assertIn("unavailable", (self.output / "audit.md").read_text())

    def test_class_folders_keep_corrupt_label_and_duplicate_diagnostics(self):
        path = self.image("chaoborus/training_data/source.png")
        same_pixels = self.root / "chaoborus/training_data/source.bmp"
        with Image.open(path) as source:
            source.save(same_pixels)
        (self.root / "dinobryon").mkdir()
        shutil.copyfile(path, self.root / "dinobryon/same_bytes.png")
        (self.root / "dirt").mkdir()
        (self.root / "dirt/corrupt.jpg").write_bytes(b"No image")
        report = audit_dataset(self.root, self.output)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["labels"], ["chaoborus", "dinobryon", "dirt"])
        self.assertEqual(report["label_check"]["labels_without_readable_images"], ["dirt"])
        self.assertEqual(report["image_counts"], {"candidates": 4, "readable": 3, "invalid": 1})
        self.assertEqual(report["dimensions"]["width"]["median"], 12)
        self.assertEqual(report["duplicates"]["duplicate_file_copies"], 1)
        self.assertEqual(report["duplicates"]["duplicate_decoded_copies"], 2)
        self.assertEqual(len(report["duplicates"]["conflicting_label_groups"]), 1)

    def test_roi_groups_are_shared_across_provider_partitions(self):
        prefix = "SPC-EAWAG-0P5X-1526948087602056-1089941170938"
        self.image(f"train/chaoborus/{prefix}-008779-020-3268-256-108-112.jpeg")
        self.image(f"val/dinobryon/{prefix}-008779-021-3268-256-108-112.jpeg", (0, 64, 0))
        self.image("test/chaoborus/renamed.jpeg", (0, 0, 64))
        report = audit_dataset(self.root, self.output, group_by="day")
        self.assertEqual(report["layout"], "provider-folders")
        self.assertEqual(report["provider_splits"], {"train": 1, "validation": 1, "test": 1})
        active_groups = report["group_metadata"]["active"]
        self.assertEqual(active_groups["covered_images"], 2)
        self.assertEqual(active_groups["cross_provider_split_groups"], ["day:2018-05-22"])
        self.assertEqual(report["group_metadata"]["capture_frame"]["distinct_groups"], 1)

    def test_explicit_metadata_cannot_silently_omit_images(self):
        self.image("roi_1.png")
        self.image("nested/roi_2.png")
        metadata = self.root.parent / "metadata.csv"
        with metadata.open("w", newline="") as stream:
            rows = csv.writer(stream)
            rows.writerow(["path", "label", "split", "group"])
            rows.writerow(["roi_1.png", "dinobryon", "train", "deployment-b"])
        report = audit_dataset(self.root, self.output, metadata)
        self.assertEqual(report["status"], "invalid")
        self.assertIn("omits a discovered image", report["error"])
        with metadata.open("a", newline="") as stream:
            csv.writer(stream).writerow(["nested/roi_2.png", "dirt", "train", "deployment-b"])
        report = audit_dataset(self.root, self.output, metadata)
        self.assertEqual(report["status"], "ok")
        self.assertEqual([record["group"] for record in report["records"]], ["csv:deployment-b"] * 2)

    def test_three_minute_groups_are_utc_aligned(self):
        base = int(datetime(2019, 9, 25, 10, tzinfo=timezone.utc).timestamp())
        for seconds in (0, 179, 180):
            micros = (base + seconds) * 1_000_000
            self.image(f"chaoborus/SPC-EAWAG-0P5X-{micros}-1089941170938-008779-020-3268-256-108-112.jpeg")

        report = audit_dataset(self.root, self.output, group_by="three_minutes")

        self.assertEqual([record["group"] for record in report["records"]], [
            "three_minutes:2019-09-25T10:00Z", "three_minutes:2019-09-25T10:00Z",
            "three_minutes:2019-09-25T10:03Z",
        ])
        self.assertEqual(report["group_metadata"]["acquisition_hour"]["labels_with_fewer_than_three_groups"], {"chaoborus": 1})

    def test_eawag_manifest_reconciles_absolute_paths_and_exposes_split_conflicts(self):
        images = "data/zooplankton_0p5x"
        paths = ["dirt/training_data/a.png", "rotifers/training_data/b.png", "dirt/training_data/c.png"]
        for path in paths:
            self.image(f"{images}/{path}")
        self.image("data/Fig_Summary.png")
        provider = self.root / "data/zoolake_train_test_val_separated"
        provider.mkdir()
        prefix = "/nonportable/source/zooplankton_0p5x/"
        for split, indices in [("train", [0, 0]), ("val", [1]), ("test", [1])]:
            (provider / f"{split}_filenames.txt").write_text("\n".join(prefix + paths[i] for i in indices))
        np.save(provider / "classes_ERIC.npy", np.array(["dirt", "rotifers"]), allow_pickle=False)

        report = audit_dataset(self.root, self.output)

        self.assertEqual(report["layout"], "eawag-release")
        self.assertEqual(report["image_counts"]["readable"], 3)
        self.assertTrue(report["provider"]["matches_local_class_order"])
        self.assertEqual(report["provider"]["manifest_counts"], {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(report["provider"]["unassigned_paths"], [paths[2]])
        self.assertEqual(report["provider"]["repeated_paths"][paths[0]], {"train": 2})
        row = next(record for record in report["records"] if record["path"].endswith("/b.png"))
        self.assertEqual(row["provider_memberships"], ["test", "validation"])
        self.assertIsNone(row["split"])
