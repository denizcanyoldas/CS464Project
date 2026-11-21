from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from zoolake.splits import PARTITIONS, create_splits


def row(label, index, *, digest=None, group=None, day=None, split=None):
    path = f"{label}/{index}.png"
    return {
        "path": path, "label": label, "sha256": digest or path,
        "group": group, "acquisition_day": day, "split": split,
    }


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.audit = self.directory / "audit.json"

    def write_audit(self, records):
        self.audit.write_text(json.dumps({
            "status": "ok",
            "root": str(self.directory),
            "labels": ["common", "rare"],
            "records": records,
        }), encoding="utf-8")

    def split(self, strategy="stratified", **kwargs):
        return create_splits(
            self.audit, self.directory / "out", strategy, 464, [464, 465], **kwargs
        )

    def test_coverage_nested_subsets_and_determinism(self):
        self.write_audit([
            *(row("common", i) for i in range(97)),
            *(row("rare", i) for i in range(3)),
        ])
        first = self.split()
        second = self.split()
        self.assertEqual(first["records"], second["records"])
        self.assertEqual(first["subsets"], second["subsets"])
        folds = Counter(item["split"] for item in first["records"])
        self.assertEqual(folds, {"train": 70, "validation": 15, "test": 15})
        self.assertEqual(first["components_per_class"]["rare"], 3)
        for partition in PARTITIONS:
            self.assertEqual(first["partition_class_counts"][partition]["rare"], 1)
        training = {row["path"] for row in first["records"] if row["split"] == "train"}
        for subsets in first["subsets"].values():
            previous = set()
            for percent, target in (("10", 7), ("25", 18), ("50", 35), ("100", 70)):
                paths = set(subsets[percent])
                self.assertTrue(previous <= paths <= training)
                self.assertEqual(len(paths), target)
                self.assertEqual(sum(path.startswith("rare/") for path in paths), 1)
                previous = paths
        self.assertTrue((self.directory / "out" / "split_summary.md").is_file())

    def test_transitive_components_and_conflicting_hash_quarantine(self):
        self.write_audit([
            row("common", "first", digest="same", group="a"),
            row("common", "second", digest="same", group="b"),
            row("rare", "linked", group="b"),
            row("common", "conflicted", digest="conflict"),
            row("rare", "conflicted", digest="conflict"),
            *(row(label, i) for label in ("common", "rare") for i in range(15)),
        ])
        split = self.split()
        by_path = {row["path"]: row for row in split["records"]}
        group = [by_path[path] for path in ("common/first.png", "common/second.png", "rare/linked.png")]
        self.assertEqual(len({row["split"] for row in group}), 1)
        self.assertEqual(len({row["component"] for row in group}), 1)
        self.assertEqual(split["quarantined_images"], 2)
        self.assertEqual(
            {row["label"] for row in split["quarantined_conflicting_hashes"]["conflict"]},
            {"common", "rare"},
        )
        self.assertNotIn("rare/conflicted.png", by_path)
        self.assertNotIn("common/conflicted.png", by_path)

    def test_provider_requires_component_isolation(self):
        clean_rows = [
            row(label, i, split=partition)
            for label in ("common", "rare")
            for i, partition in enumerate(PARTITIONS)
        ]
        self.write_audit(clean_rows)
        self.assertEqual(
            {row["path"]: row["split"] for row in self.split(strategy="provider")["records"]},
            {row["path"]: row["split"] for row in clean_rows},
        )
        clean_rows[-1]["group"] = clean_rows[0]["group"] = "shared"
        self.write_audit(clean_rows)
        with self.assertRaisesRegex(ValueError, "1 cross-fold components"):
            self.split(strategy="provider")

    def test_day_grouping_detects_minority_infeasibility(self):
        self.write_audit([
            *(row("common", i, day=str(i)) for i in range(10)),
            *(row("rare", i, day=str(i % 2)) for i in range(10)),
        ])
        with self.assertRaisesRegex(ValueError, r"rare \(2\)"):
            self.split(group_by="acquisition_day")
        self.assertEqual(self.split()["components_per_class"]["rare"], 10)

    def test_expanded_budget_is_documented_if_coverage_needs_it(self):
        self.write_audit([row(label, i) for label in ("common", "rare") for i in range(3)])
        result = self.split()
        ten_percent = result["subset_statistics"]["464"]["10"]
        self.assertEqual(ten_percent["requested_count"], 0)
        self.assertEqual(ten_percent["actual_count"], 2)


if __name__ == "__main__":
    unittest.main()
