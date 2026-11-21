"""Inventory local ZooLake images before choosing a data split."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import struct
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import median
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": "train", "training": "train", "val": "validation",
    "valid": "validation", "validation": "validation", "test": "test",
}
SPLITS = ("train", "validation", "test")
# Class names published by the OcéanIA ZooLake data card. Local IDs are
# alphabetical so folder layout, CSV row order and provider ID order do not matter.
PUBLISHED_LABELS = tuple(
    "aphanizomenon asplanchna asterionella bosmina brachionus ceratium chaoborus "
    "conochilus copepod_skins cyclops daphnia daphnia_skins diaphanosoma diatom_chain "
    "dinobryon dirt eudiaptomus filament fish fragilaria hydra kellicottia "
    "keratella_cochlearis keratella_quadrata leptodora maybe_cyano nauplius "
    "paradileptus polyarthra rotifers synchaeta trichocerca unknown unknown_plankton "
    "uroglena".split()
)
DATA_URL = (
    "https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7/"
    "resource/1cc785fa-36c2-447d-bb11-92ce1d1f3f2d/download/data.zip"
)
ROOT_HELP = (
    f"Download or copy ZooLake from {DATA_URL}, extract it, and set --data-root "
    "to the extracted archive directory or the directory containing the class folders. For a custom layout, supply "
    "--metadata with a CSV containing path,label[,split,group]."
)
# See Figure 3b, https://doi.org/10.1038/s41597-025-04988-9. Restrict
# inference to the documented camera-format strings to avoid false groups.
ROI_NAME = re.compile(
    r"(?P<camera>SPC-EAWAG-(?:0P5X|5P0X))-(?P<unix_us>[1-9]\d{15})-"
    r"(?P<frame_us>\d+)(?:-\d+){6}",
    re.IGNORECASE,
)
METADATA_EXTENSIONS = {".csv", ".tsv", ".parquet", ".json", ".jsonl", ".xlsx"}
EAWAG_IMAGE_DIR = "zooplankton_0p5x"
EAWAG_SPLIT_DIR = "zoolake_train_test_val_separated"
EAWAG_SPLIT_FILES = {"train": "train_filenames.txt", "validation": "val_filenames.txt", "test": "test_filenames.txt"}
TIME_BINS = {"acquisition_hour": 3600, "acquisition_10min": 600,
             "acquisition_5min": 300, "acquisition_3min": 180}
GROUP_FIELDS = {"frame": "capture_frame", "day": "acquisition_day", "hour": "acquisition_hour",
                "ten_minutes": "acquisition_10min", "five_minutes": "acquisition_5min",
                "three_minutes": "acquisition_3min"}
GROUP_TITLES = {"active": "Used in split input", "capture_frame": "Same camera frame",
                "acquisition_day": "Same UTC day", "acquisition_hour": "Same UTC hour",
                "acquisition_10min": "Same UTC 10-minute bin", "acquisition_5min": "Same UTC 5-minute bin",
                "acquisition_3min": "Same UTC 3-minute bin"}


def _raise_walk_error(error: OSError) -> None:
    raise error


def _inventory(root: Path) -> tuple[list[Path], list[str]]:
    images = []
    sidecars = []
    for directory, folders, files in os.walk(root, onerror=_raise_walk_error):
        folders[:] = [name for name in sorted(folders) if not name.startswith(".") and name != "__MACOSX"]
        for name in sorted(files):
            if name.startswith("."):
                continue
            path = Path(directory, name).relative_to(root)
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                images.append(path)
            elif suffix in METADATA_EXTENSIONS:
                sidecars.append(path.as_posix())
    return sorted(images), sorted(sidecars)


def _csv_records(metadata: Path, found: list[Path]) -> list[dict]:
    records = {}
    with metadata.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not reader.fieldnames or not {"path", "label"}.issubset(reader.fieldnames):
            raise ValueError("The metadata CSV requires headers path,label and optional split,group.")
        for row in reader:
            if None in row or row["path"] is None or row["label"] is None:
                raise ValueError(f"Malformed metadata CSV row {reader.line_num}.")
            path = Path(row["path"].strip())
            label = row["label"].strip()
            split = (row.get("split") or "").strip().lower()
            if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Metadata row {reader.line_num} needs a supported root-relative image path.")
            if path in records or not label or (split and split not in SPLIT_ALIASES):
                raise ValueError(f"Invalid or repeated path, label, or split in metadata row {reader.line_num}.")
            records[path] = {
                "path": path.as_posix(), "label": label, "split": SPLIT_ALIASES.get(split),
                "group": (row.get("group") or "").strip() or None,
            }
    found_set = set(found)
    for discrepancy, message in [(records.keys() - found_set, "references an absent image"),
                                 (found_set - records.keys(), "omits a discovered image")]:
        if discrepancy:
            raise ValueError(f"Metadata {message}: {min(discrepancy)} ({len(discrepancy)} paths).")
    return [records[path] for path in found]


def _folder_records(found: list[Path]) -> tuple[list[dict], str]:
    records = []
    for path in found:
        folders = path.parts[:-1]
        split = SPLIT_ALIASES.get(folders[0].lower()) if folders else None
        if split:
            folders = folders[1:]
        if len(folders) == 2 and folders[-1] == "training_data":
            folders = folders[:-1]
        if len(folders) != 1:
            raise ValueError(
                f"Cannot infer the class of {path}. Expected <class>/<image>, "
                "<class>/training_data/<image> or <split>/<class>/<image>. " + ROOT_HELP
            )
        records.append({"path": path.as_posix(), "label": folders[0], "split": split, "group": None})
    layout = "provider-folders" if any(record["split"] for record in records) else "class-folders"
    return records, layout


def _eawag_image_root(root: Path) -> Path | None:
    if root.name == EAWAG_IMAGE_DIR:
        return root
    candidates = [root / EAWAG_IMAGE_DIR, root / "data" / EAWAG_IMAGE_DIR]
    found = [path for path in candidates if path.is_dir()]
    if len(found) > 1:
        raise ValueError("Found two Eawag dataset copies. Set --data-root to the intended zooplankton_0p5x directory.")
    return next(iter(found), None)


def _eawag_provider_split(directory: Path, found: list[Path], labels: set[str]) -> tuple[dict[Path, list[str]], dict]:
    if not directory.is_dir():
        return {}, {}
    available = set(found)
    memberships = defaultdict(Counter)
    counts = Counter()
    issues = []
    for split, filename in EAWAG_SPLIT_FILES.items():
        source = directory / filename
        if not source.is_file():
            issues.append(f"Missing Eawag provider file {source.name}.")
            continue
        for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
            parts = PurePosixPath(line.strip()).parts
            if EAWAG_IMAGE_DIR not in parts or ".." in parts:
                issues.append(f"Invalid Eawag filename at {source.name}:{line_number}.")
                continue
            camera_index = parts.index(EAWAG_IMAGE_DIR)
            relative_path = Path(*parts[camera_index + 1:])
            counts[split] += 1
            if relative_path not in available:
                issues.append(f"Eawag split {split} references an absent image: {relative_path}.")
                continue
            memberships[relative_path][split] += 1
    unassigned = sorted(path.as_posix() for path in available - memberships.keys())
    repeated = {path.as_posix(): dict(counter) for path, counter in sorted(memberships.items()) if sum(counter.values()) > 1}
    cross_split = {path: counts for path, counts in repeated.items() if len(counts) > 1}

    classes_file = directory / "classes_ERIC.npy"
    class_order = None
    if classes_file.is_file():
        names = np.load(classes_file, allow_pickle=False)
        if not isinstance(names, np.ndarray) or names.ndim != 1 or names.dtype.kind != "U":
            raise ValueError(f"Expected a one-dimensional Unicode label array in {classes_file}.")
        class_order = names.tolist()
        if len(set(class_order)) != len(class_order) or set(class_order) != labels:
            raise ValueError(f"Eawag class array and observed label folders disagree in {classes_file}.")

    return {path: sorted(counter) for path, counter in memberships.items()}, {
        "manifest_counts": {split: counts[split] for split in SPLITS},
        "class_order": class_order,
        "matches_local_class_order": class_order == sorted(labels) if class_order is not None else None,
        "unassigned_paths": unassigned,
        "repeated_paths": repeated,
        "cross_split_paths": cross_split,
        "issues": issues,
    }


def _capture_metadata(record: dict, group_by: str) -> dict:
    match = ROI_NAME.fullmatch(Path(record["path"]).stem)
    fields = dict.fromkeys(GROUP_FIELDS.values())
    timestamp_us = None
    if match:
        camera = match["camera"].upper()
        timestamp_us = int(match["unix_us"])
        timestamp = timestamp_us // 1_000_000
        fields["capture_frame"] = f"{camera}:{timestamp_us}:{int(match['frame_us'])}"
        fields["acquisition_day"] = datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
        for name, duration in TIME_BINS.items():
            start = datetime.fromtimestamp(timestamp // duration * duration, tz=timezone.utc)
            fields[name] = start.strftime("%Y-%m-%dT%H:%MZ")
    group = source = None
    if record["group"]:
        group = f"csv:{record['group']}"
        source = "metadata CSV"
    elif (field := GROUP_FIELDS.get(group_by)) and fields[field]:
        group = f"{group_by}:{fields[field]}"
        source = f"{GROUP_TITLES[field]} from ROI filename"
    return {"group": group, "group_source": source, "capture_timestamp_us": timestamp_us, **fields}


def _image_details(root: Path, relative_path: str) -> dict:
    path = root / relative_path
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("Image resolves outside the dataset root.")
    raw_bytes = path.read_bytes()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw_bytes)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Expected a single ROI image, found multiple frames.")
            mode, image_format = image.mode, image.format
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            decoded_hash = hashlib.sha256(struct.pack(">II", *rgb.size))
            decoded_hash.update(rgb.tobytes())
    return {
        "width": rgb.width, "height": rgb.height, "mode": mode, "format": image_format,
        "sha256": decoded_hash.hexdigest(), "file_sha256": file_hash,
    }


def _duplicates(records: list[dict], field: str) -> list[dict]:
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record[field]].append(record)
    return [
        {
            field: key,
            "paths": [record["path"] for record in copies],
            "labels": sorted({record["label"] for record in copies}),
            "provider_splits": sorted({split for record in copies for split in _provider_memberships(record)}),
        }
        for key, copies in sorted(by_hash.items()) if len(copies) > 1
    ]


def _provider_memberships(record: dict) -> list[str]:
    if "provider_memberships" in record:
        return record["provider_memberships"]
    return [record["split"]] if record["split"] else []


def _dimension_stats(records: list[dict], field: str) -> dict:
    values = [record[field] for record in records]
    return {"min": min(values), "median": median(values), "max": max(values)}


def _group_summary(records: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    by_label = {record["label"]: set() for record in records}
    for record in records:
        if (group := record[field]) is not None:
            groups[group].append(record)
            by_label[record["label"]].add(group)
    grouped = sum(map(len, groups.values()))
    return {
        "covered_images": grouped,
        "uncovered_images": len(records) - grouped,
        "distinct_groups": len(groups),
        "groups_with_multiple_images": sum(len(items) > 1 for items in groups.values()),
        "largest_group": max((len(items) for items in groups.values()), default=0),
        "labels_with_fewer_than_three_groups": {
            label: len(values) for label, values in sorted(by_label.items()) if len(values) < 3
        },
        "cross_provider_split_groups": sorted(
            key for key, items in groups.items() if len({split for item in items for split in _provider_memberships(item)}) > 1
        ),
    }


def _describe(candidates: list[dict], records: list[dict], invalid: list[dict]) -> dict:
    labels = sorted({record["label"] for record in candidates})
    counts = Counter(record["label"] for record in records)
    invalid_counts = Counter(record["label"] for record in invalid)
    unique = defaultdict(set)
    class_groups = defaultdict(set)
    class_days = defaultdict(set)
    for record in records:
        unique[record["label"]].add(record["sha256"])
        if record["group"]:
            class_groups[record["label"]].add(record["group"])
        if record["acquisition_day"]:
            class_days[record["label"]].add(record["acquisition_day"])
    counts_by_split = Counter((record["label"], record["split"]) for record in records)
    decoded = _duplicates(records, "sha256")
    exact_files = _duplicates(records, "file_sha256")
    if records:
        dimensions = {axis: _dimension_stats(records, axis) for axis in ("width", "height")}
        imbalance = max(counts.values()) / min(counts.values())
    else:
        dimensions = None
        imbalance = None
    return {
        "labels": labels,
        "label_to_index": {label: index for index, label in enumerate(labels)},
        "label_check": {
            "missing_published_labels": sorted(set(PUBLISHED_LABELS) - set(labels)),
            "unexpected_labels": sorted(set(labels) - set(PUBLISHED_LABELS)),
            "labels_without_readable_images": [label for label in labels if counts[label] == 0],
        },
        "image_counts": {"candidates": len(candidates), "readable": len(records), "invalid": len(invalid)},
        "class_counts": {
            label: {"readable": counts[label], "invalid": invalid_counts[label],
                    "unique_decoded": len(unique[label]),
                    "active_groups": len(class_groups[label]), "acquisition_days": len(class_days[label]),
                    "provider_splits": {split: counts_by_split[label, split] for split in SPLITS},
                    "without_provider_split": counts_by_split[label, None]}
            for label in labels
        },
        "max_min_readable_class_ratio": imbalance,
        "dimensions": dimensions,
        "format_counts": dict(sorted(Counter(record["format"] for record in records).items())),
        "mode_counts": dict(sorted(Counter(record["mode"] for record in records).items())),
        "provider_splits": {split: sum(record["split"] == split for record in records) for split in SPLITS},
        "group_metadata": {
            "active": _group_summary(records, "group"),
            **{field: _group_summary(records, field) for field in GROUP_FIELDS.values()},
            "sources": dict(sorted(Counter(record["group_source"] for record in records if record["group_source"]).items())),
        },
        "duplicates": {
            "raw_file_groups": exact_files,
            "decoded_groups": decoded,
            "duplicate_file_copies": sum(len(group["paths"]) - 1 for group in exact_files),
            "duplicate_decoded_copies": sum(len(group["paths"]) - 1 for group in decoded),
            "conflicting_label_groups": [group for group in decoded if len(group["labels"]) > 1],
            "cross_provider_split_groups": [group for group in decoded if len(group["provider_splits"]) > 1],
        },
    }


def _markdown(report: dict) -> str:
    lines = ["# ZooLake dataset audit", "", f"- Audit status is **{report['status']}**.", f"- Dataset root is `{report['root']}`."]
    if report.get("error"):
        lines += [f"- {report['error']}"]
    if "image_counts" not in report:
        return "\n".join(lines) + "\n"

    counts = report["image_counts"]
    duplicates = report["duplicates"]
    ratio = report["max_min_readable_class_ratio"]
    lines += [
        f"- Detected the `{report['layout']}` layout.",
        f"- Of {counts['candidates']:,} image files, {counts['readable']:,} decoded and {counts['invalid']:,} could not be read.",
        f"- Observed {len(report['labels'])} labels. The ratio of largest to smallest readable class is "
        f"{f'{ratio:.2f}' if ratio else 'unavailable'}.",
        f"- File formats {report['format_counts']}. Image modes {report['mode_counts']}.",
        f"- Exact matching identified {duplicates['duplicate_file_copies']:,} extra copies by file bytes "
        f"and {duplicates['duplicate_decoded_copies']:,} by decoded RGB pixels.",
        f"- Of the decoded duplicate groups, {len(duplicates['conflicting_label_groups']):,} cross labels "
        f"and {len(duplicates['cross_provider_split_groups']):,} cross provider splits.",
    ]
    if report["dimensions"]:
        for axis, stats in report["dimensions"].items():
            lines.append(f"- Image {axis} ranges from {stats['min']} to {stats['max']} px, with a median of {stats['median']:g} px.")
    if provider := report.get("provider"):
        lines += [
            f"- Provider manifest row counts are {provider['manifest_counts']}.",
            f"- The manifests repeat {len(provider['repeated_paths']):,} paths, including "
            f"{len(provider['cross_split_paths']):,} in more than one split. They leave "
            f"{len(provider['unassigned_paths']):,} local images unassigned. The audit found "
            f"{len(provider['issues']):,} missing files or malformed rows.",
        ]
        if provider["class_order"] is not None:
            order = "matches" if provider["matches_local_class_order"] else "differs from"
            lines.append(f"- Official class array matches the label set and {order} the local alphabetical order.")
    for key, description in [
        ("missing_published_labels", "Published labels absent from this inventory"),
        ("unexpected_labels", "Observed labels absent from the published list"),
        ("labels_without_readable_images", "Observed labels without readable images"),
    ]:
        if report["label_check"][key]:
            lines.append(f"- {description} include {', '.join(report['label_check'][key])}.")

    lines += ["", "## Class support", "", "Provider columns count singly assigned rows. The last column includes ambiguous assignments.",
              "", "| Index | Label | Readable | Invalid | Unique decoded | UTC days | Provider train | Provider val | Provider test | Without one provider split |",
              "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for index, label in enumerate(report["labels"]):
        counts = report["class_counts"][label]
        provider = counts["provider_splits"]
        safe_label = label.replace("|", "\\|")
        lines.append(
            f"| {index} | {safe_label} | {counts['readable']} | {counts['invalid']} | {counts['unique_decoded']} | {counts['acquisition_days']} "
            f"| {provider['train']} | {provider['validation']} | {provider['test']} | {counts['without_provider_split']} |"
        )

    groups = report["group_metadata"]
    lines += ["", "## Acquisition groups", "", f"The active grouping rule is `{report['group_by']}`.", "",
              "| Group definition | Images covered | Groups | Groups with multiple images | Largest group | Cross-provider groups | Labels under three groups |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for key, title in GROUP_TITLES.items():
        group = groups[key]
        lines.append(f"| {title} | {group['covered_images']:,} | {group['distinct_groups']:,} "
                     f"| {group['groups_with_multiple_images']:,} | {group['largest_group']:,} "
                     f"| {len(group['cross_provider_split_groups']):,} "
                     f"| {', '.join(f'{label} ({size})' for label, size in group['labels_with_fewer_than_three_groups'].items()) or 'None'} |")
    files = report["metadata_files"]
    if files:
        lines += ["", f"Found {len(files):,} metadata sidecars. The JSON records their paths. Only explicit "
                  "CSV groups and matching ROI filenames contribute acquisition groups."]
    lines += ["", "Exact RGB hashes cover EXIF-oriented pixels at native resolution. Crop, compression, "
              "and near-duplicate variants can evade exact matching. See `audit.json` for invalid files, "
              "duplicate paths and group coverage."]
    return "\n".join(lines) + "\n"


def _write_text(path: Path, contents: str) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(contents)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def audit_dataset(root: Path, output: Path, metadata: Path | None = None, *, group_by: str = "frame") -> dict:
    """Save the root-relative image inventory and an audit summary to ``output``.

    Directory roots with standard image folders are auto-discovered. A strict CSV
    can explicitly provide labels, provider splits, and capture groups; its paths
    must match all discovered image paths. ``group_by`` selects the timestamp
    grouping granularity for original filenames without an explicit CSV group.
    """
    if group_by not in GROUP_FIELDS and group_by != "none":
        options = ", ".join([*GROUP_FIELDS, "none"])
        raise ValueError(f"Choose an acquisition grouping rule from {options}.")
    root = Path(root).expanduser().resolve()
    output = Path(output)
    report = {"schema_version": 1, "status": "ok", "root": str(root),
              "group_by": group_by, "labels": [], "records": [], "invalid_images": []}
    try:
        if not root.is_dir():
            raise FileNotFoundError(f"ZooLake root is unavailable or is not an extracted directory: {root}.")
        image_root = _eawag_image_root(root) or root
        found, sidecars = _inventory(image_root)
        if not found:
            raise FileNotFoundError(f"No supported image files were found under {image_root}.")
        relative_prefix = image_root.relative_to(root)
        provider = {}
        if metadata:
            candidates = _csv_records(Path(metadata).expanduser(), [relative_prefix / path for path in found])
            layout = "metadata-csv"
        else:
            candidates, layout = _folder_records(found)
            if image_root.name == EAWAG_IMAGE_DIR:
                layout = "eawag-release"
                split_dir = image_root.parent / EAWAG_SPLIT_DIR
                memberships, provider = _eawag_provider_split(split_dir, found, {c["label"] for c in candidates})
                for candidate in candidates:
                    path = Path(candidate["path"])
                    candidate["path"] = (relative_prefix / path).as_posix()
                    choices = memberships.get(path, [])
                    candidate["provider_memberships"] = choices
                    candidate["split"] = choices[0] if len(choices) == 1 else None
        sidecars = [(relative_prefix / path).as_posix() for path in sidecars]
        spelling_collisions = defaultdict(set)
        for candidate in candidates:
            spelling_collisions[candidate["label"].casefold()].add(candidate["label"])
        ambiguous = [sorted(group) for group in spelling_collisions.values() if len(group) > 1]
        if ambiguous:
            raise ValueError(f"Class names differ only by case: {ambiguous}.")
        for candidate in candidates:
            try:
                details = _image_details(root, candidate["path"])
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
                    Image.DecompressionBombWarning) as error:
                report["invalid_images"].append({**candidate, "error": str(error)})
                continue
            capture = _capture_metadata(candidate, group_by)
            report["records"].append({**candidate, **capture, **details})
        report.update(_describe(candidates, report["records"], report["invalid_images"]))
        report.update({"layout": layout, "metadata_source": str(metadata) if metadata else None,
                       "metadata_files": sidecars, "provider": provider})
        if not report["records"]:
            report.update(status="invalid", error="No images decoded successfully. Check the extracted dataset files.")
    except (FileNotFoundError, PermissionError) as error:
        report.update(status="missing", error=f"{error} {ROOT_HELP}")
    except (ValueError, OSError, csv.Error) as error:
        report.update(status="invalid", error=str(error))

    output.mkdir(parents=True, exist_ok=True)
    _write_text(output / "audit.md", _markdown(report))
    _write_text(output / "audit.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
