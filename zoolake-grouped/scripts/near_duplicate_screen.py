"""Screen fixed ZooLake train and validation folds for highly similar images."""

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits


SIDE = 32
BATCH_SIZE = 128
MIN_CONTRAST = 0.02
ASPECT_NUMERATOR = 21
ASPECT_DENOMINATOR = 20
PRIMARY = {"min_cosine": 0.995, "max_relative_l2": 0.10}
SENSITIVITY = {"min_cosine": 0.98, "max_relative_l2": 0.20}


def load_thumbnails(root, rows):
    thumbnails = np.empty((len(rows), SIDE, SIDE), dtype=np.float32)
    dimensions = np.empty((2, len(rows)), dtype=np.int64)
    for index, row in enumerate(rows):
        source = (root / row["path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError(f"Image path is outside the audited data root: {row['path']}")
        raw = source.read_bytes()
        if row.get("file_sha256") and sha256(raw).hexdigest() != row["file_sha256"]:
            raise ValueError(f"Image bytes changed since the audit: {row['path']}")
        with Image.open(BytesIO(raw)) as image:
            gray = ImageOps.exif_transpose(image).convert("L")
            dimensions[:, index] = gray.size
            resized = gray.resize((SIDE, SIDE), Image.Resampling.LANCZOS)
            thumbnails[index] = np.asarray(resized, dtype=np.float32) / 255.0
    flat = thumbnails.reshape(len(rows), -1)
    norms = np.linalg.norm(flat, axis=1)
    readable_contrast = flat.std(axis=1) >= MIN_CONTRAST
    return thumbnails, dimensions, norms, readable_contrast


def match_pairs(train, validation):
    train_gray, (train_width, train_height), train_norm, train_contrast = train
    val_gray, val_dimensions, val_norm, val_contrast = validation
    train_flat = train_gray.reshape(len(train_gray), -1)
    train_unit = train_flat / np.maximum(train_norm[:, None], 1e-12)
    matches = {"aligned_primary": set(), "d4_primary": set(), "d4_sensitivity": set()}

    # The aspect ratio becomes its reciprocal after a quarter-turn. Applying
    # transforms to validation only suffices for all relative D4 orientations.
    for reflection in (False, True):
        base = val_gray[:, :, ::-1] if reflection else val_gray
        for turns in range(4):
            transformed = np.ascontiguousarray(np.rot90(base, k=turns, axes=(1, 2)))
            val_flat = transformed.reshape(len(transformed), -1)
            val_unit = val_flat / np.maximum(val_norm[:, None], 1e-12)
            val_width, val_height = val_dimensions[::-1] if turns % 2 else val_dimensions
            for start in range(0, len(val_gray), BATCH_SIZE):
                stop = start + BATCH_SIZE
                cosine = val_unit[start:stop] @ train_unit.T
                # Cross-multiplication gives an inclusive 5% aspect test at
                # exact boundary ratios, avoiding rounded logarithms.
                val_ratio_num = val_width[start:stop, None] * train_height
                train_ratio_num = val_height[start:stop, None] * train_width
                aspect_match = (
                    (ASPECT_DENOMINATOR * val_ratio_num <= ASPECT_NUMERATOR * train_ratio_num)
                    & (ASPECT_DENOMINATOR * train_ratio_num <= ASPECT_NUMERATOR * val_ratio_num)
                )
                candidate = (
                    (cosine >= SENSITIVITY["min_cosine"]) & aspect_match
                    & val_contrast[start:stop, None] & train_contrast
                )
                for local_index, train_index in zip(*np.nonzero(candidate)):
                    val_index = start + local_index
                    relative_error = np.linalg.norm(val_flat[val_index] - train_flat[train_index]) / max(
                        val_norm[val_index], train_norm[train_index]
                    )
                    if relative_error > SENSITIVITY["max_relative_l2"]:
                        continue
                    pair = (int(val_index), int(train_index))
                    matches["d4_sensitivity"].add(pair)
                    if cosine[local_index, train_index] >= PRIMARY["min_cosine"] and relative_error <= PRIMARY["max_relative_l2"]:
                        matches["d4_primary"].add(pair)
                        if not reflection and turns == 0:
                            matches["aligned_primary"].add(pair)
    return matches


def describe_pairs(pairs, train, validation):
    label_pairs = Counter()
    capture_gaps = []
    same_day = 0
    for val_index, train_index in sorted(pairs):
        val_row, train_row = validation[val_index], train[train_index]
        label_pairs[val_row["label"], train_row["label"]] += 1
        val_day = val_row.get("acquisition_day")
        same_day += bool(val_day and val_day == train_row.get("acquisition_day"))
        val_time = val_row.get("capture_timestamp_us")
        train_time = train_row.get("capture_timestamp_us")
        if val_time is not None and train_time is not None:
            capture_gaps.append(abs(val_time - train_time) / 1_000_000)

    return {
        "unique_pairs": len(pairs),
        "distinct_validation_images": len({val_index for val_index, _ in pairs}),
        "distinct_train_images": len({train_index for _, train_index in pairs}),
        "same_label_pairs": sum(count for (val_label, train_label), count in label_pairs.items() if val_label == train_label),
        "same_utc_day_pairs": same_day,
        "pairs_with_capture_timestamps": len(capture_gaps),
        "min_capture_gap_seconds": min(capture_gaps, default=None),
        "pairs_at_most_30_seconds_apart": sum(gap <= 30 for gap in capture_gaps),
        "pairs_at_most_180_seconds_apart": sum(gap <= 180 for gap in capture_gaps),
        "label_pairs": [
            {"validation": val_label, "train": train_label, "count": count}
            for (val_label, train_label), count in sorted(label_pairs.items())
        ],
    }


def screen(split_path, root_override):
    split_bytes = split_path.read_bytes()
    manifest = json.loads(split_bytes)
    root = Path(root_override or os.getenv("ZOOLAKE_DATA_ROOT") or manifest["root"]).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("ZooLake directory unavailable. Use --data-root with the archive extraction used for auditing.")
    train = [row for row in manifest["records"] if row["split"] == "train"]
    validation = [row for row in manifest["records"] if row["split"] == "validation"]
    if not train or not validation:
        raise ValueError("The split must contain train and validation records.")

    with threadpool_limits(limits=4):
        train_views = load_thumbnails(root, train)
        val_views = load_thumbnails(root, validation)
        matches = match_pairs(train_views, val_views)
    train_eligible = int(np.count_nonzero(train_views[3]))
    val_eligible = int(np.count_nonzero(val_views[3]))
    return {
        "schema_version": 1,
        "split_sha256": sha256(split_bytes).hexdigest(),
        "input_images": {"train": len(train), "validation": len(validation)},
        "low_contrast_excluded": {"train": len(train) - train_eligible, "validation": len(validation) - val_eligible},
        "image_pairs_scanned": len(train) * len(validation),
        "d4_comparisons_scanned": 8 * len(train) * len(validation),
        "parameters": {
            "thumbnail_pixels": [SIDE, SIDE],
            "thumbnail_transform": "EXIF-oriented grayscale, Lanczos resize, intensities divided by 255",
            "min_thumbnail_stddev": MIN_CONTRAST,
            "max_aspect_ratio_factor_inclusive": ASPECT_NUMERATOR / ASPECT_DENOMINATOR,
            "relative_l2_definition": "L2 difference divided by the larger thumbnail norm",
            "primary": PRIMARY,
            "sensitivity": SENSITIVITY,
        },
        "results": {name: describe_pairs(pairs, train, validation) for name, pairs in matches.items()},
        "interpretation": (
            "The thresholds identify visually similar candidates. Capture identity requires independent evidence. "
            "Crop shifts, non-D4 orientations, and illumination changes can evade the screen. "
            "Related captures from one acquisition session can still span partitions."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("artifacts/splits/splits.json"))
    parser.add_argument("--data-root", type=Path, help="Defaults to ZOOLAKE_DATA_ROOT or the audited root in the split")
    parser.add_argument("--out", type=Path, default=Path("results/near_duplicate_screen.json"))
    args = parser.parse_args()
    try:
        report = screen(args.split, args.data_root)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, KeyError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    results = report["results"]
    print(f"Found {results['d4_primary']['unique_pairs']} primary and "
          f"{results['d4_sensitivity']['unique_pairs']} sensitivity train–validation pairs.")


if __name__ == "__main__":
    main()
