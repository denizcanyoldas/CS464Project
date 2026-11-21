"""Read the authors' public ZIP directly. No extraction or pickle loading is needed."""
import csv
import hashlib
import io
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from PIL import Image

SOURCE = ("https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7/"
          "resource/1cc785fa-36c2-447d-bb11-92ce1d1f3f2d/download/data.zip")
ARCHIVE_SHA256 = "ce038e19af73dd5aff0beb9ed0dc012bb0314efeb341337667b62d1e1c383d4f"
SPLIT_DIR = "data/zoolake_train_test_val_separated/"


def read_image(archive, path):
    with Image.open(io.BytesIO(archive.read(path))) as image:
        return image.convert("RGB")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(archive):
    memberships = defaultdict(list)
    for split in ("train", "val", "test"):
        for line in archive.read(SPLIT_DIR + split + "_filenames.txt").decode().splitlines():
            path = line.replace("/Zoolake_Dataset/", "data/", 1)
            memberships[path].append(split)

    paths = sorted(n for n in archive.namelist() if n.endswith(".jpeg"))
    if set(memberships) - set(paths):
        raise ValueError("A publisher split references a missing JPEG.")
    rows, groups = [], defaultdict(list)
    for path in paths:
        image = read_image(archive, path)
        digest = hashlib.sha256(f"{image.width},{image.height}:".encode() + image.tobytes()).hexdigest()
        row = dict(path=path, label=PurePosixPath(path).parts[-3],
                   width=image.width, height=image.height, pixel_sha256=digest,
                   listed_splits="|".join(memberships.get(path, [])))
        rows.append(row)
        groups[digest].append(row)

    clean, excluded = [], []
    for row in rows:
        group = groups[row["pixel_sha256"]]
        if len(group) > 1:
            labels = {entry["label"] for entry in group}
            reason = "conflicting_labels" if len(labels) > 1 else "duplicate_group"
            excluded.append(dict(**row, reason=reason))
            continue
        split = row["listed_splits"]
        if split not in ("train", "val", "test"):
            raise ValueError(f"Missing or ambiguous split for {row['path']}")
        clean.append(dict(**row, split=split))
    totals = Counter(s for appearances in memberships.values() for s in appearances)
    return clean, excluded, dict(totals)
