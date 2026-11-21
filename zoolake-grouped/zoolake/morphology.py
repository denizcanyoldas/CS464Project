"""Single-view, pixel-only shape and texture descriptors for square ZooLake canvases."""

import math

import numpy as np
from PIL import Image


def _uniform_patterns():
    """Uniform circular patterns have at most two transitions; all others share bin 9."""
    lookup = []
    for code in range(256):
        shifted = ((code << 1) & 255) | (code >> 7)
        lookup.append(code.bit_count() if (code ^ shifted).bit_count() <= 2 else 9)
    return np.array(lookup, dtype=np.uint8)


_TEXTURE_BINS = _uniform_patterns()
_NEIGHBORS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _shape(mask, side):
    y, x = np.nonzero(mask)
    if not len(x):
        return (0.0,) * 7
    area = len(x)
    xy = np.column_stack((x - x.mean(), y - y.mean()))
    minor, major = np.sqrt(np.maximum(np.linalg.eigvalsh(xy.T @ xy / area), 0))
    # Count all exposed pixel edges, including those on canvas borders and inside holes.
    padded = np.pad(mask, 1)
    perimeter = (np.count_nonzero(padded[1:] != padded[:-1])
                 + np.count_nonzero(padded[:, 1:] != padded[:, :-1]))
    radius90 = np.sqrt(np.quantile(np.sum(xy * xy, axis=1), .90))
    return (area / side**2, minor / side, major / side,
            minor / major if major else 0.0, perimeter / side,
            4 * math.pi * area / perimeter**2, radius90 / side)


def morphology_features(image: Image.Image) -> np.ndarray:
    """Seven measurements at three brightness thresholds and ten local texture bins."""
    gray = np.asarray(image.convert("L"))
    if gray.shape[0] != gray.shape[1]:
        raise ValueError("Morphology features require a square preprocessed canvas.")
    side = gray.shape[0]
    result = []
    for threshold in (12, 40, 96):
        result.extend(_shape(gray > threshold, side))

    foreground = gray > 12
    # Compare with a small absolute margin to suppress single-level intensity changes.
    center = gray.astype(np.int16) + 3
    padded = np.pad(gray, 1)
    code = np.zeros_like(gray)
    for bit, (dy, dx) in enumerate(_NEIGHBORS):
        neighbor = padded[1 + dy:1 + dy + side, 1 + dx:1 + dx + side]
        code |= (neighbor >= center).astype(np.uint8) << bit
    histogram = np.bincount(_TEXTURE_BINS[code[foreground]], minlength=10)
    result.extend(histogram / max(1, histogram.sum()))
    return np.asarray(result, dtype=np.float32)
