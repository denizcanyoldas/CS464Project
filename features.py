"""Image preprocessing and features, shared by training and single-image inference."""
import math

import cv2
import numpy as np
from PIL import Image

SIDE = 128
HOG = cv2.HOGDescriptor((SIDE, SIDE), (64, 64), (32, 32), (32, 32), 9)


def prepare(image):
    """Center the full image and leave enough black margin for diagonal rotations."""
    image = image.convert("RGB")
    width, height = image.size
    scale = (SIDE - 2) / math.hypot(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    tile = Image.new("RGB", (SIDE, SIDE))
    tile.paste(image.resize(size, Image.Resampling.BILINEAR),
               ((SIDE - size[0]) // 2, (SIDE - size[1]) // 2))
    return np.asarray(tile)


def rotate(tile, angle):
    """Exact quarter turns, otherwise bilinear interpolation over a black background."""
    if angle % 90 == 0:
        return np.ascontiguousarray(np.rot90(tile, (angle // 90) % 4))
    return np.asarray(Image.fromarray(tile).rotate(angle, Image.Resampling.BILINEAR,
                                                   expand=False, fillcolor=(0, 0, 0)))


def d4(tile):
    for angle in (0, 90, 180, 270):
        view = rotate(tile, angle)
        yield view
        yield np.ascontiguousarray(view[:, ::-1])


def describe(tile):
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    gradients = HOG.compute(gray).ravel()
    # Otsu separates bright foreground from the dark camera background.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hu = np.abs(cv2.HuMoments(cv2.moments(mask, binaryImage=True)).ravel())
    shape = -np.log10(np.maximum(hu, 1e-20))
    colors = []
    for channel in range(3):
        histogram = cv2.calcHist([tile], [channel], mask, [16], [0, 256]).ravel()
        colors.extend(histogram / max(histogram.sum(), 1))
    return np.concatenate((gradients, colors, shape)).astype(np.float32)
