"""Shared aspect-preserving preprocessing and discrete image symmetries."""

from pathlib import Path

from PIL import Image, ImageOps


def load_canvas(path: Path, side: int) -> Image.Image:
    """Shrink oversized ROIs and center them on a black RGB canvas."""
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    image.thumbnail((side, side), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (side, side))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


_ROTATIONS = (
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
)


def d4_view(image: Image.Image, index: int) -> Image.Image:
    """Compute one of four rotations of the original or reflected image."""
    if index not in range(8):
        raise ValueError(f"D4 transform index must be in [0, 7], got {index}")
    source = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if index >= 4 else image
    rotation = _ROTATIONS[index % 4]
    return source if rotation is None else source.transpose(rotation)


def d4_views(image: Image.Image):
    """Yield all eight views, with the untransformed image first."""
    return (d4_view(image, index) for index in range(8))
