"""Image loading and encoding for the Claude vision API.

Claude bills images by pixel count (roughly width*height/750 tokens) and
downsizes anything over ~1.15 megapixels or 1568px on the long side.
Resizing locally first keeps the upload small and the cost predictable.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, ImageOps

MAX_LONG_SIDE = 1568
MAX_PIXELS = 1_150_000
JPEG_QUALITY = 85


def load_resized(path: str | Path, crop: tuple[float, float, float, float] | None = None) -> Image.Image:
    """Load, optionally crop to a normalised (x, y, w, h) box, then cap the pixel count.

    Cropping happens BEFORE the resize, so a small far-away region keeps its full
    native resolution instead of being downsampled with the rest of the frame.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if crop is not None:
        W, H = img.size
        x, y, w, h = crop
        box = (int(max(0, x) * W), int(max(0, y) * H), int(min(1, x + w) * W), int(min(1, y + h) * H))
        if box[2] - box[0] >= 8 and box[3] - box[1] >= 8:
            img = img.crop(box)
    w, h = img.size
    scale = min(1.0, MAX_LONG_SIDE / max(w, h), (MAX_PIXELS / (w * h)) ** 0.5)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return img


def encode_image(path: str | Path, crop: tuple[float, float, float, float] | None = None) -> tuple[str, str, tuple[int, int]]:
    """Return (base64_data, media_type, (width, height)) ready for an image block."""
    img = load_resized(path, crop)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return data, "image/jpeg", img.size


def estimate_image_tokens(size: tuple[int, int]) -> int:
    w, h = size
    return int(w * h / 750)
