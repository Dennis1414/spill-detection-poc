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


def load_resized(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_LONG_SIDE / max(w, h), (MAX_PIXELS / (w * h)) ** 0.5)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return img


def encode_image(path: str | Path) -> tuple[str, str, tuple[int, int]]:
    """Return (base64_data, media_type, (width, height)) ready for an image block."""
    img = load_resized(path)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return data, "image/jpeg", img.size


def estimate_image_tokens(size: tuple[int, int]) -> int:
    w, h = size
    return int(w * h / 750)
