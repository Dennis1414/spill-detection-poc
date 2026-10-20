"""Composite a synthetic liquid spill onto a real floor frame.

Used to make positive before/after pairs from fixed-camera footage that has no
spills in it. Real liquid on a floor does not hide the floor: it darkens it and
tints it (light passes through the liquid twice), keeps the floor texture visible,
has a soft edge with a slightly brighter wet rim, and shows a stretched reflection
of the room lighting. The overlay follows that physically rather than pasting an
opaque shape, so the model is judged on liquid-like cues, not on sticker edges.
"""
from __future__ import annotations

import random

import cv2
import numpy as np

# kind -> (transmission colour BGR 0..1 at full depth, depth range, sheen strength)
# transmission < 1 darkens; e.g. coffee passes red, absorbs blue.
KINDS = {
    "water": ((0.66, 0.67, 0.66), (0.6, 1.0), 0.55),   # wet concrete/stone darkens by roughly a third
    "coffee": ((0.30, 0.48, 0.70), (0.6, 1.0), 0.35),
    "oil": ((0.42, 0.44, 0.46), (0.6, 1.0), 0.75),
    "juice": ((0.35, 0.75, 0.95), (0.55, 0.9), 0.35),
    "cola": ((0.28, 0.36, 0.45), (0.6, 1.0), 0.5),
}
# "milk" was dropped: an opaque white liquid cannot be rendered as a transmission tint, and the
# pale wash it produced read as glare. Keep light liquids for real footage.


def _blob_mask(h: int, w: int, cx: float, cy: float, rx: float, ry: float, rng: random.Random) -> np.ndarray:
    """Soft irregular ellipse mask in 0..1 with a few lobes, squashed like a puddle seen from above."""
    pts = []
    n = 36
    for i in range(n):
        t = 2 * np.pi * i / n
        # gentle, low-frequency wobble: real puddles are rounded blobs, not stars
        wob = 1 + 0.12 * np.sin(2 * t + rng.random() * 6) + 0.05 * np.sin(5 * t + rng.random() * 6)
        pts.append((cx + rx * wob * np.cos(t), cy + ry * wob * np.sin(t)))
    mask = np.zeros((h, w), np.float32)
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 1.0)
    for _ in range(rng.randint(0, 3)):  # satellite drops
        dx, dy = rng.uniform(-1.5, 1.5) * rx, rng.uniform(-1.1, 1.1) * ry
        r = max(1, int(min(rx, ry) * rng.uniform(0.1, 0.2)))
        cv2.circle(mask, (int(cx + dx), int(cy + dy)), r, 1.0, -1)
    sigma = max(0.8, min(rx, ry) * 0.2)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), sigma), 0, 1)


def composite_spill(
    img: np.ndarray,
    center: tuple[float, float],
    width_frac: float,
    kind: str,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Return (new image, normalised bbox [x, y, w, h]) with a spill of `kind` centred at `center`.

    width_frac is the spill width as a fraction of the frame width. The height is
    squashed to look like a puddle seen from a high camera.
    """
    rng = random.Random(seed)
    trans, (d_lo, d_hi), sheen = KINDS[kind]
    H, W = img.shape[:2]
    cx, cy = center[0] * W, center[1] * H
    rx = max(4.0, width_frac * W / 2)
    ry = max(3.0, rx * rng.uniform(0.35, 0.55))
    mask = _blob_mask(H, W, cx, cy, rx, ry, rng)
    depth = rng.uniform(d_lo, d_hi)
    # thinner at the edges: depth profile follows the (already soft) mask
    m = mask * depth

    out = img.astype(np.float32)
    t = np.array(trans, np.float32)
    # transmission: floor colour multiplied by the liquid colour, proportional to depth
    factor = 1 - m[..., None] * (1 - t)
    out = out * factor
    # wet floors look slightly more saturated and contrasty
    mean = out.mean(axis=2, keepdims=True)
    out = out + (out - mean) * 0.15 * m[..., None]

    # wet rim: a thin brighter edge where the liquid surface curves (meniscus)
    edge = np.clip(cv2.Laplacian(mask, cv2.CV_32F, ksize=3), 0, None)
    edge = cv2.GaussianBlur(edge, (0, 0), 1.0)
    edge = edge / (edge.max() + 1e-6) * 0.35 * sheen
    out = out + (255 - out) * edge[..., None]

    # stretched reflection of the room lighting, offset toward the top of the puddle
    hl = np.zeros((H, W), np.float32)
    hx, hy = cx - rx * rng.uniform(0.1, 0.35), cy - ry * rng.uniform(0.15, 0.4)
    cv2.ellipse(hl, (int(hx), int(hy)), (max(1, int(rx * 0.35)), max(1, int(ry * 0.22))),
                rng.uniform(-15, 15), 0, 360, 1.0, -1)
    hl = cv2.GaussianBlur(hl, (0, 0), max(1.0, rx * 0.15)) * mask * sheen
    out = out + (255 - out) * hl[..., None] * 0.8

    out = np.clip(out, 0, 255).astype(np.uint8)
    ys, xs = np.where(mask > 0.15)
    if len(xs) == 0:
        return out, [center[0], center[1], 0.0, 0.0]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return out, [x0 / W, y0 / H, (x1 - x0 + 1) / W, (y1 - y0 + 1) / H]
