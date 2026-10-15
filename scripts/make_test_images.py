"""Generate a few synthetic floor images for a first smoke test when you have no real photos yet.

    python scripts/make_test_images.py data/frames/synthetic

Makes tiled-floor renders with and without a dark puddle, plus a glossy-reflection
negative. Synthetic images are only for checking that the pipeline runs end to end;
accuracy numbers on them mean nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def floor(w=1024, h=768, tile=96, base=(200, 195, 185)) -> np.ndarray:
    img = np.full((h, w, 3), base, np.uint8)
    noise = np.random.default_rng(0).integers(-8, 8, (h, w, 1), np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for y in range(0, h, tile):
        cv2.line(img, (0, y), (w, y), (160, 155, 145), 2)
    for x in range(0, w, tile):
        cv2.line(img, (x, 0), (x, h), (160, 155, 145), 2)
    # perspective: squash the top so it looks like a camera looking down a corridor
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.3, 0], [w * 0.7, 0], [w, h], [0, h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(90, 90, 95))


def puddle(img: np.ndarray, cx: int, cy: int, rx: int, ry: int, color, alpha=0.55) -> np.ndarray:
    layer = img.copy()
    cv2.ellipse(layer, (cx, cy), (rx, ry), 15, 0, 360, color, -1)
    cv2.ellipse(layer, (cx + rx // 2, cy + ry // 3), (rx // 3, ry // 2), -20, 0, 360, color, -1)
    out = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
    cv2.ellipse(out, (cx - rx // 4, cy - ry // 4), (rx // 5, ry // 8), 15, 0, 360, (235, 235, 235), -1)  # highlight
    return out


def reflection(img: np.ndarray) -> np.ndarray:
    layer = img.copy()
    h, w = img.shape[:2]
    for i in range(3):
        x = int(w * (0.3 + 0.2 * i))
        cv2.rectangle(layer, (x, int(h * 0.1)), (x + 40, int(h * 0.95)), (255, 255, 250), -1)
    return cv2.addWeighted(layer, 0.25, img, 0.75, 0)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/frames/synthetic")
    out.mkdir(parents=True, exist_ok=True)
    base = floor()
    cv2.imwrite(str(out / "neg_clean_floor.jpg"), base)
    cv2.imwrite(str(out / "neg_glossy_reflection.jpg"), reflection(base))
    cv2.imwrite(str(out / "pos_water_puddle.jpg"), puddle(base, 520, 560, 150, 70, (150, 140, 120)))
    cv2.imwrite(str(out / "pos_dark_spill.jpg"), puddle(base, 400, 500, 120, 60, (40, 30, 25), 0.85))
    cv2.imwrite(str(out / "pos_small_oil.jpg"), puddle(base, 650, 620, 45, 22, (60, 90, 110), 0.7))
    print(f"wrote 5 images to {out}")


if __name__ == "__main__":
    main()
