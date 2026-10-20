"""Fixed-camera helpers: background difference to propose candidate regions.

A security camera does not move, so "what changed since the clean reference?" is a
cheap first filter that runs on the camera server. Only the changed regions are sent
to the vision model, cropped at full resolution. People and trolleys also show up as
changes; the model's job is to say which changes are liquid on the floor.

    cands = propose_candidates(reference_bgr, current_bgr)
    for c in cands:  # each is a normalised (x, y, w, h) box, largest first
        detector.classify_pair(ref_path, cur_path, crop=c)
"""
from __future__ import annotations

import cv2
import numpy as np

Box = tuple[float, float, float, float]


def _prep(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (7, 7), 0)


def change_mask(reference: np.ndarray, current: np.ndarray, threshold: int = 22) -> np.ndarray:
    """Binary mask of pixels that differ from the reference beyond `threshold` (0..255)."""
    diff = cv2.absdiff(_prep(reference), _prep(current))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def propose_candidates(
    reference: np.ndarray,
    current: np.ndarray,
    later: np.ndarray | None = None,
    threshold: int = 12,
    min_area_frac: float = 0.0006,
    pad_frac: float = 0.25,
    min_crop_frac: float = 0.18,
    max_candidates: int = 12,
    merge_max_area: float = 0.3,
) -> list[Box]:
    """Return normalised (x, y, w, h) crop boxes around changed regions, largest first.

    `later` is an optional frame from the same camera a few seconds after `current`.
    When given, only changes present in BOTH frames count: a spill stays put, a person
    walks on. This is the cheapest and strongest filter a camera server has.

    Boxes are padded and widened to at least `min_crop_frac` of the frame so the model
    sees some surrounding floor for context, and merged when they overlap.
    """
    H, W = current.shape[:2]
    mask = change_mask(reference, current, threshold)
    persistent = moving = None
    if later is not None:
        persistent = change_mask(reference, later, threshold)   # still different from the empty floor later
        moving = change_mask(current, later, threshold)         # changed between now and later (walkers, escalators)
        mask = cv2.bitwise_and(mask, cv2.dilate(persistent, np.ones((15, 15), np.uint8)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area_frac * W * H:
            continue
        boxes.append((x, y, w, h))
        if persistent is not None:
            # a spill is still there later AND looks the same later; a walker is gone, an
            # escalator or a standing-then-moving person keeps changing. Score in [0, 1].
            p = float(persistent[y:y + h, x:x + w].mean()) / 255.0
            s = 1.0 - float(moving[y:y + h, x:x + w].mean()) / 255.0
            scores.append(p * s)
        else:
            scores.append(float(w * h))
    # most spill-like first (ties by area), so a small spill is not crowded out by people
    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], -boxes[i][2] * boxes[i][3]))
    boxes = [boxes[i] for i in order]

    padded: list[Box] = []
    for x, y, w, h in boxes[: max_candidates * 4]:
        px, py = int(w * pad_frac), int(h * pad_frac)
        side_w = max(w + 2 * px, int(min_crop_frac * W))
        side_h = max(h + 2 * py, int(min_crop_frac * H))
        cx, cy = x + w / 2, y + h / 2
        nx = max(0.0, min(1 - side_w / W, (cx - side_w / 2) / W))
        ny = max(0.0, min(1 - side_h / H, (cy - side_h / 2) / H))
        padded.append((nx, ny, min(1.0, side_w / W), min(1.0, side_h / H)))

    # Merge neighbours into one crop while the union stays small enough to keep detail:
    # several standing people near a spill then cost one call instead of four.
    out: list[Box] = []
    for box in padded:
        merged = False
        for i, o in enumerate(out):
            u = _union(box, o)
            if _iou(box, o) > 0.5 or (u[2] * u[3] <= merge_max_area and _iou(box, o) > 0.0):
                out[i] = u
                merged = True
                break
        if not merged:
            out.append(box)
        if len(out) >= max_candidates:
            break
    return out


def _union(a: Box, b: Box) -> Box:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def _iou(a: Box, b: Box) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def box_to_frame(box_in_crop: list[float], crop: Box) -> list[float]:
    """Map a bbox the model gave inside a crop back to full-frame fractions."""
    bx, by, bw, bh = box_in_crop
    cx, cy, cw, ch = crop
    return [cx + bx * cw, cy + by * ch, bw * cw, bh * ch]


def contains(crop: Box, point: tuple[float, float]) -> bool:
    x, y = point
    return crop[0] <= x <= crop[0] + crop[2] and crop[1] <= y <= crop[1] + crop[3]
