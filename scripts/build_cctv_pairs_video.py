"""Build before/after pairs from a real fixed-camera video (CCTV, walkway, corridor).

    python scripts/build_cctv_pairs_video.py data/raw/cctv/vtest.avi --name walkway --pairs 30

Reference = per-pixel median of frames from the first seconds of the clip, which
removes passers-by and leaves the empty floor (the same trick a camera server
would use). Current frames are sampled later in the clip, so people are mid-walk.

  negative pair : reference vs sampled frame (real motion only)
  positive pair : same, plus a synthetic spill composited onto open floor
  static pairs  : reference vs reference (+ spill) for the no-motion case

Appends to data/cctv_pairs/pairs.jsonl so several videos can be combined.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spilldetect.synth import KINDS, composite_spill  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cctv_pairs import floor_point  # noqa: E402

OUT = Path("data/cctv_pairs")


def median_background(cap, seconds: float, fps: float, max_side: int) -> np.ndarray:
    frames = []
    n = int(seconds * fps)
    step = max(1, n // 40)
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(_resize(f, max_side))
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _resize(f: np.ndarray, max_side: int) -> np.ndarray:
    h, w = f.shape[:2]
    s = min(1.0, max_side / max(h, w))
    return f if s == 1.0 else cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--name", required=True, help="short id used in pair ids and file names")
    ap.add_argument("--pairs", type=int, default=30, help="number of sampled current frames (each gives a pos and a neg)")
    ap.add_argument("--bg-seconds", type=float, default=8.0, help="seconds used for the median background")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--later-seconds", type=float, default=2.0,
                    help="gap to the 'later' frame used for the persistence check")
    ap.add_argument("--floor-y", type=float, nargs=2, default=(0.3, 0.95),
                    help="vertical band (fractions) where open floor can be, e.g. 0.3 0.95")
    ap.add_argument("--floor-poly", default=None,
                    help="floor zone polygon as 'x,y x,y x,y ...' in frame fractions; spills are placed inside it "
                         "(in a real install the integrator draws this once per camera)")
    args = ap.parse_args()
    poly = None
    if args.floor_poly:
        poly = np.array([[float(v) for v in pt.split(",")] for pt in args.floor_poly.split()], np.float32)

    rng = random.Random(args.seed)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Frame counts and seeking are unreliable for MPEG streams, so read the whole clip
    # sequentially (downscaled) and work from memory.
    all_frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        all_frames.append(_resize(f, args.max_side))
    total = len(all_frames)
    if total < 20:
        sys.exit(f"only {total} decodable frames in {args.video}")
    n_bg = min(total // 2, int(args.bg_seconds * fps))
    ref = np.median(np.stack(all_frames[: n_bg : max(1, n_bg // 40)]), axis=0).astype(np.uint8)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    ref_out = OUT / "images" / f"{args.name}_ref.jpg"
    cv2.imwrite(str(ref_out), ref, [cv2.IMWRITE_JPEG_QUALITY, 92])

    start = n_bg + 1
    idxs = sorted(rng.sample(range(start, total), min(args.pairs, total - start)))
    records = []

    # temporarily narrow floor_point's vertical band through a wrapper
    def inside(x, y, w):
        lo, hi = args.floor_y
        if not (lo <= y <= hi):
            return False
        if poly is None:
            return True
        # the whole spill footprint must sit inside the floor zone, not just its centre
        return cv2.pointPolygonTest(poly, (float(x), float(y)), True) > 0.6 * w

    def fp(r, c, w):
        for _ in range(120):
            x, y = floor_point(r, c, rng, w)
            if inside(x, y, w):
                return x, y
        return 0.5, sum(args.floor_y) / 2

    later_gap = int(args.later_seconds * fps)
    for i, idx in enumerate(idxs):
        cur = all_frames[idx]
        later = all_frames[min(total - 1, idx + later_gap)]  # same camera, a few seconds on
        neg_out = OUT / "images" / f"{args.name}_{i}_neg.jpg"
        neg_later = OUT / "images" / f"{args.name}_{i}_neg_later.jpg"
        cv2.imwrite(str(neg_out), cur, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(neg_later), later, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"id": f"{args.name}_{i}_neg", "seq": args.name, "ref": str(ref_out), "cur": str(neg_out),
                        "later": str(neg_later), "has_spill": False, "gt_bbox": None, "kind": None,
                        "motion": True, "frame": idx})
        kind = rng.choice(list(KINDS))
        width = rng.uniform(0.04, 0.14)
        centre = fp(ref, cur, width)
        seed = rng.randint(0, 10**6)
        pos, bbox = composite_spill(cur, centre, width, kind, seed=seed)
        pos_later, _ = composite_spill(later, centre, width, kind, seed=seed)  # the spill is still there
        pos_out = OUT / "images" / f"{args.name}_{i}_pos.jpg"
        pos_later_out = OUT / "images" / f"{args.name}_{i}_pos_later.jpg"
        cv2.imwrite(str(pos_out), pos, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(pos_later_out), pos_later, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"id": f"{args.name}_{i}_pos", "seq": args.name, "ref": str(ref_out), "cur": str(pos_out),
                        "later": str(pos_later_out), "has_spill": True, "gt_bbox": bbox, "kind": kind,
                        "motion": True, "frame": idx, "spill_width_frac": round(width, 3)})
    for j in range(3):
        kind = rng.choice(list(KINDS))
        width = rng.uniform(0.04, 0.12)
        pos, bbox = composite_spill(ref, fp(ref, ref, width), width, kind, seed=rng.randint(0, 10**6))
        p = OUT / "images" / f"{args.name}_static{j}_pos.jpg"
        cv2.imwrite(str(p), pos, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"id": f"{args.name}_static{j}_pos", "seq": args.name, "ref": str(ref_out), "cur": str(p),
                        "later": str(p), "has_spill": True, "gt_bbox": bbox, "kind": kind, "motion": False,
                        "spill_width_frac": round(width, 3)})
    records.append({"id": f"{args.name}_static_neg", "seq": args.name, "ref": str(ref_out), "cur": str(ref_out),
                    "later": str(ref_out), "has_spill": False, "gt_bbox": None, "kind": None, "motion": False})

    with (OUT / "pairs.jsonl").open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"{args.video}: {total} frames at {fps:.0f} fps -> {len(records)} pairs appended to {OUT / 'pairs.jsonl'}")


if __name__ == "__main__":
    main()
