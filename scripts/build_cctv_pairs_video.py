"""Build before/after pairs from a fixed-camera video (CCTV, walkway, corridor, atrium).

    python scripts/build_cctv_pairs_video.py clip.mp4 --name atrium --pairs 35 --floor-poly "x,y x,y ..."
    python scripts/build_cctv_pairs_video.py clip.mp4 --name atrium --dry-run      # reference + drift only

Reference = per-pixel median of frames from the first seconds of the clip, which
removes passers-by and leaves the empty floor (the same trick a camera server
would use). Current frames are sampled later in the clip, so people are mid-walk.

Every frame is registered to the first frame with a homography (ORB features on
the whole frame, RANSAC) before use, so tripod drift or a slow pan does not show
up as "change". The residual drift and the overlap are reported; clips whose
camera really moves (overlap below --min-overlap) are rejected.

  negative pair : reference vs sampled frame (real motion only)
  positive pair : same, plus a synthetic spill composited onto open floor
  static pairs  : reference vs reference (+ spill) for the no-motion case
Each pair also stores a "later" frame a few seconds on, for the persistence check.

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


def _resize(f: np.ndarray, max_side: int) -> np.ndarray:
    h, w = f.shape[:2]
    s = min(1.0, max_side / max(h, w))
    return f if s == 1.0 else cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


class Registrar:
    """Aligns frames to an anchor frame with a homography. Returns (warped, overlap_fraction, shift_px)."""

    def __init__(self, anchor: np.ndarray):
        self.h, self.w = anchor.shape[:2]
        self.orb = cv2.ORB_create(3000)
        g = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)
        self.kp, self.des = self.orb.detectAndCompute(g, None)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(g, None)
        if des is None or self.des is None or len(kp) < 30:
            return frame, 1.0, 0.0
        matches = sorted(self.bf.match(des, self.des), key=lambda m: m.distance)[:800]
        if len(matches) < 30:
            return frame, 1.0, 0.0
        src = np.float32([kp[m.queryIdx].pt for m in matches])
        dst = np.float32([self.kp[m.trainIdx].pt for m in matches])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or inl is None or inl.sum() < 25:
            return frame, 1.0, 0.0
        corners = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]]).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        shift = float(np.abs(moved - corners.reshape(-1, 2)).mean())
        warped = cv2.warpPerspective(frame, H, (self.w, self.h), borderMode=cv2.BORDER_REPLICATE)
        # overlap: fraction of the anchor covered by the warped frame
        mask = cv2.warpPerspective(np.full((self.h, self.w), 255, np.uint8), H, (self.w, self.h))
        return warped, float((mask > 0).mean()), shift


def read_frames(path: str, wanted: set[int], max_side: int) -> dict[int, np.ndarray]:
    """Single sequential pass keeping only the wanted indices (memory-bounded, seek-free)."""
    cap = cv2.VideoCapture(path)
    out = {}
    i = 0
    last = max(wanted)
    while i <= last:
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = _resize(f, max_side)
        i += 1
    cap.release()
    return out


def count_frames(path: str) -> tuple[int, float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 20:  # unreliable container (MPEG-2): count by reading
        n = 0
        while cap.read()[0]:
            n += 1
    cap.release()
    return n, fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--name", required=True)
    ap.add_argument("--pairs", type=int, default=30, help="sampled current frames (each gives a pos and a neg)")
    ap.add_argument("--bg-seconds", type=float, default=6.0)
    ap.add_argument("--later-seconds", type=float, default=2.0)
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor-y", type=float, nargs=2, default=(0.3, 0.98))
    ap.add_argument("--floor-poly", default=None, help="floor zone 'x,y x,y ...' in frame fractions")
    ap.add_argument("--spill-width", type=float, nargs=2, default=(0.03, 0.12),
                    help="spill width range as a fraction of frame width")
    ap.add_argument("--min-overlap", type=float, default=0.85, help="reject clip if registration overlap is lower")
    ap.add_argument("--max-sat", type=float, default=70, help="floor picker: max mean HSV saturation of a floor patch")
    ap.add_argument("--max-edge", type=float, default=0.08, help="floor picker: max Canny edge density of a floor patch")
    ap.add_argument("--no-stabilize", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the reference + a gridded copy, report drift, no pairs")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total, fps = count_frames(args.video)
    if total < 20:
        sys.exit(f"only {total} decodable frames in {args.video}")
    n_bg = min(total // 2, int(args.bg_seconds * fps))
    bg_idx = list(range(0, n_bg, max(1, n_bg // 40)))
    start = n_bg + 1
    sample_idx = sorted(rng.sample(range(start, total), min(args.pairs, total - start)))
    gap = int(args.later_seconds * fps)
    later_idx = [min(total - 1, i + gap) for i in sample_idx]
    wanted = set(bg_idx) | set(sample_idx) | set(later_idx) | {0}
    frames = read_frames(args.video, wanted, args.max_side)
    anchor = frames[0]

    reg = None if args.no_stabilize else Registrar(anchor)
    overlaps, shifts = [], []

    def aligned(i: int) -> np.ndarray:
        f = frames[i]
        if reg is None:
            return f
        w, ov, sh = reg.align(f)
        overlaps.append(ov)
        shifts.append(sh)
        return w

    ref = np.median(np.stack([aligned(i) for i in bg_idx]), axis=0).astype(np.uint8)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    ref_out = OUT / "images" / f"{args.name}_ref.jpg"
    cv2.imwrite(str(ref_out), ref, [cv2.IMWRITE_JPEG_QUALITY, 92])

    cur_frames = {i: aligned(i) for i in sample_idx}
    later_frames = {i: aligned(i) for i in later_idx}
    H, W = ref.shape[:2]
    drift = (f"registration: mean shift {np.mean(shifts):.1f}px, max {np.max(shifts):.1f}px, "
             f"min overlap {np.min(overlaps):.2f}") if shifts else "registration: off"
    print(f"{args.video}: {total} frames at {fps:.0f} fps, {W}x{H}; {drift}")
    if shifts and np.min(overlaps) < args.min_overlap:
        print(f"  REJECTED: camera moves too much (overlap {np.min(overlaps):.2f} < {args.min_overlap})")
        return

    if args.dry_run:
        grid = ref.copy()
        for k in range(1, 10):
            x, y = int(W * k / 10), int(H * k / 10)
            cv2.line(grid, (x, 0), (x, H), (0, 255, 255), 1)
            cv2.line(grid, (0, y), (W, y), (0, 255, 255), 1)
            cv2.putText(grid, f"{k/10:.1f}", (x + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(grid, f"{k/10:.1f}", (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(OUT / "images" / f"{args.name}_ref_grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  wrote {ref_out} and a gridded copy for tracing the floor polygon")
        return

    poly = None
    if args.floor_poly:
        poly = np.array([[float(v) for v in pt.split(",")] for pt in args.floor_poly.split()], np.float32)

    def inside(x, y, w):
        lo, hi = args.floor_y
        if not (lo <= y <= hi):
            return False
        return poly is None or cv2.pointPolygonTest(poly, (float(x), float(y)), True) > 0.6 * w

    def fp(r, c, w):
        for _ in range(150):
            x, y = floor_point(r, c, rng, w, args.max_sat, args.max_edge)
            if inside(x, y, w):
                return x, y
        return None

    records = []
    skipped = 0
    for i, idx in enumerate(sample_idx):
        cur = cur_frames[idx]
        later = later_frames[later_idx[i]]
        neg_out = OUT / "images" / f"{args.name}_{i}_neg.jpg"
        neg_later = OUT / "images" / f"{args.name}_{i}_neg_later.jpg"
        cv2.imwrite(str(neg_out), cur, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(neg_later), later, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"id": f"{args.name}_{i}_neg", "seq": args.name, "ref": str(ref_out), "cur": str(neg_out),
                        "later": str(neg_later), "has_spill": False, "gt_bbox": None, "kind": None,
                        "motion": True, "frame": idx, "res": [W, H]})
        kind = rng.choice(list(KINDS))
        width = rng.uniform(*args.spill_width)
        centre = fp(ref, cur, width)
        if centre is None:
            skipped += 1
            continue
        seed = rng.randint(0, 10**6)
        pos, bbox = composite_spill(cur, centre, width, kind, seed=seed)
        pos_later, _ = composite_spill(later, centre, width, kind, seed=seed)
        pos_out = OUT / "images" / f"{args.name}_{i}_pos.jpg"
        pos_later_out = OUT / "images" / f"{args.name}_{i}_pos_later.jpg"
        cv2.imwrite(str(pos_out), pos, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(pos_later_out), pos_later, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"id": f"{args.name}_{i}_pos", "seq": args.name, "ref": str(ref_out), "cur": str(pos_out),
                        "later": str(pos_later_out), "has_spill": True, "gt_bbox": bbox, "kind": kind,
                        "motion": True, "frame": idx, "spill_width_frac": round(width, 3), "res": [W, H]})
    for j in range(2):
        kind = rng.choice(list(KINDS))
        width = rng.uniform(*args.spill_width)
        centre = fp(ref, ref, width)
        if centre is None:
            continue
        pos, bbox = composite_spill(ref, centre, width, kind, seed=rng.randint(0, 10**6))
        p = OUT / "images" / f"{args.name}_static{j}_pos.jpg"
        cv2.imwrite(str(p), pos, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"id": f"{args.name}_static{j}_pos", "seq": args.name, "ref": str(ref_out), "cur": str(p),
                        "later": str(p), "has_spill": True, "gt_bbox": bbox, "kind": kind, "motion": False,
                        "spill_width_frac": round(width, 3), "res": [W, H]})
    records.append({"id": f"{args.name}_static_neg", "seq": args.name, "ref": str(ref_out), "cur": str(ref_out),
                    "later": str(ref_out), "has_spill": False, "gt_bbox": None, "kind": None, "motion": False,
                    "res": [W, H]})

    with (OUT / "pairs.jsonl").open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  {len(records)} pairs appended ({sum(r['has_spill'] for r in records)} with spill, "
          f"{skipped} positives skipped: no open floor found)")


if __name__ == "__main__":
    main()
