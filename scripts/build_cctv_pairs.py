"""Build a fixed-camera evaluation set of (reference, current) frame pairs.

    python scripts/build_cctv_pairs.py            # -> data/cctv_pairs/

Source: the fixed-camera sequences in the Roboflow Spill_Detection train split
(classroom fall clips video0-4, the fall-NN-cam0 clips, the shop CCTV clip, the gym
and standing clips). None of them contain a spill, so:

  negative pair : reference = first frame of the clip, current = a later frame.
                  Real people moving, falling, chairs shifted, lighting flicker.
  positive pair : same as above, plus a synthetic spill composited onto the floor
                  of the current frame (spilldetect.synth). Exact bbox ground truth.

Both kinds also come in a "static" variant where current == reference (plus the
spill for positives), i.e. no motion at all. Output: pairs.jsonl + images/.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spilldetect.cctv import change_mask  # noqa: E402
from spilldetect.synth import KINDS, composite_spill  # noqa: E402

SRC = Path("data/raw/spill_detection/train/images")
OUT = Path("data/cctv_pairs")
# video1_edited must come before video[0-4] or the alternation swallows it as "video1".
# "standing" is a person posing on a white backdrop, not a floor camera, so it is left out.
SEQ_RE = re.compile(r"^(video1_edited_mp4|video[0-4]|Shoplifting054_x264|split11|fall-\d+-cam\d)")
PER_SEQ = 6          # current frames sampled per sequence
SEED = 0


def _thumb(p: Path):
    """Brightness-normalised thumbnail of the UPPER 45% of the frame.

    Walls, shelves and windows identify a camera view; the floor below is where
    people fall and move, so it is left out of the similarity test.
    """
    g = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
    g = g[: int(g.shape[0] * 0.45)]
    g = cv2.resize(g, (64, 24)).astype("float32")
    return (g - g.mean()) / (g.std() + 1e-6)


def sequences() -> dict[str, list[Path]]:
    """Group frames by clip, then keep only frames that genuinely share the camera view.

    Roboflow train splits contain augmented copies (crops, zooms, flips, colour shifts),
    so two frames with the same clip name can be different framings. We pick the frame
    most similar to all others as the reference and keep frames whose thumbnail differs
    from it by less than MAX_DIFF on average; the reference is returned first.
    """
    MAX_DIFF = 0.38  # mean abs difference of normalised thumbnails; different rooms score 0.45+
    seqs: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(SRC.iterdir()):
        m = SEQ_RE.match(p.name)
        if m:
            seqs[m.group(1)].append(p)
    out: dict[str, list[Path]] = {}
    for name, frames in seqs.items():
        if len(frames) < 6:
            continue
        th = [_thumb(p) for p in frames]
        n = len(th)
        d = [[abs(th[i] - th[j]).mean() for j in range(n)] for i in range(n)]
        ref_i = min(range(n), key=lambda i: sum(d[i]))
        keep = [frames[j] for j in sorted(range(n), key=lambda j: d[ref_i][j]) if d[ref_i][j] < MAX_DIFF]
        if len(keep) >= 4:
            out[name] = keep  # keep[0] is the reference (distance 0 to itself)
        print(f"  {name:22} {n:3} frames, {len(keep):3} share the camera view")
    return out


def floor_point(ref, cur, rng: random.Random, width_frac: float = 0.1,
                max_sat: float = 70, max_edge: float = 0.08) -> tuple[float, float]:
    """Pick a spot in the lower part of the frame that looks like open floor.

    Open floor: unchanged between ref and cur, not a letterbox border, some texture,
    and few edges (tables with objects, shelves and people are edge-dense). The
    whole footprint of the future spill must satisfy this, not just its centre.
    """
    H, W = cur.shape[:2]
    moved = change_mask(ref, cur)
    grey = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)[:, :, 1]
    edges = cv2.Canny(grey, 60, 160)
    r = max(6, int(width_frac * W * 0.6))
    best, best_score = (0.5, 0.8), 1e9
    for _ in range(200):
        x, y = rng.uniform(0.1, 0.9), rng.uniform(0.25, 0.93)
        px, py = int(x * W), int(y * H)
        sl = (slice(max(0, py - r), py + r), slice(max(0, px - r), px + r))
        patch = grey[sl]
        if patch.size == 0 or moved[sl].mean() > 6 or not (40 < patch.mean() < 225) or patch.std() < 4:
            continue
        if sat[sl].mean() > max_sat:  # grass, carpets, painted surfaces: not the stone/tile floor we want
            continue
        edge_density = edges[sl].mean() / 255
        if edge_density > max_edge:
            continue
        score = edge_density * 10 + abs(y - 0.75) + rng.uniform(0, 0.6)  # smooth, lowish, with variety
        if score < best_score:
            best, best_score = (x, y), score
        if best_score < 0.25:
            break
    return best


def main() -> None:
    rng = random.Random(SEED)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    records = []
    seqs = sequences()
    print(f"{len(seqs)} fixed-camera sequences")
    for name, frames in sorted(seqs.items()):
        ref_path = frames[0]
        ref = cv2.imread(str(ref_path))
        ref_out = OUT / "images" / f"{name}_ref.jpg"
        cv2.imwrite(str(ref_out), ref, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n_take = min(PER_SEQ, len(frames) - 1)
        picks = sorted(rng.sample(frames[1:], n_take))
        for i, cur_path in enumerate(picks):
            cur = cv2.imread(str(cur_path))
            if cur.shape != ref.shape:
                cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]))
            # negative: real motion only
            neg_out = OUT / "images" / f"{name}_{i}_neg.jpg"
            cv2.imwrite(str(neg_out), cur, [cv2.IMWRITE_JPEG_QUALITY, 92])
            records.append({"id": f"{name}_{i}_neg", "seq": name, "ref": str(ref_out), "cur": str(neg_out),
                            "has_spill": False, "gt_bbox": None, "kind": None, "motion": True})
            # positive: same frame + synthetic spill on an unchanged floor spot
            kind = rng.choice(list(KINDS))
            width = rng.uniform(0.05, 0.16)
            pos, bbox = composite_spill(cur, floor_point(ref, cur, rng, width), width, kind, seed=rng.randint(0, 10**6))
            pos_out = OUT / "images" / f"{name}_{i}_pos.jpg"
            cv2.imwrite(str(pos_out), pos, [cv2.IMWRITE_JPEG_QUALITY, 92])
            records.append({"id": f"{name}_{i}_pos", "seq": name, "ref": str(ref_out), "cur": str(pos_out),
                            "has_spill": True, "gt_bbox": bbox, "kind": kind, "motion": True,
                            "spill_width_frac": round(width, 3)})
        # static variants: no motion, spill only (one per sequence) and identical frames (one per sequence)
        kind = rng.choice(list(KINDS))
        width = rng.uniform(0.04, 0.12)
        pos, bbox = composite_spill(ref, floor_point(ref, ref, rng, width), width, kind, seed=rng.randint(0, 10**6))
        p = OUT / "images" / f"{name}_static_pos.jpg"
        cv2.imwrite(str(p), pos, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"id": f"{name}_static_pos", "seq": name, "ref": str(ref_out), "cur": str(p),
                        "has_spill": True, "gt_bbox": bbox, "kind": kind, "motion": False,
                        "spill_width_frac": round(width, 3)})
        records.append({"id": f"{name}_static_neg", "seq": name, "ref": str(ref_out), "cur": str(ref_out),
                        "has_spill": False, "gt_bbox": None, "kind": None, "motion": False})

    with (OUT / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    pos = sum(r["has_spill"] for r in records)
    print(f"wrote {len(records)} pairs ({pos} with spill, {len(records) - pos} without) to {OUT}")


if __name__ == "__main__":
    main()
