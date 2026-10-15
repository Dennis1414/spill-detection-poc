"""Extract frames from a video at a fixed rate, skipping near-duplicate frames.

Usage:
    python scripts/sample_frames.py path/to/video.mp4 data/frames/my_clip --fps 1
    python scripts/sample_frames.py path/to/folder_of_videos data/frames --fps 2

Robots don't need 30 fps for spill detection; 1 to 2 fps is plenty for a PoC
and keeps the API bill small. --min-diff drops frames that barely changed
(robot stopped, static scene) so you don't pay to classify the same view twice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference on a small greyscale thumbnail, 0..255."""
    sa = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (64, 36))
    sb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (64, 36))
    return float(np.abs(sa.astype(np.int16) - sb.astype(np.int16)).mean())


def sample_video(video: Path, out_dir: Path, fps: float, min_diff: float, max_side: int) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  cannot open {video}", file=sys.stderr)
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = saved = 0
    last_kept = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if last_kept is None or frame_diff(last_kept, frame) >= min_diff:
                h, w = frame.shape[:2]
                scale = min(1.0, max_side / max(h, w))
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                t = idx / src_fps
                name = f"{video.stem}_t{t:07.2f}s.jpg"
                cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                last_kept = frame
                saved += 1
        idx += 1
    cap.release()
    print(f"  {video.name}: {idx} frames read, {saved} saved -> {out_dir}")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="video file or folder of videos")
    ap.add_argument("out", help="output folder for frames")
    ap.add_argument("--fps", type=float, default=1.0, help="frames per second to keep (default 1)")
    ap.add_argument("--min-diff", type=float, default=3.0,
                    help="skip a frame if it differs from the last kept one by less than this (0 = keep all)")
    ap.add_argument("--max-side", type=int, default=1280, help="downscale so the long side is at most this")
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        sys.exit(f"no videos found at {src}")
    total = 0
    for v in videos:
        total += sample_video(v, out / v.stem if src.is_dir() else out, args.fps, args.min_diff, args.max_side)
    print(f"saved {total} frames")


if __name__ == "__main__":
    main()
