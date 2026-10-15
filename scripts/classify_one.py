"""Classify one or more images and print the JSON result.

Usage:
    python scripts/classify_one.py photo.jpg
    python scripts/classify_one.py a.jpg b.jpg --effort medium
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spilldetect import SpillDetector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    args = ap.parse_args()

    det = SpillDetector(effort=args.effort)
    for img in args.images:
        d = det.classify(img)
        print(f"\n== {img}  ({d.image_size[0]}x{d.image_size[1]}, {d.latency_s:.1f}s, served by {d.served_by})")
        if d.error:
            print(f"ERROR: {d.error}")
            if d.raw_text:
                print(d.raw_text)
        else:
            print(json.dumps(d.result.model_dump(), indent=2))
    u = det.usage
    print(f"\nusage: {u.requests} req, {u.input_tokens} in, {u.output_tokens} out, "
          f"{u.cache_read_tokens} cache-read, est ${u.cost_usd:.4f}")


if __name__ == "__main__":
    main()
